#!/usr/bin/env python3
"""
Base class for LangGraph-based orchestrators

File: klea_utils/graph/base.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import contextvars
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, final

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import StreamTransformer
from langgraph.types import RunnableConfig
from mcp.types import Tool
from platformdirs import PlatformDirs
from pydantic import BaseModel, Field, create_model

from klea_utils.llm import LLMModel
from klea_utils.mcp.schemas import ToolInfo
from klea_utils.paths import get_config_dir, init_dir, resolve_app_config_path
from klea_utils.stores.config import RetrieverConfig
from klea_utils.stores.retrieval.bm25 import BM25RetrieverManager
from klea_utils.stores.retrieval.vs import VSRetriever
from klea_utils.tools import build_tool_description, clean_tool_meta

# Per-request context variable carrying per-session model overrides (api_key,
# model, provider, etc.).  Set by the API layer before graph.ainvoke() and
# read by _invoke_llm() so that nodes don't need to thread overrides through
# their signatures.  Falls back to an empty dict if not set.
model_overrides_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "model_overrides", default={}
)


class _CustomChannelEnabler(StreamTransformer):
    """Enables the ``custom`` channel in LangGraph v3 event streams.

    Non-LLM nodes use ``get_stream_writer()`` to emit progress events on the
    custom channel.  LangGraph requires a ``StreamTransformer`` declaring
    ``required_stream_modes = ("custom",)`` for that channel to be enabled.
    This no-op transformer satisfies that requirement.
    """

    required_stream_modes = ("custom",)

    def __init__(self, scope=()):
        super().__init__(scope)

    def init(self):
        return {}

    def process(self, event):
        return True


class BaseLangGraph(ABC):
    """Abstract base class for LangGraph-based orchestrators.

    Provides common infrastructure for:
    - Configuration loading from env files
    - MCP client creation from JSON config
    - LLM model setup (delegated to subclasses)
    - LangGraph compilation and execution
    - Session checkpointing
    - Dual-stream logging

    Subclasses must implement:
    - :meth:`_setup_models`: Create LLM model instances
    - :meth:`_create_graph`: Build and compile the LangGraph
    - Set ``env_class`` to the appropriate Pydantic settings class
    """

    #: Pydantic BaseSettings class for env loading.
    #: Subclasses must set this to their AppEnv class.
    env_class: type[BaseModel]

    #: Pydantic BaseModel class for configuration loading.
    #: Subclasses must set this to their AppConfig class.
    config_class: type[BaseModel]

    #: Name of the environment variable that controls the env file path.
    env_var: str = "ENV_FILE"

    #: Default config file name if the environment variable is not set.
    env_file_default: str = "config.env"

    #: Logger name for this orchestrator, also used as the app name
    #: for ``platformdirs`` data/cache directories.
    graph_name: str = "BaseLangGraph"

    def __init__(
        self,
        logging_level: int = logging.DEBUG,
        checkpoint: str = "inmemory",
        log_file: bool = True,
    ):
        """Initialise the base orchestrator.

        :param logging_level: Logging level for the orchestrator
        :param checkpoint: Checkpointer mode  ---  ``"inmemory"`` (volatile, default),
            ``"sqlite"`` (persistent via ``self.paths.user_data_dir``),
            or ``"none"`` (no checkpointing).  When set to ``"none"``, nodes
            that need conversation history receive ``memory=False``.
        :param log_file: When ``True`` (default), configure the process-wide
            root logger with a rotating file handler writing to
            ``{self.paths.user_data_dir}/{self.graph_name}.log``, alongside
            the checkpoints and session database.  Set to ``False`` in
            short-lived processes (e.g. tests) to avoid writing log files.
        """
        self.env_file = os.getenv(self.env_var, self.env_file_default)
        self.app_env: BaseModel

        # Graph-level default models per role.  Per-request model
        # overrides are merged at invoke time via ``model_overrides_ctx``
        # and do NOT change this dict.
        self.llm_models: dict[str, LLMModel] = {}

        self.tools_info: dict[str, dict[str, ToolInfo]] = {}
        self.domain_mcp_configs: dict[str, MCPConfig] = {}
        self.checkpointer_mode = checkpoint
        self.memory = checkpoint != "none"
        self.checkpointer = None

        self.paths = PlatformDirs(self.graph_name.lower())

        self.config_dict: dict[str, Any]

        self.graph: CompiledStateGraph | None = None

        self.mcp_config: MCPConfig | None = None
        self.mcp_client: Client | None = None
        self.mcp_tools: list[Tool] | None = None

        self.retriever_config: RetrieverConfig | None = None
        self.stores: VSRetriever | None = None
        self.bm25_stores: BM25RetrieverManager | None = None
        # Graph-wide fallback retrieval settings.  Individual vector stores
        # may override these in the config with their own default_k / k_max /
        # k_inc values.
        self.default_k: int = 5
        self.k_max: int = 10
        self.k_inc: int = 1

        self.QueryDomainSchema: type[BaseModel] | None = None

        from klea_utils.plogging import setup_root_logger

        setup_root_logger(
            self.graph_name,
            stderr_level=logging_level,
            log_dir=self.paths.user_data_dir if log_file else None,
        )
        self.logger = logging.getLogger(self.graph_name)

    def _load_env(self) -> None:
        """Load env file, and configuration

        Uses ``self.env_class`` and ``self.env_file`` to locate and parse
        the env file, then resolves the application config file (from
        ``self.app_env.app_config_file``) via
        :func:`klea_utils.paths.resolve_app_config_path` -- the working
        directory first, then the per-app config directory.  Raises
        ``FileNotFoundError`` if either file does not exist.
        """
        env_file_path = Path(self.env_file)
        if not env_file_path.exists():
            raise FileNotFoundError(
                f"""Could not find env file: {self.env_file}. You can use the {self.env_var} environment variable to specify the env file."""
            )

        self.app_env = self.env_class(_env_file=self.env_file)
        assert self.app_env
        self.logger.debug(f"env file: {self.env_file}")
        self.logger.debug(f"env: {self.app_env}")

        if "app_config_file" not in self.env_class.model_fields:
            raise FileNotFoundError(
                f"No config file provided. Please provide one in the env file ({self.env_file})."
                + f"You can use the {self.env_var} environment variable to specify the env file."
            )

        # An explicit empty value means "not set" -- fall back to the field
        # default so a line like ``KLEA_AGENT_APP_CONFIG_FILE=`` does not
        # resolve to the current directory itself.
        app_config_file = self.app_env.app_config_file
        if not app_config_file:
            app_config_file = self.env_class.model_fields["app_config_file"].default
            self.logger.debug(
                f"empty app_config_file -- using default {app_config_file}"
            )

        config_file = resolve_app_config_path(
            app_config_file, get_config_dir(self.paths)
        )
        self.logger.debug(f"config file: {config_file}")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            self.logger.debug(f"{config_dict = }")
            self.app_config = self.config_class(**config_dict)
            self.logger.debug(f"{self.app_config = }")

    def _create_mcp_client(self) -> None:
        """Create MCP client from the JSON config file.

        Reads the MCP server configurations from ``self.app_env.mcp_config_file``
        and creates a ``fastmcp.Client`` instance.
        """
        if self.mcp_config and self.mcp_config.mcpServers:
            self.logger.debug(f"{self.mcp_config = }")
            self.mcp_client = Client(self.mcp_config)
        else:
            self.logger.warning("No MCP server configured.")
            self.mcp_client = None

    async def _get_mcp_tools(self) -> None:
        """Get MCP tools."""
        if self.mcp_client:
            async with self.mcp_client:
                self.mcp_tools = await self.mcp_client.list_tools()
            self.logger.debug(f"{self.mcp_tools =}")
            self._build_tools_info()

    def _build_tools_info(self) -> None:
        """Build per-domain tool metadata from fetched MCP tools."""
        self.tools_info = {}
        if not self.mcp_tools or not self.domain_mcp_configs:
            return

        # map server names to domains
        domain_servers: dict[str, list[str]] = {}
        num_servers = 0
        for domain, config in self.domain_mcp_configs.items():
            if config.mcpServers:
                domain_servers[domain] = list(config.mcpServers.keys())
                num_servers += len(list(config.mcpServers.keys()))

        for domain, server_names in domain_servers.items():
            domain_tools_info: dict[str, ToolInfo] = {}
            for t in self.mcp_tools:
                if "dummy" in t.name:
                    continue
                # tools will be prefixed with server names
                if num_servers > 1 and not any(
                    t.name.startswith(s + "_") for s in server_names
                ):
                    continue
                # otherwise, there's only one server
                # Klea expects MCP tools to follow the docstring-first
                # convention (summary + Use when / Do not use for bullets +
                # one example; params via Args:), see build_tool_description
                # and docs/concepts/mcp.rst.
                domain_tools_info[t.name] = ToolInfo(
                    title=t.title,
                    description=build_tool_description(t),
                    meta=clean_tool_meta(t.meta),
                )
            self.tools_info[domain] = domain_tools_info
        self.logger.debug(f"{self.tools_info = }")

    async def _get_vector_stores(self) -> None:
        """Get vector stores"""
        emb = self.llm_models.get("embedding")
        if self.retriever_config and emb and emb.model_name:
            self.stores = VSRetriever(
                config=self.retriever_config,
                logger=self.logger,
                embedding_model=emb.model_name,
                default_k=self.default_k,
                k_max=self.k_max,
                k_inc=self.k_inc,
            )
            self.stores.setup()
            self.logger.info(f"Vector stores loaded: {self.stores.domains}")

            # dynamically generate schema for domains
            all_domains = self.stores.domains.copy()
            all_domains.append("undefined")

            self.QueryDomainSchema = create_model(
                "QueryDomainSchema",
                query_domains=(
                    list[Literal[tuple(all_domains)]],
                    Field(default=["undefined"], validate_default=True),
                ),
            )
        else:
            self.logger.warning("No vector stores configured.")

        # BM25 keyword stores need no embedding model, so build them whenever
        # any domain configures bm25_stores.
        if self.retriever_config and any(
            domain.bm25_stores for domain in self.retriever_config.domains.values()
        ):
            self.bm25_stores = BM25RetrieverManager(
                config=self.retriever_config,
                logger=self.logger,
                default_k=self.default_k,
                k_max=self.k_max,
                k_inc=self.k_inc,
            )
            self.logger.info(f"BM25 stores loaded: {self.bm25_stores.domains}")

    def _export_graph_png(self, filename: str) -> None:
        """Export the LangGraph as a Mermaid PNG diagram.

        Skipped when running inside Docker (``RUNNING_IN_DOCKER`` env var set).

        :param filename: Output file path for the PNG
        """
        if os.environ.get("RUNNING_IN_DOCKER", 0):
            return
        try:
            assert self.graph
            self.graph.get_graph().draw_mermaid_png(output_file_path=filename)
        except BaseException as e:
            self.logger.error("Something went wrong generating lang graph png")
            self.logger.error(e)

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _configure_resources(self) -> None:
        """Configure vector stores and MCP servers

        Subclasses should implement this to populate ``self.retriever_config``,
        ``self.mcp_config``, and ``self.domain_mcp_configs``, which will be used
        to create the vector store class, mcp client, and per-domain tool descriptions.
        """
        ...

    @abstractmethod
    def _setup_models(self) -> None:
        """Set up LLM model instances.

        Subclasses should populate ``self.llm_models`` with ``LLMModel``
        entries keyed by role (e.g. ``"chat"``, ``"plan"``, ``"guard"``).
        These are graph-wide defaults; per-request overrides are applied
        at runtime via ``model_overrides_ctx``.
        """
        ...

    def _provider_defaults_for_role(self, role: str) -> dict[str, dict[str, Any]]:
        """Return per-provider default params for *role* from the config.

        Reads the optional ``providers`` config section, e.g.::

            {"huggingface": {"chat": {"max_output_tokens": 2048}}}

        and returns just the entries relevant to *role*, so each
        ``LLMModel`` carries the defaults for its own role.  Shared by all
        graphs so ``_setup_models`` implementations do not repeat it.

        :param role: Model role (``chat``, ``plan``, ``guard``, ...).
        :returns: ``{provider: {param: value}}`` for *role*.
        """
        result: dict[str, dict[str, Any]] = {}
        # ``app_config`` is the concrete ``config_class`` instance; access
        # via getattr so this works for any subclass without extra typing.
        providers = getattr(self.app_config, "providers", {})
        for provider, role_configs in providers.items():
            if role in role_configs:
                result[provider] = dict(role_configs[role])
        return result

    @abstractmethod
    async def _create_graph(self) -> None:
        """Build and compile the LangGraph, storing it in ``self.graph``.

        This is where subclasses define their nodes, edges, and conditional
        routing logic.
        """
        ...

    # ------------------------------------------------------------------
    # Hook methods -- override for pre/post setup work
    # ------------------------------------------------------------------

    def _pre_setup(self) -> None:
        """Hook called before the standard setup sequence.

        Override to perform subclass-specific initialisation before
        config loading and model setup.
        """
        pass

    async def _setup_checkpointer(self) -> None:
        """Set up the checkpointer.

        ``setup()`` calls this hook before ``_load_env()``.
        The checkpointer is ``None`` when ``checkpoint="none"``.
        """
        if self.checkpointer_mode == "sqlite":
            import aiosqlite

            db_path = init_dir(self.paths.user_data_dir) / "checkpoints.db"
            self.logger.debug("Opening sqlite checkpointer at %s", db_path)
            conn = await aiosqlite.connect(str(db_path))
            self.checkpointer = AsyncSqliteSaver(conn)
            self.logger.debug("Sqlite checkpointer ready")
        elif self.checkpointer_mode == "inmemory":
            self.checkpointer = InMemorySaver()
            self.logger.debug("In-memory checkpointer ready")

    def _post_setup(self) -> None:
        """Hook called after the standard setup sequence.

        Override to perform subclass-specific finalisation after
        the LangGraph has been compiled.
        """
        pass

    # ------------------------------------------------------------------
    # Template method
    # ------------------------------------------------------------------

    async def _pre_graph(self) -> None:
        """Hook called after MCP client setup but before graph compilation.

        Override to perform subclass-specific initialisation that depends
        on config and MCP client but must happen         before the LangGraph is built.
        """
        pass

    @final
    async def setup(self) -> None:
        """Set up the orchestrator.

        Calls hooks and template methods in this order:
        1. ``_pre_setup()``
        2. ``_setup_checkpointer()``
        3. ``_load_env()``
        4. ``_setup_models()``
        5. ``_create_mcp_client()``
        6. ``_pre_graph()``
        7. ``_create_graph()``
        8. ``_post_setup()``
        """
        self._pre_setup()
        await self._setup_checkpointer()
        self._load_env()
        self._configure_resources()
        self._setup_models()
        self._create_mcp_client()
        await self._get_mcp_tools()
        await self._get_vector_stores()
        await self._pre_graph()
        await self._create_graph()
        self._post_setup()

    # ------------------------------------------------------------------
    # Execution methods -- identical across all implementations
    # ------------------------------------------------------------------

    async def run_graph_invoke_state(
        self, state: dict, thread_id: str = "default_thread"
    ) -> dict:
        """Run the graph, accepting and returning full state dicts.

        :param state: Initial graph state (must contain ``query`` key)
        :param thread_id: Session/thread identifier for checkpointing
        :returns: Final graph state
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)

        final_state = await self.graph.ainvoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    # TODO: fields to be extracted from the final state to be returned should
    # be configurable with a schema
    async def run_graph_invoke(
        self, query: str, thread_id: str = "default_thread"
    ) -> str:
        """Run the graph with a simple string query.

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :returns: The ``message_for_user`` field from the final state
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        final_state = await self.graph.ainvoke({"query": query}, config=config)

        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
            return message
        else:
            return "I was unable to answer"

    async def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph and yield intermediate ``message_for_user`` values.

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :yields: ``message_for_user`` strings from each node
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        async for chunk in self.graph.astream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {state!r}")
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_thread") -> Any:
        """Run the graph and return the raw astream result.

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :returns: Raw async generator from ``graph.astream()``
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res

    async def run_graph_astream_events(
        self, query: str, thread_id: str = "default_thread"
    ):
        """Run the graph and yield structured streaming events.

        Yields dicts with:

        ``{"type": "progress", "node": "<label>"}``
            When the graph enters a new node (via ``write_custom_stream``)
        ``{"type": "info", "node": "<label>", "data": {...}}``
            Structured summary data from a node after execution
        ``{"type": "debug", "node": "<label>", "data": {...}}``
            Full data dump from a node after execution
        ``{"type": "token", "content": "<chunk>", "node": "<label>"}``
            LLM token chunk from the current node
        ``{"type": "usage", "node": "<label>", "data": {...}}``
            Per-node token usage (input / output / total tokens)
        ``{"type": "complete", "message_for_user": "<answer>"}``
            Final answer from the completed graph

        Uses LangGraph's ``astream_events`` v3 protocol.  Progress events
        from all nodes (LLM and non-LLM) arrive via the ``custom`` channel.
        LLM token output is read from the ``messages`` channel.
        A ``StreamTransformer`` enables the custom channel so those events
        flow through.

        Reference: https://docs.langchain.com/oss/python/langgraph/event-streaming

        :param query: User query string
        :param thread_id: Session/thread identifier for checkpointing
        :yields: Structured event dicts
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        assert self.graph, "Graph not compiled. Call setup() first."

        stream = await self.graph.astream_events(
            {"query": query},
            config=config,
            version="v3",
            transformers=[_CustomChannelEnabler],
        )

        current_node = ""
        node_start = time.monotonic()
        total_start = time.monotonic()
        last_values: dict = {}

        async for event in stream:
            method = event["method"]

            if method == "custom":
                data = event["params"]["data"]
                if not isinstance(data, dict) or not data.get("node"):
                    continue

                event_type = data.get("type")

                if event_type == "progress":
                    node = data["node"]
                    if node != current_node:
                        now = time.monotonic()
                        if current_node:
                            self.logger.debug(
                                "Node [%s] took %.2fs",
                                current_node,
                                now - node_start,
                            )
                        node_start = now
                        current_node = node
                        self.logger.debug(f"Progress: {current_node}")
                        yield {"type": "progress", "node": current_node}

                elif event_type in ("info", "debug", "state", "usage"):
                    data_out = data.get("data", {}).copy()
                    data_out["timing_seconds"] = round(time.monotonic() - node_start, 2)
                    yield {
                        "type": event_type,
                        "node": data["node"],
                        "data": data_out,
                    }

            elif method == "messages":
                data = event["params"]["data"]
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    if item.get("event") == "content-block-delta":
                        delta = item.get("delta", {})
                        if "text" in delta:
                            yield {
                                "type": "token",
                                "content": delta["text"],
                                "node": current_node,
                            }

            elif method == "values":
                last_values = event["params"]["data"]

        total_elapsed = time.monotonic() - total_start
        if current_node:
            self.logger.debug(
                "Node [%s] took %.2fs",
                current_node,
                time.monotonic() - node_start,
            )
        self.logger.info("Graph completed in %.2fs", total_elapsed)

        message = ""
        if last_values:
            if isinstance(last_values, dict):
                message = last_values.get("message_for_user", "")
            elif hasattr(last_values, "message_for_user"):
                message = last_values.message_for_user
        yield {"type": "complete", "message_for_user": message}
