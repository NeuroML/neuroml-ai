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
from textwrap import dedent
from typing import Any, Literal, cast, final

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.stream import StreamTransformer
from langgraph.types import RunnableConfig
from mcp.types import Tool
from platformdirs import PlatformDirs
from pydantic import BaseModel, create_model

from klea_utils.paths import init_dir
from klea_utils.plogging import setup_logger
from klea_utils.stores.config import VectorStoresConfig
from klea_utils.stores.retrieval import VSRetriever

_log = setup_logger(__name__)

# Per-request context variable carrying per-session model overrides (api_key,
# model, provider, etc.).  Set by the API layer before graph.ainvoke() and
# read by _invoke_llm() so that nodes don't need to thread overrides through
# their signatures.  Falls back to an empty dict if not set.
model_overrides_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "model_overrides", default={}
)


class LLMModel(BaseModel):
    """Container for a single LLM model instance and its runtime configuration.

    ``instance`` holds the model object (typically a ``_ConfigurableModel``
    returned by ``init_chat_model``).  ``role_defaults`` stores role-wide
    default parameters (e.g. ``max_tokens``, ``temperature``) that apply to
    every node sharing this role, unless overridden by node or user config.

    ``build_config()`` performs a four-layer merge:

    **Layer 0**  ---  ``role_defaults``: role-wide parameters (e.g.
    ``{"max_tokens": 4096}``).

    **Layer 1**  ---  ``model_name``: the default model identifier from
    the graph config.

    **Layer 2**  ---  ``context_overrides``: per-request fields from the API
    (``model``, ``api_key``, etc.).  Only applied when ``modifiable=True``,
    and skipping any keys frozen by node defaults.

    **Layer 3**  ---  ``node_defaults``: frozen per-node defaults (always win).

    ``modifiable`` controls whether the model can be changed at runtime
    (both the API and web UI reject modifications to locked roles).
    Set to ``False`` to lock a role (e.g. guard) against user overrides
    in managed deployments.
    """

    model_name: str = ""
    parsed_model: Any = None
    instance: Any
    role_defaults: dict[str, Any] = {}
    modifiable: bool = True

    def build_config(
        self,
        context_overrides: dict[str, Any] | None = None,
        node_defaults: dict[str, Any] | None = None,
    ) -> RunnableConfig:
        """Merge up to four layers of model configuration into a ``RunnableConfig``.

        Layer order (lowest -> highest priority):

        0. ``self.role_defaults``   ---   role-wide parameters
        1. ``self.model_name``      ---   role model from graph config
        2. ``context_overrides``    ---   per-request user overrides
        3. ``node_defaults``        ---   frozen per-node defaults

        :param context_overrides: Per-request fields from the API
            (e.g. ``model``, ``api_key``).  Only applied when
            ``self.modifiable is True``, and skipping any keys
            present in ``node_defaults``.
        :param node_defaults: Frozen per-node defaults (e.g.
            ``{"temperature": 0.3}``).  Always win.
        :returns: A ``RunnableConfig`` with the ``configurable`` key
            populated.
        """
        _log.debug(
            f"{self.modifiable = }\n"
            f"{self.role_defaults = }\n"
            f"{self.model_name = }\n"
            f"{context_overrides = }\n"
            f"{node_defaults = }"
        )

        # Layer 0: role-wide defaults from graph config
        overrides: dict[str, Any] = dict(self.role_defaults)
        _log.debug(f"Layer 0 (role defaults):\n{overrides = }")

        # Layer 1: role model identifier
        overrides["model"] = self.model_name
        _log.debug(f"Layer 1 (model):\n{overrides = }")

        # Layer 2: context overrides (only if modifiable).
        # Skip any keys the node has frozen in model_defaults.
        if self.modifiable and context_overrides:
            for k, v in context_overrides.items():
                if node_defaults and k in node_defaults:
                    _log.debug(
                        f"Skipping context override '{k}' (frozen by node defaults)"
                    )
                    continue
                overrides[k] = v
            _log.debug(f"Layer 2 (context):\n{overrides = }")

        # Layer 3: node defaults  ---  always win
        if node_defaults:
            overrides.update(node_defaults)
            _log.debug(f"Layer 3 (node defaults):\n{overrides = }")

        # Parse the final model string into LangChain-compatible components.
        # Klea stores full provider-prefixed model strings internally (e.g.
        # "custom:gpt-4o:https://endpoint/v1"), but the _ConfigurableModel
        # expects the bare model name plus separate model_provider and base_url.
        from klea_utils.llm import parse_model_name

        parsed = parse_model_name(overrides["model"])
        overrides["model"] = parsed.model_name
        if parsed.provider == "custom":
            overrides["model_provider"] = "openai"
        elif parsed.provider:
            overrides["model_provider"] = parsed.provider
        if parsed.suffix:
            overrides["base_url"] = parsed.suffix
        _log.debug(f"After model string parse:\n{overrides = }")

        # Map generic "api_key" to provider-specific token field names so a
        # single user-facing field works across all providers.
        if "api_key" in overrides:
            overrides.setdefault("huggingfacehub_api_token", overrides["api_key"])
            _log.debug(f"After api_key mapping:\n{overrides = }")

        # Wrap in the "configurable" key expected by _ConfigurableModel.
        return cast(RunnableConfig, {"configurable": overrides})


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
    ):
        """Initialise the base orchestrator.

        :param logging_level: Logging level for the orchestrator
        :param checkpoint: Checkpointer mode  ---  ``"inmemory"`` (volatile, default),
            ``"sqlite"`` (persistent via ``self.paths.user_data_dir``),
            or ``"none"`` (no checkpointing).  When set to ``"none"``, nodes
            that need conversation history receive ``memory=False``.
        """
        self.env_file = os.getenv(self.env_var, self.env_file_default)
        self.app_env: BaseModel

        # Graph-level default models per role.  Per-request model
        # overrides are merged at invoke time via ``model_overrides_ctx``
        # and do NOT change this dict.
        self.llm_models: dict[str, LLMModel] = {}

        self.tools_description: dict[str, str] = {}
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

        self.stores_config: VectorStoresConfig | None = None
        self.stores: VSRetriever | None = None
        self.default_k: int = 5
        self.k_max: int = 10

        self.QueryDomainSchema: type[BaseModel] | None = None

        from klea_utils.plogging import setup_logger

        self.logger = setup_logger(self.graph_name, stderr_level=logging_level)

    def _load_env(self) -> None:
        """Load env file, and configuration

        Uses ``self.env_class`` and ``self.env_file`` to locate and parse
        the configuration file. Raises FileNotFoundError if the file does not exist.
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

        if "app_config_file" in self.env_class.model_fields:
            config_file = Path(self.app_env.app_config_file)
            if not config_file.exists():
                raise FileNotFoundError(f"Could not find config file: {config_file}")
            else:
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                    self.logger.debug(f"{config_dict = }")
                    self.app_config = self.config_class(**config_dict)
                    self.logger.debug(f"{self.app_config = }")
        else:
            raise FileNotFoundError(
                f"No config file provided. Please provide one in the env file ({self.env_file})."
                + f"You can use the {self.env_var} environment variable to specify the env file."
            )

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
            self._build_tools_description()

    def _build_tools_description(self) -> None:
        """Build per-domain tool descriptions from fetched MCP tools."""
        self.tools_description = {}
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
            desc = ""
            ctr = 0
            for t in self.mcp_tools:
                if "dummy" in t.name:
                    continue
                # tools will be prefixed with server names
                if num_servers > 1:
                    if not any(t.name.startswith(s + "_") for s in server_names):
                        continue
                # otherwise, there's only one server
                ctr += 1
                desc += dedent(f"""
                    ## {ctr}.  {t.name}

                    ### Description

                    {t.description}

                    """)
                if t.inputSchema:
                    desc += dedent(f"""
                        ### Parameters

                        {t.inputSchema.get("properties")}

                        """)
            self.tools_description[domain] = desc

    async def _get_vector_stores(self) -> None:
        """Get vector stores"""
        emb = self.llm_models.get("embedding")
        if self.stores_config and emb and emb.model_name:
            self.stores = VSRetriever(
                vs_config=self.stores_config,
                logger=self.logger,
                embedding_model=emb.model_name,
                default_k=self.default_k,
                k_max=self.k_max,
            )
            self.stores.setup()
            self.logger.info(f"Vector stores loaded: {self.stores.domains}")

            # dynamically generate schema for domains
            all_domains = self.stores.domains.copy()
            all_domains.append("undefined")

            self.QueryDomainSchema = create_model(
                "QueryDomainSchema",
                query_domains=(list[Literal[tuple(all_domains)]], "undefined"),
            )
        else:
            self.logger.warning("No vector stores configured.")

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

        Subclasses should implement this to populate ``self.stores_config``,
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

                elif event_type in ("info", "debug", "state"):
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
