#!/usr/bin/env python3
"""
Abstract node classes for LangGraph processing nodes

File: klea_utils/nodes/abstract.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Type, final

from langchain.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


class NodeStreamData(BaseModel):
    """Data payload for node streaming events.

    This is the contract between nodes and the frontend.
    """

    summary: str = Field(
        description="Human-readable summary, always rendered by frontend"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data, rendered as collapsible JSON",
    )


class NodeStreamEvent(BaseModel):
    """Full streaming event emitted by nodes.

    This is the contract between the graph infrastructure and the frontend.
    """

    type: Literal["info", "debug"] = Field(description="Event type")
    node: str = Field(description="Node label")
    data: NodeStreamData = Field(description="Event payload")


class AbstractLangGraphNode[TSchema: BaseModel, TReturn](ABC):
    """Abstract base class for all LangGraph nodes.

    Generic over TReturn to support both state-updating nodes (Dict[str, Any])
    and other nodes, e.g., router nodes (str) and tool caller nodes.

    Provides a consistent interface: all nodes have a logger and an
    execute(state) method.
    """

    def __init__(self, logger: logging.Logger, label: str):
        """Initialise

        Creates a new hierarchical logger.

        :param logger: Parent logger instance (used to derive child logger name)
        :param label: Human-readable label for this node, used as the
            LangGraph node name for UI progress display
        """
        # Child logger -- inherits the parent's dual-stream handlers
        # (set up by BaseLangGraph via plogging.setup_logger) through
        # propagation, so this class does NOT configure its own
        # handlers.
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")
        self.label = label

    def write_custom_stream(self, event: dict) -> None:
        """Emit a custom event to the LangGraph v3 stream.

        Writes to the ``custom`` channel via ``get_stream_writer()``.
        Requires a ``StreamTransformer`` with ``required_stream_modes =
        ("custom",)`` registered so the channel is enabled (done by
        ``BaseLangGraph.run_graph_astream_events()``).

        Call this at the top of ``execute()`` to emit progress, or
        anywhere to emit debug or intermediate data for UI consumers.

        :param event: Dict to emit as a custom protocol event
        """
        from langgraph.config import get_stream_writer

        get_stream_writer()(event)

    @abstractmethod
    async def execute(self, state: TSchema) -> TReturn:
        """Execute this node and return the result.

        :param state: Current graph state
        :returns: State updates (dict) or routing label (str)
        """
        ...


class AbstractLLMNode[TSchema: BaseModel](
    AbstractLangGraphNode[TSchema, Dict[str, Any]]
):
    """Abstract base class for LangGraph nodes that use LLMs.

    Subclasses **must** set :attr:`model_type` to a key present in the
    ``llm_models`` dict (e.g. ``"chat"``, ``"plan"``, ``"guard"``).

    Implements a template execution flow:
    1. Pre-execution check (optional skip)
    2. Build prompt (system + human)
    3. Invoke LLM
    4. Process output (structured or raw)
    5. Update state
    """

    model_type: str = ""

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
        temperature: float,
        output_schema: Type[TSchema] | None = None,
    ):
        """Initialize with logger and model.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        :param temperature: Sampling temperature
        :param output_schema: Pydantic schema for structured output
        """
        super().__init__(logger, label)
        self.llm_models = llm_models
        try:
            self._llm_entry = self.llm_models[self.model_type]
        except KeyError:
            raise KeyError(
                f"Node '{type(self).__name__}' has model_type='{self.model_type}', "
                f"but llm_models only has keys: {list(self.llm_models)}"
            ) from None
        self.temperature = temperature
        self._output_schema = output_schema

    @final
    async def execute(self, state: BaseModel) -> Dict[str, Any]:
        """Template method defining standard execution flow"""
        # Clear previous execution context to prevent stale data.
        # These are instance variables (not locals) so that streaming hooks
        # (_pre_exec_stream, _post_exec_stream, _get_info, _get_debug) can
        # access intermediate values for progress reporting.
        self._last_state = None
        self._last_human_prompt = None
        self._last_system_prompt = None
        self._last_template = None
        self._last_variables = None
        self._last_prompt = None
        self._last_llm = None
        self._last_output = None
        self._last_result = None
        self._last_state_updates = None

        self.logger.debug(f"{state =}")

        if not self._pre_exec(state):
            self.logger.debug("Pre-exec check failed, skipping execution")
            return {}

        self._last_state = state
        self._pre_exec_stream()

        self._last_human_prompt = self._get_human_prompt(state)
        self._last_system_prompt = self._get_system_prompt(state)
        self._last_template = self._create_prompt_template(
            self._last_system_prompt, self._last_human_prompt
        )
        self._last_variables = self._get_prompt_variables(state)
        self._last_prompt = self._invoke_prompt(
            self._last_template, self._last_variables
        )
        self._last_llm = self._configure_llm()
        self._last_output = self._invoke_llm(self._last_llm, self._last_prompt)
        self._last_result = self._process_output(self._last_output)
        self._last_state_updates = self._update_state(self._last_result, state)

        self._post_exec_stream()

        self.logger.debug(f"{self._last_state_updates =}")
        return self._last_state_updates

    @abstractmethod
    def _pre_exec(self, state: BaseModel) -> bool:
        """Pre-execution check. Override to conditionally skip node execution.

        Return False to skip execution (returns empty dict).
        Return True (default) to proceed with the standard flow.
        """
        ...

    def _pre_exec_stream(self) -> None:
        """Emit streaming event before LLM invocation.

        Default: emits a ``progress`` event with the node label.
        Override to customise pre-execution streaming.
        """
        self.write_custom_stream({"type": "progress", "node": self.label})

    def _post_exec_stream(self) -> None:
        """Emit streaming events after LLM invocation.

        Default: emits ``info`` and ``debug`` events from ``_get_info`` and
        ``_get_debug`` if they return non-None values.
        Override to customise post-execution streaming.
        """
        info = self._get_info()
        if info:
            event = NodeStreamEvent(type="info", node=self.label, data=info)
            self.write_custom_stream(event.model_dump())
        debug = self._get_debug()
        if debug:
            event = NodeStreamEvent(type="debug", node=self.label, data=debug)
            self.write_custom_stream(event.model_dump())

    def _get_info(self) -> NodeStreamData | None:
        """Return structured summary data for an ``info`` stream event.

        Override in subclasses to provide node-specific summary data.
        Has access to all ``self._last_*`` values.

        :returns: NodeStreamData with summary and details, or None to skip

        Example::

            return NodeStreamData(
                summary="Classified into: neuron, morphology",
                details={"classified_domains": ["neuron", "morphology"]}
            )
        """
        return None

    def _get_debug(self) -> NodeStreamData | None:
        """Return structured debug data for a ``debug`` stream event.

        Override in subclasses to provide node-specific debug data.
        Has access to all ``self._last_*`` values.

        :returns: NodeStreamData with summary and details, or None to skip

        Example::

            info = self._get_info()
            details = info.details.copy()
            details["system_prompt"] = self._last_system_prompt
            return NodeStreamData(summary=info.summary, details=details)
        """
        return None

    @abstractmethod
    def _configure_llm(self) -> Runnable:
        """Configure LLM with structured output"""
        ...

    @abstractmethod
    def _invoke_llm(
        self, llm: Runnable, prompt: PromptValue
    ) -> AIMessage | dict[str, Any]:
        """Invoke LLM with default temperature - can be overridden"""
        ...

    @abstractmethod
    def _process_output(self, output: AIMessage | dict[str, Any]) -> Any:
        """Common output processing with error handling"""
        ...

    @abstractmethod
    def _invoke_prompt(
        self, prompt_template: ChatPromptTemplate, variables: Any | Dict[str, Any]
    ) -> PromptValue:
        """Format prompt with state-specific parameters"""
        ...

    @abstractmethod
    def _get_human_prompt(self, state: BaseModel) -> str:
        """Return human prompt for this node"""
        ...

    @abstractmethod
    def _get_system_prompt(self, state: BaseModel) -> str:
        """Return system prompt for this node"""
        ...

    @abstractmethod
    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate for this node"""
        ...

    @abstractmethod
    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with state-specific parameters"""
        ...

    @abstractmethod
    def _update_state(self, result: Any, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        ...

    @abstractmethod
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails"""
        ...


class AbstractRouterNode[TSchema: BaseModel](AbstractLangGraphNode[TSchema, str]):
    """Abstract class for LangGraph router nodes.

    Router nodes inspect the state and return a string label that determines
    which edge to follow next. Used with ``add_conditional_edges()``.
    """

    @abstractmethod
    async def execute(self, state: TSchema) -> str:
        """Return the routing label (edge name) based on state."""
        ...
