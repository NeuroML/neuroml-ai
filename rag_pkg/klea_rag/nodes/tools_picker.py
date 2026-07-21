#!/usr/bin/env python3
"""
Tools picker node for RAG

File: rag_pkg/klea_rag/nodes/tools_picker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.nodes.abstract import NodeStreamData
from klea_utils.nodes.base import BaseLLMNode

from klea_rag.schemas import RAGState, ToolCallsSchema


class ToolsPicker(BaseLLMNode[RAGState]):
    """Node that selects tools to augment vector store retrieval."""

    model_type = "chat"

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
        temperature: float = 0.01,
        domain_tools_description: dict[str, str] | None = None,
    ):
        """Initialise the tools picker node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLModel}`` dict (from ``BaseLangGraph.llm_models``)
        :param temperature: Sampling temperature
        :param domain_tools_description: Per-domain tool descriptions
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            temperature=temperature,
            output_schema=ToolCallsSchema,
            memory=False,
        )
        self._domain_tools_description = domain_tools_description or {}

    def _get_tool_descriptions(self, domains: list[str]) -> str:
        """Get combined tool descriptions for the given domains."""
        parts = []
        for d in domains:
            if d in self._domain_tools_description:
                parts.append(self._domain_tools_description[d])
        return "\n\n".join(parts)

    @override
    def _pre_exec(self, state: RAGState) -> bool:
        """Pre-execution check.

        If no tool description is available for the current domain, skip.
        """
        if not self._get_tool_descriptions(state.query_domains):
            return False
        return True

    @override
    def _get_human_prompt(self, state: RAGState) -> str:
        """Return empty string -- this node only uses a system prompt."""
        return ""

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with query and retrieval context."""
        return {
            "query": state.query,
            "tools_description": self._get_tool_descriptions(state.query_domains),
        }

    @override
    def _update_state(self, result: ToolCallsSchema, state: RAGState) -> Dict[str, Any]:
        """Update state with selected tool calls."""
        return {"tool_calls": result.tool_calls}

    @override
    def _get_default_error_result(self) -> ToolCallsSchema:
        """Return default result when processing fails."""
        return ToolCallsSchema()

    @override
    def _get_info(self) -> NodeStreamData:
        """Return selected tools."""
        assert self._last_state_updates is not None
        tool_calls = self._last_state_updates.get("tool_calls", [])
        tool_names = [tc.name for tc in tool_calls]
        if tool_names:
            summary = f"Selected {len(tool_names)} tool(s): {', '.join(tool_names)}"
        else:
            summary = "No tools selected"
        return NodeStreamData(
            summary=summary,
            details={
                "tool_names": tool_names,
                "tool_count": len(tool_names),
            },
        )

    @override
    def _get_debug(self) -> NodeStreamData:
        """Return info + query, prompts, and full tool calls."""
        assert self._last_state is not None
        assert self._last_system_prompt is not None
        assert self._last_human_prompt is not None
        assert self._last_state_updates is not None
        info = self._get_info()
        details = info.details.copy()
        state: RAGState = self._last_state  # type: ignore[assignment]
        details.update(
            {
                "query": state.query,
                "system_prompt": self._last_system_prompt,
                "human_prompt": self._last_human_prompt,
            }
        )
        # Add full tool calls with arguments
        tool_calls = self._last_state_updates.get("tool_calls", [])
        if tool_calls:
            details["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments, "reason": tc.reason}
                for tc in tool_calls
            ]
        return NodeStreamData(summary=info.summary, details=details)
