#!/usr/bin/env python3
"""
Tools picker node for RAG

File: rag_pkg/klea_rag/nodes/tools_picker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
from typing import Any, override

from klea_utils.llm import extract_llm_output_content, prompt_value_to_messages
from klea_utils.mcp.schemas import ToolInfo
from klea_utils.nodes.abstract import NodeStreamData
from klea_utils.nodes.base import BaseLLMNode

from klea_rag.schemas import RAGState, ToolCallsSchema


class ToolsPicker(BaseLLMNode[RAGState]):
    """Node that selects tools to augment vector store retrieval."""

    model_type = "chat"
    model_defaults = {"temperature": 0.01}

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
        domain_tools_info: dict[str, dict[str, ToolInfo]] | None = None,
    ):
        """Initialise the tools picker node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        :param domain_tools_info: Per-domain tool metadata
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            output_schema=ToolCallsSchema,
            memory=False,
        )
        self._domain_tools_info = domain_tools_info or {}

    def _get_tool_descriptions(self, domains: list[str]) -> str:
        """Get combined tool descriptions for the given domains."""
        parts = []
        for d in domains:
            if d in self._domain_tools_info:
                parts.extend(
                    info.description or ""
                    for info in self._domain_tools_info[d].values()
                )
        if parts:
            return "\n\n".join(parts)
        return ""

    @override
    def _pre_exec(self, state: RAGState) -> bool:
        """Pre-execution check.

        If no tool description is available for the current domain, skip.
        """
        return bool(self._get_tool_descriptions(state.query_domains))

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
    def _update_state(self, result: ToolCallsSchema, state: RAGState) -> dict[str, Any]:
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
        tool_names = [tc.tool for tc in tool_calls]
        if tool_names:
            summary = f"Selected {len(tool_names)} tool(s): {', '.join(tool_names)}"
        else:
            summary = "No tools selected"
        return NodeStreamData(
            heading="Tool Selection",
            summary=summary,
            details={
                "tool_names": tool_names,
                "tool_count": len(tool_names),
            },
        )

    @override
    def _get_debug(self) -> NodeStreamData:
        """Return info + input prompt, raw output, processed output, and full tool calls."""
        assert self._last_state is not None
        assert self._last_prompt is not None
        assert self._last_output is not None
        assert self._last_result is not None
        assert self._last_state_updates is not None
        info = self._get_info()
        details = info.details.copy()
        details.update(
            {
                "input_prompt": prompt_value_to_messages(self._last_prompt),
                "unprocessed_output": extract_llm_output_content(self._last_output),
                "processed_output": str(self._last_result),
            }
        )
        # Add full tool calls with arguments
        tool_calls = self._last_state_updates.get("tool_calls", [])
        if tool_calls:
            details["tool_calls"] = [
                {"name": tc.tool, "arguments": tc.args, "reason": tc.reason}
                for tc in tool_calls
            ]
        return NodeStreamData(
            heading=info.heading, summary=info.summary, details=details
        )

    @override
    def _get_status(self) -> NodeStreamData:
        """Return human-readable selected tool calls."""
        assert self._last_state_updates is not None
        tool_calls = self._last_state_updates.get("tool_calls", [])

        display_parts: list[str] = []
        for tc in tool_calls:
            tool_info = next(
                (
                    info
                    for domain_tools in self._domain_tools_info.values()
                    if (info := domain_tools.get(tc.tool)) is not None
                ),
                None,
            )
            title = tool_info.title if tool_info and tool_info.title else tc.tool
            display_parts.append(
                "**{title}**\n\n{arguments}".format(
                    title=title,
                    arguments="\n".join(
                        f"- `{key}`: `{value if isinstance(value, str) else json.dumps(value)}`"
                        for key, value in tc.args.items()
                    ),
                )
            )
        return NodeStreamData(
            heading="Tool Selection",
            summary=f"Tools selected: {len(tool_calls)}",
            display="\n\n".join(display_parts),
        )
