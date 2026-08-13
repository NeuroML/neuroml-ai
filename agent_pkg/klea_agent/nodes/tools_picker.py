#!/usr/bin/env python3
"""
Tools picker node

File: klea_agent/nodes/tools_picker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, override

from klea_agent.schemas import KleaAgentState, ToolCallSchema
from klea_utils.mcp.schemas import ToolInfo
from klea_utils.nodes.base import BaseLLMNode


class ToolsPicker(BaseLLMNode[KleaAgentState]):
    """Node that selects the best tools for the current step."""

    model_type = "plan"
    model_defaults = {"temperature": 0.01}

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
    ):
        """Initialise the tools picker node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            output_schema=ToolCallSchema,
            memory=False,
        )
        self._tools_description = ""

    def set_tools_info(self, tools_info: dict[str, dict[str, ToolInfo]]) -> None:
        """Set tool metadata (called by orchestrator after construction)."""
        self._tools_description = (
            "\n\n".join(
                info.description or ""
                for domain_info in tools_info.values()
                for info in domain_info.values()
            )
            if tools_info
            else ""
        )

    @override
    def _get_human_prompt(self, state: KleaAgentState) -> str:
        """Return empty string  ---  this node only uses a system prompt."""
        return ""

    @override
    def _get_prompt_variables(self, state: KleaAgentState) -> dict:
        """Format prompt with current step state."""
        current_step_index = state.plan.current_step_index
        current_step = state.plan.step_list[current_step_index]

        return {
            "current_step": current_step,
            "artefacts": state.artefacts,
            "observations": state.tool_responses,
            "tools_description": self._tools_description,
        }

    @override
    def _update_state(
        self, result: ToolCallSchema, state: KleaAgentState
    ) -> dict[str, Any]:
        """Update state with the selected tool call."""
        return {"tool_call": result}

    @override
    def _get_default_error_result(self) -> ToolCallSchema:
        """Return default result when processing fails."""
        return ToolCallSchema(tool="INVALID")
