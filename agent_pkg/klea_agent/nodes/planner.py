#!/usr/bin/env python3
"""
Planner node for KleaAgent

File: klea_agent/nodes/planner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, ClassVar, override

from klea_utils.mcp.schemas import ToolInfo
from klea_utils.nodes.base import BaseLLMNode
from pydantic import BaseModel

from klea_agent.schemas import KleaAgentState, PlanSchema


class Planner(BaseLLMNode[PlanSchema]):
    """Node that creates or updates an execution plan."""

    model_type = "plan"
    model_defaults: ClassVar[dict[str, Any]] = {"temperature": 0.01}

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
    ):
        """Initialise the planner node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            output_schema=PlanSchema,
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
    def _get_prompt_variables(self, state: KleaAgentState) -> dict:
        """Format prompt with current plan state."""
        return {
            "query": state.query,
            "goal": state.goal,
            "step_list": state.plan.step_list,
            "current_step_index": state.plan.current_step_index,
            "artefacts": state.artefacts,
            "discovery": state.discovery_persistent,
            "observations": state.step_outputs,
            "tools_description": self._tools_description,
        }

    @override
    def _update_state(self, result: PlanSchema, state: BaseModel) -> dict[str, Any]:
        """Update plan and generate summary for user."""
        plan_summary = "## Plan summary:\n\n"
        for step in result.step_list:
            plan_summary += f"- {step.step_number}: {step.description}"

        plan = state.plan  # type: ignore
        plan.step_list = result.step_list

        return {"plan": plan, "message_for_user": plan_summary}

    @override
    def _get_default_error_result(self) -> PlanSchema:
        """Return default result when processing fails."""
        return PlanSchema(status="failed")
