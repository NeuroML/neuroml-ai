#!/usr/bin/env python3
"""
Goal setter node

File: code_pkg/klea_code/nodes/goal_setter.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_code.schemas import GoalSchema, KleaCodeState
from klea_utils.nodes.base import BaseLLMNode
from pydantic import BaseModel


class GoalSetter(BaseLLMNode[GoalSchema]):
    """Goal setter node"""

    model_type = "plan"
    model_defaults = {"temperature": 0.01}

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
        output_schema: type[GoalSchema],
        memory: bool = False,
    ):
        """Initialise the goal setter node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        :param output_schema: Pydantic schema for structured output
        :param memory: Whether to append memory content to the system prompt
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            output_schema=output_schema,
            memory=memory,
        )

    @override
    def _get_prompt_variables(self, state: KleaCodeState) -> dict:
        """Format prompt with state-specific parameters"""
        variables = {"query": state.query}
        self.logger.debug(f"{variables =}")
        return variables

    @override
    def _update_state(self, result: GoalSchema, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        state_update = {"goal": result, "message_for_user": result.goal}
        self.logger.debug(state_update)
        return state_update

    @override
    def _get_default_error_result(self) -> GoalSchema:
        """Return default result when processing fails"""
        return self.output_schema(goal="Invalid", success_criteria="Invalid")
