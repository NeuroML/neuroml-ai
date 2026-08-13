#!/usr/bin/env python3
"""
Initialise graph state node

File: klea_agent/nodes/init_graph.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, override

from klea_agent.schemas import GoalSchema, KleaAgentState, PlanSchema
from klea_utils.nodes.abstract import AbstractLangGraphNode


class InitGraphState(AbstractLangGraphNode[KleaAgentState, dict[str, Any]]):
    """Initialise/reset graph state before each iteration."""

    def __init__(self, logger: logging.Logger, label: str):
        """Initialise with a logger."""
        super().__init__(logger, label)

    @override
    async def execute(self, state: KleaAgentState) -> dict[str, Any]:
        """Reset state fields to their initial values."""
        self.write_custom_stream({"type": "progress", "node": self.label})
        return {
            "message_for_user": "",
            "plan": PlanSchema(),
            "goal": GoalSchema(),
            "tool_call": None,
            "tool_responses": [],
        }
