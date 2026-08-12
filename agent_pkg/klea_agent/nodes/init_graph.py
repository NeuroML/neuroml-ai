#!/usr/bin/env python3
"""
Initialise graph state node

File: code_pkg/klea_code/nodes/init_graph.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, override

from klea_code.schemas import GoalSchema, KleaCodeState, PlanSchema
from klea_utils.nodes.abstract import AbstractLangGraphNode


class InitGraphState(AbstractLangGraphNode[KleaCodeState, dict[str, Any]]):
    """Initialise/reset graph state before each iteration."""

    def __init__(self, logger: logging.Logger, label: str):
        """Initialise with a logger."""
        super().__init__(logger, label)

    @override
    async def execute(self, state: KleaCodeState) -> dict[str, Any]:
        """Reset state fields to their initial values."""
        self.write_custom_stream({"type": "progress", "node": self.label})
        return {
            "message_for_user": "",
            "plan": PlanSchema(),
            "goal": GoalSchema(),
            "tool_call": None,
            "tool_responses": [],
        }
