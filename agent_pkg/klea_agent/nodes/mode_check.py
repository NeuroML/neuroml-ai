#!/usr/bin/env python3
"""
Mode check node — offers downgrade when scientific has no curated source.

File: klea_agent/nodes/mode_check.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from collections.abc import Callable
from typing import Any, override

from klea_utils.interrupt.dialog import Action, InterruptDialog
from klea_utils.nodes.abstract import (
    AbstractLangGraphNode,
    NodeStreamData,
    NodeStreamEvent,
)
from langgraph.errors import NodeInterrupt

from klea_agent.schemas import KleaAgentState

logger = logging.getLogger(__name__)


class ModeCheck(AbstractLangGraphNode[KleaAgentState, dict[str, Any]]):
    """Check if scientific mode lacks a curated knowledge source.

    When ``mode == scientific`` but no retriever/stores are configured,
    the graph cannot satisfy ADR-0029 grounding.  This node sets
    ``needs_downgrade`` and a user-facing message so the frontend can
    offer “add a source or switch to general (unverified)”.  In general
    mode it is a no-op.

    The ``has_stores`` callable is supplied by the orchestrator and
    should return ``True`` when at least one curated source is available.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        has_stores: Callable[[], bool] | None = None,
    ):
        super().__init__(logger, label)
        self._has_stores = has_stores or (lambda: False)

    @override
    async def execute(self, state: KleaAgentState) -> dict[str, Any]:
        self.write_custom_stream({"type": "progress", "node": self.label})
        self.logger.debug(f"{state.mode = }")
        mode = getattr(state, "mode", "general") or "general"
        # Emit mode for inspection even when interrupting
        info = NodeStreamData(
            heading="Mode Check",
            summary=f"Mode {mode} — {'needs downgrade' if mode == 'scientific' else 'ok'}",
            details={"mode": mode},
        )
        self.write_custom_stream(
            NodeStreamEvent(type="info", node=self.label, data=info).model_dump()
        )
        if mode == "scientific":
            has = False
            try:
                has = bool(self._has_stores())
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"has_stores check failed: {exc}")
                has = False
            self.logger.debug(f"{has = } for scientific check")
            if not has:
                dialog = InterruptDialog(
                    kind="needs_downgrade",
                    title="Scientific mode needs a source",
                    message=(
                        "No curated knowledge source found for scientific mode. "
                        "Either add an approved source, or switch to general mode to continue as unverified."
                    ),
                    actions=[
                        Action(
                            label="Switch to general",
                            value={"mode": "general"},
                            style="positive",
                        ),
                        Action(label="Ok", value=None),
                    ],
                )
                # Also emit state for status drawer before interrupting
                status = NodeStreamData(
                    heading="Mode Check",
                    summary="Scientific mode needs a source",
                    details={"mode": mode, "needs_downgrade": True},
                    display=dialog.message,
                )
                self.write_custom_stream(
                    NodeStreamEvent(
                        type="state", node=self.label, data=status
                    ).model_dump()
                )
                raise NodeInterrupt(dialog.to_interrupt_value())
        return {}
