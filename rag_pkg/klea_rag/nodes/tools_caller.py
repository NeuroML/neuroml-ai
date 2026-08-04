#!/usr/bin/env python3
"""
Tools caller node for RAG

File: rag_pkg/klea_rag/nodes/tools_caller.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
from typing import Any, override

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from klea_utils.nodes.abstract import (
    AbstractLangGraphNode,
    NodeStreamData,
    NodeStreamEvent,
)

from klea_rag.schemas import RAGState


class ToolsCaller(AbstractLangGraphNode[RAGState, dict[str, Any]]):
    """Node that calls MCP tools based on tool_calls in state."""

    def __init__(self, logger: logging.Logger, label: str, mcp_client: Client | None):
        """Initialise the tools caller node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param mcp_client: MCP client instance (None skips tool calls)
        """
        super().__init__(logger=logger, label=label)
        self._mcp_client = mcp_client

    @override
    async def execute(self, state: RAGState) -> dict[str, Any]:
        self.logger.debug(f"{state =}")

        # no _pre_exec here
        if not state.tool_calls or not self._mcp_client:
            self.logger.debug("Pre-exec check failed, skipping execution")
            return {}

        self.write_custom_stream({"type": "progress", "node": self.label})

        results: list[CallToolResult] = []

        async with self._mcp_client:
            tasks = [
                self._mcp_client.call_tool(
                    name=tc.tool,
                    arguments=tc.args,
                    raise_on_error=False,
                )
                for tc in state.tool_calls
            ]
            results = await asyncio.gather(*tasks)

        self.logger.debug(f"{results =}")

        # Emit info event with tool call summary
        tool_names = [tc.tool for tc in state.tool_calls]
        success_count = sum(1 for r in results if not r.is_error)
        info_data = NodeStreamData(
            heading="Tool Execution",
            summary=f"Called {len(tool_names)} tool(s), {success_count} succeeded",
            details={
                "tool_names": tool_names,
                "total_calls": len(tool_names),
                "successful_calls": success_count,
                "failed_calls": len(tool_names) - success_count,
            },
        )
        info_event = NodeStreamEvent(type="info", node=self.label, data=info_data)
        self.write_custom_stream(info_event.model_dump())

        # Emit debug event with full tool call details and results
        debug_details = info_data.details.copy()
        debug_details["tool_calls"] = [
            {"tool": tc.tool, "arguments": tc.args, "reason": tc.reason}
            for tc in state.tool_calls
        ]
        debug_details["tool_results"] = [
            {
                "tool": tool_names[i] if i < len(tool_names) else f"tool_{i}",
                "is_error": r.is_error,
                "content": str(r.content) if r.content else None,
                "structured_content": r.structured_content,
            }
            for i, r in enumerate(results)
        ]
        debug_data = NodeStreamData(
            heading=info_data.heading, summary=info_data.summary, details=debug_details
        )
        debug_event = NodeStreamEvent(type="debug", node=self.label, data=debug_data)
        self.write_custom_stream(debug_event.model_dump())

        # Replace because we want fresh results at RAG loop
        return {"tool_results": results}
