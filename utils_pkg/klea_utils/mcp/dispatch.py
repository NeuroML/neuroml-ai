#!/usr/bin/env python3
"""
Client-side MCP tool-call dispatch with permission gating.

File: klea_utils/mcp/dispatch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
from typing import Any

from fastmcp.client.client import CallToolResult
from mcp.types import TextContent

from klea_utils.mcp.tool_impls.permission import check_tool_arguments_permissions

logger = logging.getLogger(__name__)


def _denied_result(denials: list[str]) -> CallToolResult:
    """Build a non-halting error result for a permission-denied tool call."""
    return CallToolResult(
        content=[TextContent(type="text", text="\n".join(denials))],
        structured_content=None,
        meta=None,
        is_error=True,
    )


async def dispatch_tool_calls(
    mcp_client: Any,
    tool_calls: list[tuple[str, dict[str, Any]]],
    tools_meta: dict[str, dict[str, Any]] | None = None,
    project_root: str | None = None,
) -> list[CallToolResult]:
    """Gate and dispatch tool calls against an MCP server.

    For each ``(tool, args)`` pair, the tool's ``meta`` (from ``tools_meta``)
    is checked with :func:`check_tool_arguments_permissions` before the call
    reaches the server; denied calls never touch the server and instead
    produce a synthetic non-halting error result.  Allowed calls are
    dispatched in parallel, and the returned list stays aligned with the
    input *tool_calls* order.

    :param mcp_client: MCP client used for ``call_tool``.  Its reentrant
        context is entered/exited by this helper.  Typed as ``Any`` because
        :class:`fastmcp.Client.call_tool` has a complex signature (optional
        ``arguments``, keyword-only ``raise_on_error``, ``CallToolResult |
        ToolTask`` return) that a structural protocol would not cleanly
        match; tests substitute a fake implementing the subset used here.
    :param tool_calls: ``(tool name, arguments)`` pairs to invoke.
    :param tools_meta: Mapping of tool name to the tool's ``meta`` dict
        (e.g. ``{t.name: t.meta for t in mcp_tools}``).  Path arguments
        declared under ``checkpaths`` are permission-checked client-side.
    :param project_root: Boundary directory for the permission gate.
        Defaults to the current working directory.
    :returns: One :class:`CallToolResult` per input call, in input order.
    """
    tools_meta = tools_meta or {}
    results: list[CallToolResult] = []
    pending: list[tuple[int, Any]] = []

    async with mcp_client:
        for i, (tool, args) in enumerate(tool_calls):
            tool_meta = tools_meta.get(tool)
            denials = check_tool_arguments_permissions(tool_meta, args, project_root)
            if denials:
                logger.warning(
                    f"Denied tool call before dispatch\n{tool = }\n{denials = }"
                )
                results.append(_denied_result(denials))
            else:
                pending.append(
                    (
                        i,
                        mcp_client.call_tool(
                            name=tool,
                            arguments=args,
                            raise_on_error=False,
                        ),
                    )
                )

        if pending:
            indices, coros = zip(*pending)
            # Insert dispatched results at their original positions so the
            # returned list stays aligned with the input tool_calls.
            for idx, res in zip(indices, await asyncio.gather(*coros)):
                results.insert(idx, res)

    return results
