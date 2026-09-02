#!/usr/bin/env python3
"""
Client-side MCP tool-call dispatch with permission gating.

File: klea_utils/mcp/dispatch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
from typing import Any, cast

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
    n = len(tool_calls)
    results: list[CallToolResult | None] = [None] * n
    pending: list[tuple[int, Any]] = []

    async with mcp_client:
        for i, (tool, args) in enumerate(tool_calls):
            tool_meta = tools_meta.get(tool)
            denials = check_tool_arguments_permissions(tool_meta, args, project_root)
            if denials:
                logger.warning(
                    f"Denied tool call before dispatch\n{tool = }\n{denials = }"
                )
                results[i] = _denied_result(denials)
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
            gathered = await asyncio.gather(*coros, return_exceptions=True)
            for idx, res in zip(indices, gathered):
                if isinstance(res, BaseException):
                    tool, args = tool_calls[idx]
                    logger.warning(
                        f"Tool call failed\n{tool = }\n{idx = }\n{args = }\n{res = }"
                    )
                    results[idx] = CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"{res.__class__.__name__}: {res}",
                            )
                        ],
                        structured_content=None,
                        meta=None,
                        is_error=True,
                    )
                else:
                    results[idx] = res  # type: ignore[assignment]

    if any(r is None for r in results):
        missing = [i for i, r in enumerate(results) if r is None]
        offending = [(i, tool_calls[i]) for i in missing]
        logger.error(f"dispatch left unfilled slots\n{offending = }")
        raise RuntimeError(f"dispatch internal error: unfilled results at {missing}")

    return cast(list[CallToolResult], results)
