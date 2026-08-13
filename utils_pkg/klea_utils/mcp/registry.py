#!/usr/bin/env python3
"""
Shared MCP tool registration helpers.

File: klea_utils/mcp/registry.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import inspect
import logging
from types import ModuleType
from typing import Any

from fastmcp import FastMCP

from klea_utils.mcp.schemas import ToolInfo

logger = logging.getLogger(__name__)


def register_tools(mcp: FastMCP, modules: list[ModuleType]):
    """Register tools from a given module

    :param mcp: FastMCP server to register the tools on.
    :param modules: list of modules with tool function definitions

    """
    for module in modules:
        for fname, fn in inspect.getmembers(module, inspect.isfunction):
            if fname.endswith("_tool"):
                if hasattr(fn, "_tool_meta"):
                    metadata: ToolInfo = fn._tool_meta

                    kwargs: dict[str, Any] = {}

                    # Only pass an explicit description when ToolInfo
                    # provides one.  Otherwise let fastmcp derive the
                    # LLM-facing description from the docstring's opening
                    # text block (klea's docstring-first convention).
                    # Passing the raw docstring here would dump the whole
                    # Args/Returns prose into the tool description,
                    # duplicating parameter text that the client also shows
                    # from the schema.
                    #
                    # Docstring conventions (summary + Use when / Do not
                    # use for bullets + one example, ~100-250 tokens) are
                    # documented in docs/concepts/mcp.rst, "Tool
                    # description length and style".
                    if metadata.description is not None:
                        kwargs["description"] = metadata.description
                    if metadata.title is not None:
                        kwargs["title"] = metadata.title
                    if metadata.tags is not None:
                        kwargs["tags"] = metadata.tags
                    if metadata.meta is not None:
                        kwargs["meta"] = metadata.meta

                    mcp.tool(fn, **kwargs)
                    logger.debug(f"Registered MCP tool: {fname}")
                else:
                    logger.error(f"Tool missing ToolInfo metadata: {fname}")
                    raise ValueError(f"{fname} is missing ToolInfo")


def tool_meta(metadata: ToolInfo):
    """Decorator that attaches :class:`ToolInfo` metadata to a tool function.

    Usage::

        @tool_meta(ToolInfo(tags={"bundled", "web"}))
        async def web_fetch_tool(ctx: Context, url: str, ...):
            ...

    The metadata is read by :func:`register_tools` when the function is
    registered on a FastMCP server (it sets ``description``, ``title``,
    ``tags``, and ``meta`` on the tool if provided).  Functions must be
    named with a ``_tool`` suffix for :func:`register_tools` to discover
    them.

    :param metadata: :class:`ToolInfo` to attach to the decorated function.
    :returns: The decorated function, unchanged, with ``_tool_meta`` set.
    """

    def wrapper(fn):
        fn._tool_meta = metadata
        return fn

    return wrapper
