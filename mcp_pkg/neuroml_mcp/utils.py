#!/usr/bin/env python3
"""
MCP utils

File: neuroml_mcp/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import inspect
from typing import Any

from klea_utils.mcp.schemas import ToolInfo
from klea_utils.paths import cleanup_dir, get_cache_dir, init_dir
from platformdirs import PlatformDirs

NML_MCP_DIRS = PlatformDirs("nml_mcp")


def init_cache_dir():
    """Initialise cache directory if it doesn't exist."""
    init_dir(get_cache_dir(NML_MCP_DIRS))


def cleanup_cache_dir():
    """Clean up the cache contents.

    To be used at end of each session.
    """
    cleanup_dir(get_cache_dir(NML_MCP_DIRS))


def register_tools(mcp, modules: list):
    """Register tools from a given module

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
                else:
                    raise ValueError(f"{fname} is missing ToolInfo")


def tool_meta(metadata: ToolInfo):
    """Attach metadata to tools."""

    def wrapper(fn):
        fn._tool_meta = metadata
        return fn

    return wrapper
