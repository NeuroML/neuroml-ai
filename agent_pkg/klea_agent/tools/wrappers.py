#!/usr/bin/env python3
"""
FastMCP wrappers for the shared Klea bundled tools.

File: klea_agent/tools/wrappers.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Annotated, Any

from fastmcp import Context
from klea_utils.mcp.registry import tool_meta
from klea_utils.mcp.schemas import ToolInfo
from klea_utils.mcp.tools.list_files import list_files as list_files_impl
from klea_utils.mcp.tools.web_fetch import web_fetch as web_fetch_impl
from pydantic import Field


@tool_meta(ToolInfo(tags={"bundled", "web"}))
async def web_fetch(
    ctx: Context,
    url: Annotated[str, Field(min_length=1)],
    timeout: Annotated[float, Field(ge=1.0, le=120.0)] = 30.0,
    max_chars: Annotated[int, Field(ge=1, le=1_000_000)] = 100_000,
) -> dict[str, Any]:
    """Fetch a URL and return its text content.
    Use this tool to read web pages, docs, or other HTTP resources.

    Args:
        url: HTTP or HTTPS URL to fetch.
        timeout: Request timeout in seconds.
        max_chars: Maximum number of characters of content to return.

    Returns:
        Dictionary with url, status_code, content_type, content, truncated, error.

    Example:
        web_fetch(url="https://example.com")
    """
    session = ctx.lifespan_context.get("aiohttp_session")
    return await web_fetch_impl(
        session=session,
        url=url,
        timeout=timeout,
        max_chars=max_chars,
    )


@tool_meta(ToolInfo(tags={"bundled", "files"}))
async def list_files(
    path: Annotated[
        str,
        Field(
            description=(
                "Directory path to list. Must be relative to current working "
                "directory and cannot contain '..' for security"
            ),
            min_length=1,
        ),
    ],
    max_depth: Annotated[
        int | None,
        Field(description="Maximum directory depth to traverse. 'None' for unlimited"),
    ] = None,
    pattern: Annotated[
        str,
        Field(
            description=(
                """
                Space separated file patterns to filter based on files type.
                Correct: '*.py'
                Correct: '*.md'
                Correct: '*.py *.md'
            """
            )
        ),
    ] = "*",
    include_files: Annotated[
        bool, Field(description="Whether to include files in results")
    ] = True,
    include_directories: Annotated[
        bool, Field(description="Whether to include directories in results")
    ] = True,
    recursive: Annotated[
        bool, Field(description="If True, traverse subdirectories recursively")
    ] = False,
    max_results: Annotated[
        int, Field(description="Maximum number of entries to return", ge=1, le=10000)
    ] = 100,
) -> dict[str, Any]:
    """List files and directories with filtering and metadata.
    Use this tool to explore file system structure and find specific files.

    Example: list_files(path=".", pattern="*.py", recursive=True)
    """
    return list_files_impl(
        path=path,
        max_depth=max_depth,
        pattern=pattern,
        include_files=include_files,
        include_directories=include_directories,
        recursive=recursive,
        max_results=max_results,
    )
