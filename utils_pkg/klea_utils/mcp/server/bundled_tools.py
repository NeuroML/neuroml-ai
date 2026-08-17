#!/usr/bin/env python3
"""
FastMCP tool wrappers for the shared Klea bundled tools.

File: klea_utils/mcp/server/bundled_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from klea_utils.mcp.registry import tool_meta
from klea_utils.mcp.schemas import ToolInfo
from klea_utils.mcp.tool_impls.download_file import download_file as download_file_impl
from klea_utils.mcp.tool_impls.list_files import list_files as list_files_impl
from klea_utils.mcp.tool_impls.read_file import read_file as read_file_impl
from klea_utils.mcp.tool_impls.web_fetch import web_fetch as web_fetch_impl

#: Common tags carried by every bundled tool, so "enable the common set"
#: is a single `include_tags: ["bundled"]` in the app config.
BUNDLED_TAG = "bundled"


@tool_meta(ToolInfo(tags={BUNDLED_TAG, "web"}, read_only=True))
async def web_fetch(
    ctx: Context,
    url: Annotated[str, Field(min_length=1)],
    timeout: Annotated[float, Field(ge=1.0, le=120.0)] = 30.0,
    max_chars: Annotated[int, Field(ge=1, le=1_000_000)] = 100_000,
) -> dict[str, Any]:
    """Fetch a URL and return its text content.

    Use this tool to read web pages, docs, or other HTTP resources.

    Use when:
    - Reading a page or document from the web.
    - Checking a URL that a user or another tool referenced.

    Do not use for:
    - Downloading a file to disk (use the download file tool instead).

    Example: web_fetch(url="https://example.com")

    Args:
        url: HTTP or HTTPS URL to fetch.
        timeout: Request timeout in seconds.
        max_chars: Maximum number of characters of content to return.

    Returns:
        Dictionary with url, status_code, content_type, content, truncated, error.
    """
    session = ctx.lifespan_context.get("http_session")
    return await web_fetch_impl(
        session=session,
        url=url,
        timeout=timeout,
        max_chars=max_chars,
    )


@tool_meta(
    ToolInfo(tags={BUNDLED_TAG, "local", "files"}, checkpaths=["path"], read_only=True)
)
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

    Use this tool to explore the local file system structure and find
    specific files.

    Use when:
    - Discovering what files exist in the working directory.
    - Finding files by name, type, or location.

    Do not use for:
    - Reading a file's contents (use the read file tool instead).

    Example: list_files(path=".", pattern="*.py", recursive=True)

    Args:
        path: Directory path to list. Must be relative to the current working
            directory and cannot contain '..' for security.
        max_depth: Maximum directory depth to traverse. 'None' for unlimited.
        pattern: Space separated file patterns to filter files by type.
        include_files: Whether to include files in results.
        include_directories: Whether to include directories in results.
        recursive: If True, traverse subdirectories recursively.
        max_results: Maximum number of entries to return.

    Returns:
        Dictionary with list of files, truncated flag, and error.
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


@tool_meta(
    ToolInfo(tags={BUNDLED_TAG, "local", "files"}, checkpaths=["path"], read_only=True)
)
async def read_file(
    path: Annotated[
        str,
        Field(
            description=(
                "File path to read. Must be relative to current working "
                "directory and cannot contain '..' for security"
            ),
            min_length=1,
        ),
    ],
    offset: Annotated[
        int,
        Field(description="1-indexed line to start reading from", ge=1),
    ] = 1,
    limit: Annotated[
        int | None,
        Field(description="Maximum number of lines to return. 'None' for end of file"),
    ] = 2000,
    max_chars: Annotated[
        int,
        Field(description="Hard cap on characters of content to return", ge=1),
    ] = 100_000,
) -> dict[str, Any]:
    """Read a file and return a slice of its text content.

    Use this tool to inspect source files, logs, or documents as plain text.
    Document formats (PDF, office files) are converted to Markdown first.

    Use when:
    - You need to see the contents of a file in the project.
    - You want to page through a large file by line numbers.

    Do not use for:
    - Listing a directory (use the list files tool instead).
    - Fetching remote content (use the web fetch tool instead).

    Example: read_file(path="README.md", offset=1, limit=100)

    Args:
        path: File path to read. Must be relative to the current working
            directory and cannot contain '..' for security.
        offset: 1-indexed line to start reading from.
        limit: Maximum number of lines to return. None reads to the end.
        max_chars: Hard cap on characters of content to return.

    Returns:
        Dictionary with content, line range, total_lines, truncated, error.
    """
    return read_file_impl(
        path=path,
        offset=offset,
        limit=limit,
        max_chars=max_chars,
    )


@tool_meta(
    ToolInfo(
        tags={BUNDLED_TAG, "web", "download"},
        checkpaths=["file_path"],
        destructive=True,
        open_world=True,
    )
)
async def download_file(
    ctx: Context,
    url: Annotated[str, Field(min_length=1)],
    file_path: Annotated[str, Field(min_length=1)],
) -> dict[str, Any]:
    """Download a URL to a local file.

    Use this tool to fetch binary or text resources from the web and save
    them to disk for later reading or processing.

    Use when:
    - Downloading a file such as a PDF, dataset, or archive.
    - Saving remote content locally before inspecting it.

    Do not use for:
    - Reading a web page as text (use the web fetch tool instead).

    Example: download_file(url="https://example.com/paper.pdf", file_path="paper.pdf")

    Args:
        url: HTTP or HTTPS URL to download.
        file_path: Destination path, relative to the working directory.
            Existing files are overwritten.

    Returns:
        Dictionary with the saved path, or an error on failure.
    """
    session = ctx.lifespan_context.get("http_session")
    target = await download_file_impl(
        session=session,
        url=url,
        file_path=file_path,
    )
    if target is None:
        return {
            "error": "Download failed (check the URL, network, or file path permissions)."
        }
    return {"saved_to": str(target)}
