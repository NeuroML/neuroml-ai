#!/usr/bin/env python3
"""
General code execution tools.

Note that docstrings here should be written for the LLM to read.

File: neuroml_mcp/tools/code_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from klea_utils.mcp.registry import tool_meta
from klea_utils.mcp.schemas import ToolInfo
from pydantic import Field

from neuroml_mcp.tools.sandbox.sandbox import RunPythonCode

from .sandbox import nml_mcp_sandbox

# set the implementation for development
sbox = nml_mcp_sandbox


@tool_meta(ToolInfo(title="Echo text", tags={"testing"}))
async def dummy_code_tool(
    astring: str,
) -> str:
    """Return the input string in a sentence (testing tool only).

    Use this tool to test and debug the MCP tool infrastructure.

    Use when:
    - Unit testing the tool server or the tool picker.

    Do not use for:
    - Any real task - this tool provides no real functionality.

    Example: dummy_code_tool("hello")

    Args:
        astring: String to be echoed back.
    """
    return f"I got {astring}"


@tool_meta(ToolInfo(title="List files and directories", tags={"testing"}))
async def list_files_tool(
    path: Annotated[str, Field(min_length=1)],
    max_depth: int | None = None,
    # LLMs are trained on shell style globs, so they insist on using space
    # separated file patterns. So we explicitly support these. Otherwise, this
    # becomes error prone.
    pattern: str = "*",
    include_files: bool = True,
    include_directories: bool = True,
    recursive: bool = False,
    max_results: Annotated[int, Field(ge=1, le=10000)] = 100,
) -> dict[str, Any]:
    """List files and directories with filtering and metadata.

    Use this tool to explore the file system structure and find specific
    files.

    Use when:
    - You need to see what files and directories exist under a path.
    - You want to locate files matching a pattern (e.g. '*.py').

    Do not use for:
    - Reading the contents of a file (use the file reading tool instead).
    - Running commands or scripts (use the code execution tool instead).

    Example: list_files_tool(path=".", pattern="*.py", recursive=True)

    Args:
        path: Directory path to list. Must be relative to the current working
            directory and cannot contain '..' for security.
        max_depth: Maximum directory depth to traverse. 'None' for unlimited.
        pattern: Space separated file patterns to filter based on file type.
            Correct: '*.py', '*.md', '*.py *.md'.
        include_files: Whether to include files in results.
        include_directories: Whether to include directories in results.
        recursive: If True, traverse subdirectories recursively.
        max_results: Maximum number of entries to return.

    Returns:
        Dict with the matching files, an error message (if any), and a
        truncated flag.
    """
    the_path = Path(path)
    truncated = "False"
    error = ""
    files: list[dict[str, Any]] = []
    paths: list[Path] = []

    if ".." in path:
        return {
            "files": [],
            "truncated": "False",
            "error": "Path contains '..', exiting.",
        }

    patterns = pattern.split()
    patterns = list(set(patterns))

    try:
        for p in patterns:
            if recursive:
                paths.extend(list(the_path.rglob(p)))
            else:
                paths.extend(list(the_path.glob(p)))

        if len(paths) > max_results:
            truncated = "True"

        for f in paths[:max_results]:
            ftype = "file"
            if f.is_dir():
                ftype = "directory"
            if f.is_symlink():
                ftype = "link"
            files.append(
                {
                    "path": str(f),
                    "type": ftype,
                    "modified time": f.stat().st_mtime,
                    "size": f.stat().st_size,
                }
            )
    except Exception as e:
        error = e.__str__()

    result = {"files": files, "error": error, "truncated": truncated}

    return result


@tool_meta(ToolInfo(title="Execute Python code", tags={"testing"}))
async def run_python_code_tool(
    code: Annotated[str, Field(min_length=1)],
) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment.

    Use this tool to test code snippets, generate models, and perform
    calculations.

    Use when:
    - You need to run a short Python snippet to test or compute something.
    - You need to generate or manipulate NeuroML structures with code.

    Do not use for:
    - Simple file operations (use the file tools instead).
    - Long-running or interactive programs (the sandbox rejects these).

    Example: run_python_code_tool("import numpy; print('numpy version:', numpy.__version__)")

    Args:
        code: Complete Python code to execute. Must be valid Python syntax
            and cannot require interactive input.

    Returns:
        Dict with the execution result.
    """
    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)
