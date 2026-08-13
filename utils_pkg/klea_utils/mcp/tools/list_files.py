#!/usr/bin/env python3
"""
File listing implementation for Klea MCP tools.

File: klea_utils/mcp/tools/list_files.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def list_files(
    path: str,
    max_depth: int | None = None,
    pattern: str = "*",
    include_files: bool = True,
    include_directories: bool = True,
    recursive: bool = False,
    max_results: int = 100,
) -> dict[str, Any]:
    """List files and directories with filtering and metadata.

    Framework-agnostic implementation shared across Klea MCP servers.  Apps
    wrap this in an MCP tool (see klea_utils.mcp.registry).

    :param path: Directory path to list.  Must be relative to current working
        directory and cannot contain '..' for security.
    :param max_depth: Maximum directory depth to traverse.  'None' for
        unlimited.
    :param pattern: Space separated file patterns to filter based on file type.
    :param include_files: Whether to include files in results.
    :param include_directories: Whether to include directories in results.
    :param recursive: If True, traverse subdirectories recursively.
    :param max_results: Maximum number of entries to return.

    :returns: dict with files, error, truncated.
    """
    logger.debug(
        f"Listing files\n{path = }\n{pattern = }\n{recursive = }\n{max_results = }"
    )

    the_path = Path(path)
    truncated = "False"
    error = ""
    files: list[dict[str, Any]] = []
    paths: list[Path] = []

    if ".." in path:
        logger.warning(f"Rejecting path containing '..': {path}")
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
    except OSError as e:
        logger.warning(f"Error listing {path}: {e}")
        error = e.__str__()

    logger.debug(
        f"Listed files\n{path = }\n{len(files) = }\n{truncated = }\n{error = }"
    )

    result = {"files": files, "error": error, "truncated": truncated}

    return result
