#!/usr/bin/env python3
"""
File listing implementation for Klea MCP tools.

File: klea_utils/mcp/tool_impls/list_files.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

from klea_utils.mcp.errors import PermissionDeniedError
from klea_utils.mcp.tool_impls.permission import check_path_access

logger = logging.getLogger(__name__)


def list_files(
    path: str,
    max_depth: int | None = None,
    pattern: str = "*",
    include_files: bool = True,
    include_directories: bool = True,
    recursive: bool = False,
    max_results: int = 100,
    project_root: str | None = None,
) -> dict[str, Any]:
    """List files and directories with filtering and metadata.

    Framework-agnostic implementation shared across Klea MCP servers.  Apps
    wrap this in an MCP tool (see klea_utils.mcp.registry).

    :param path: Directory path to list.  Must be relative to current working
        directory and cannot contain '..' for security.
    :param max_depth: Maximum directory depth to traverse.  1 lists the
        immediate entries inside *path*, 2 also descends one directory
        deeper, and so on.  ``None`` for unlimited.
    :param pattern: Space separated file patterns to filter based on file type.
    :param include_files: Whether to include files in results.
    :param include_directories: Whether to include directories in results.
    :param recursive: If True, traverse subdirectories recursively.
    :param max_results: Maximum number of entries to return.
    :param project_root: Boundary directory for the permission check.
        Defaults to the current working directory.

    :returns: dict with files, error, truncated.
    """
    logger.debug(
        f"Listing files\n"
        f"{path = }\n"
        f"{max_depth = }\n"
        f"{pattern = }\n"
        f"{include_files = }\n"
        f"{include_directories = }\n"
        f"{recursive = }\n"
        f"{max_results = }"
    )

    the_path = Path(path)
    truncated = False
    error = ""
    files: list[dict[str, Any]] = []
    paths: list[Path] = []

    if ".." in path:
        logger.warning(f"Rejecting path containing '..': {path}")
        return {
            "files": [],
            "truncated": False,
            "error": "Path contains '..', exiting.",
        }

    try:
        check_path_access(the_path, project_root)
    except PermissionDeniedError as exc:
        logger.warning(f"Permission denied for {path}")
        return {
            "files": [],
            "truncated": False,
            "error": str(exc),
        }

    patterns = list(set(pattern.split()))

    def _matches(entry: Path) -> bool:
        return any(fnmatch.fnmatch(entry.name, p) for p in patterns)

    def _include(entry: Path) -> bool:
        # Symlinks are always listed: they carry their own `link` type so the
        # caller can decide how to treat them; the include_* flags only apply
        # to real files and directories.
        if entry.is_symlink():
            return True
        is_dir = entry.is_dir()
        if is_dir:
            return include_directories
        return include_files

    try:
        if not recursive:
            with os.scandir(the_path) as it:
                for entry in it:
                    p = Path(entry.path)
                    if _matches(p) and _include(p):
                        paths.append(p)
        else:
            depth_limit = max_depth if max_depth is not None else float("inf")
            stack: list[tuple[Path, int]] = [(the_path, 1)]
            while stack:
                d, depth = stack.pop()
                if depth > depth_limit:
                    continue
                with os.scandir(d) as it:
                    entries = list(it)
                for entry in entries:
                    p = Path(entry.path)
                    if _matches(p) and _include(p):
                        paths.append(p)
                    if (
                        p.is_dir()
                        and not p.is_symlink()
                        and (max_depth is None or depth < max_depth)
                    ):
                        stack.append((p, depth + 1))

        if len(paths) > max_results:
            truncated = True

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
