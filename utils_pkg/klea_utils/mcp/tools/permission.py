#!/usr/bin/env python3
"""
Path permission checking for file-accessing MCP tools.

File: klea_utils/mcp/tools/permission.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
from pathlib import Path

from klea_utils.mcp.errors import PermissionDeniedError

logger = logging.getLogger(__name__)


# TODO (deferred): replace the hard boundary check with a full permission
# service: config rulesets (allow/deny/ask per path pattern) plus an
# LLM-free interactive approval loop (graph pause + TUI/web input),
# opencode-style.  Tracked as a separate task; for now access is denied to
# any path outside *project_root* and no user approval exists yet.
def check_path_access(
    path: str | os.PathLike, project_root: str | os.PathLike | None = None
) -> None:
    """Raise :class:`PermissionDeniedError` when *path* is not permitted.

    The permission layer is currently a stub: *path* is allowed only when it
    resolves inside *project_root* (default: the current working directory).
    Both sides are fully resolved first, so ``..`` traversal and symlink
    escapes outside the boundary are caught.  Anything else is denied with
    no way to grant access yet.

    :param path: File or directory path the tool wants to access.
    :param project_root: Boundary directory inside which access is allowed.
        Defaults to the current working directory.
    :raises PermissionDeniedError: when *path* resolves outside the boundary.
    """
    the_path = Path(path).expanduser().resolve()
    root = (
        Path(project_root).expanduser().resolve()
        if project_root
        else Path.cwd().resolve()
    )

    try:
        the_path.relative_to(root)
    except ValueError:
        logger.warning(f"Permission denied: {the_path} is outside {root}")
        raise PermissionDeniedError(
            f"Access to path outside the project directory is denied: {the_path}"
        ) from None

    logger.debug(f"Permission granted: {the_path} is inside {root}")
