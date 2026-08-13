#!/usr/bin/env python3
"""
MCP utils

File: neuroml_mcp/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from klea_utils.paths import cleanup_dir, get_cache_dir, init_dir
from platformdirs import PlatformDirs

logger = logging.getLogger(__name__)

NML_MCP_DIRS = PlatformDirs("nml_mcp")


def init_cache_dir():
    """Initialise cache directory if it doesn't exist."""
    logger.debug("Initialising cache dir")
    init_dir(get_cache_dir(NML_MCP_DIRS))


def cleanup_cache_dir():
    """Clean up the cache contents.

    To be used at end of each session.
    """
    logger.debug("Cleaning up cache dir")
    cleanup_dir(get_cache_dir(NML_MCP_DIRS))
