#!/usr/bin/env python3
"""
General related tools

File: mcp_pkg/neuroml_mcp/tools/web_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
from pathlib import Path

import httpx
from klea_utils.paths import get_cache_dir
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..utils import NML_MCP_DIRS

logger = logging.getLogger(__name__)


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    reraise=True,
)
async def _download_file_by_content(
    session, url: str, params: dict, timeout, file_path: Path
) -> Path | None:
    """Download a file content and save to provided file path with overwriting.

    Note that since this overwrites, this should not be exposed directly as a tool.
    Use a wrapper around this.
    """
    response = await session.get(
        url, params=params, timeout=timeout, follow_redirects=True
    )
    if response.is_success:
        file_contents = response.text
        with open(file_path, "w") as f:
            f.write(file_contents)
        logger.info(f"File saved to {file_path}")
        return file_path
    logger.warning(f"Failed to download {url}: HTTP {response.status_code}")
    return None


async def _download_file_to_cache_by_content(
    session, url: str, params: dict, timeout, disk_file_name: str
) -> Path | None:
    """Wrapper to download file to the cache, by content"""
    file_path = get_cache_dir(NML_MCP_DIRS) / Path(disk_file_name)
    return await _download_file_by_content(session, url, params, timeout, file_path)
