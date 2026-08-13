#!/usr/bin/env python3
"""
File download implementation for Klea MCP tools.

File: klea_utils/mcp/tools/download_file.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any

import httpx

from klea_utils.api.utils import _make_retryer_httpx
from klea_utils.mcp.tools.session import SessionLike

logger = logging.getLogger(__name__)


async def download_file(
    session: SessionLike | None,
    url: str,
    file_path: str | Path,
    params: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout = 30.0,
    retries: int = 3,
) -> Path | None:
    """Download a URL to *file_path* (overwriting) and return the path.

    Framework-agnostic implementation shared across Klea MCP servers.  Apps
    wrap this in an MCP tool that supplies ``session`` from their lifespan
    context (see klea_utils.mcp.lifespan).  Note that since this overwrites,
    this should not be exposed directly as a tool; use a wrapper around this.

    Transient failures (timeouts, connection errors, HTTP 5xx/429) are
    retried with exponential backoff.  Returns ``None`` when the download
    fails (non-2xx response, or no session available).

    :param session: HTTP session to use for the request.  ``None`` when no
        session is available.
    :param url: HTTP or HTTPS URL to download.
    :param file_path: Destination file path (existing files are overwritten).
    :param params: Optional query parameters for the request.
    :param timeout: Request timeout in seconds.
    :param retries: Number of attempts for transient failures.
    :returns: The written :class:`Path`, or ``None`` on failure.
    """
    logger.debug(
        f"Downloading\n"
        f"{url = }\n"
        f"{file_path = }\n"
        f"{params = }\n"
        f"{timeout = }\n"
        f"{retries = }"
    )

    if session is None:
        logger.warning(f"No HTTP session available for: {url}")
        return None

    async def _do_download() -> Path | None:
        response = await session.get(
            url,
            params=params,
            timeout=timeout,
            follow_redirects=True,
        )
        if not response.is_success:
            if response.status_code == 429 or response.status_code >= 500:
                # Transient server-side error; raise so the retryer retries.
                response.raise_for_status()
            logger.warning(f"Failed to download {url}: HTTP {response.status_code}")
            return None
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(response.text)
        logger.info(f"Saved downloaded file to {target}")
        return target

    retryer = _make_retryer_httpx(attempts=retries)
    try:
        return await retryer(_do_download)
    except (TimeoutError, httpx.HTTPError) as exc:
        logger.warning(f"Download failed for {url}: {exc}")
        return None


async def download_file_to_cache(
    session: SessionLike | None,
    url: str,
    cache_dir: str | Path,
    file_name: str,
    params: dict[str, Any] | None = None,
    timeout: float | httpx.Timeout = 30.0,
    retries: int = 3,
) -> Path | None:
    """Download a URL into *cache_dir* as *file_name* and return the path.

    Convenience wrapper around :func:`download_file` for callers that keep a
    per-app cache directory (see ``klea_utils.paths.get_cache_dir``).

    :param session: HTTP session to use for the request.  ``None`` when no
        session is available.
    :param url: HTTP or HTTPS URL to download.
    :param cache_dir: Directory in which to store the downloaded file.
    :param file_name: File name under *cache_dir* (existing files overwritten).
    :param params: Optional query parameters for the request.
    :param timeout: Request timeout in seconds.
    :param retries: Number of attempts for transient failures.
    :returns: The written :class:`Path`, or ``None`` on failure.
    """
    target = Path(cache_dir) / file_name
    return await download_file(
        session=session,
        url=url,
        file_path=target,
        params=params,
        timeout=timeout,
        retries=retries,
    )
