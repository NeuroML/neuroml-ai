#!/usr/bin/env python3
"""
Web fetch implementation for Klea MCP tools.

File: klea_utils/mcp/tools/web_fetch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Protocol
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class _SessionLike(Protocol):
    """Minimal session interface needed by :func:`web_fetch`.

    Kept structural so tests can substitute a fake and so the implementation
    does not depend on a specific HTTP client library.
    """

    def get(self, url: str, **kwargs: Any) -> Any: ...


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text suitable for an LLM."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def web_fetch(
    session: _SessionLike | None,
    url: str,
    timeout: float = 30.0,
    max_chars: int = 100_000,
) -> dict[str, Any]:
    """Fetch a URL and return its text content.

    Framework-agnostic implementation shared across Klea MCP servers.  Apps
    wrap this in an MCP tool that supplies ``session`` from their lifespan
    context (see klea_utils.mcp.lifespan).

    :param session: HTTP session to use for the request.  ``None`` when no
        session is available.
    :param url: HTTP or HTTPS URL to fetch.
    :param timeout: Request timeout in seconds.
    :param max_chars: Maximum number of characters of content to return.

    :returns: dict with url, status_code, content_type, content, truncated,
        error.
    """
    logger.debug(f"Fetching URL\n{url = }\n{timeout = }\n{max_chars = }")

    truncated = False
    error = ""
    content = ""
    status_code: int | None = None
    content_type = ""

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        logger.warning(f"Rejecting invalid URL: {url}")
        return {
            "url": url,
            "status_code": None,
            "content_type": "",
            "content": "",
            "truncated": False,
            "error": "URL must be an absolute http:// or https:// URL.",
        }

    if session is None:
        logger.warning(f"No HTTP session available for: {url}")
        return {
            "url": url,
            "status_code": None,
            "content_type": "",
            "content": "",
            "truncated": False,
            "error": "HTTP session not initialized",
        }

    # TODO: consider a standard Firefox/Chrome User-Agent; many sites block
    # custom agents and otherwise return empty or challenge pages.
    headers = {
        "Accept": "text/markdown, text/html, text/plain;q=0.9, */*;q=0.1",
        "User-Agent": "klea-web-fetch/0.0.1",
    }

    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as response:
            status_code = response.status
            content_type = response.headers.get("Content-Type", "")
            raw = await response.text()

            if "html" in content_type.lower():
                content = _html_to_text(raw)
            else:
                content = raw

            if len(content) > max_chars:
                content = content[:max_chars]
                truncated = True

            if status_code >= 400:
                error = f"HTTP {status_code}"

            logger.debug(
                f"Fetched URL\n"
                f"{url = }\n"
                f"{status_code = }\n"
                f"{content_type = }\n"
                f"{len(content) = }\n"
                f"{truncated = }"
            )
    except TimeoutError:
        logger.warning(f"Request timed out after {timeout} seconds: {url}")
        error = f"Request timed out after {timeout} seconds."
    except aiohttp.ClientError as e:
        logger.warning(f"Request failed for {url}: {e}")
        error = e.__str__()

    return {
        "url": url,
        "status_code": status_code,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
        "error": error,
    }
