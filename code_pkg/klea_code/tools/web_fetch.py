#!/usr/bin/env python3
"""
Web fetch tool for bundled Klea Code tools.

File: code_pkg/klea_code/tools/web_fetch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from pydantic import Field


async def web_fetch(
    url: Annotated[
        str,
        Field(
            description="HTTP or HTTPS URL to fetch",
            min_length=1,
        ),
    ],
    timeout: Annotated[
        float,
        Field(
            description="Request timeout in seconds",
            ge=1.0,
            le=120.0,
        ),
    ] = 30.0,
    max_chars: Annotated[
        int,
        Field(
            description="Maximum number of characters of content to return",
            ge=1,
            le=1_000_000,
        ),
    ] = 100_000,
) -> dict[str, Any]:
    """Fetch a URL and return its text content.
    Use this tool to read web pages, docs, or other HTTP resources.

    Example: web_fetch(url="https://example.com")
    """
    truncated = False
    error = ""
    content = ""
    status_code: int | None = None
    content_type = ""

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return {
            "url": url,
            "status_code": None,
            "content_type": "",
            "content": "",
            "truncated": False,
            "error": "URL must be an absolute http:// or https:// URL.",
        }

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                url,
                headers={
                    "Accept": "text/markdown, text/html, text/plain;q=0.9, */*;q=0.1",
                    "User-Agent": "klea-code-web-fetch/0.0.1",
                },
            )
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            content = response.text

            if len(content) > max_chars:
                content = content[:max_chars]
                truncated = True

            if response.is_error:
                error = f"HTTP {response.status_code}"
    except httpx.TimeoutException:
        error = f"Request timed out after {timeout} seconds."
    except httpx.RequestError as e:
        error = e.__str__()

    return {
        "url": url,
        "status_code": status_code,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
        "error": error,
    }
