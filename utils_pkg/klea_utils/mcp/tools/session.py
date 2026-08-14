#!/usr/bin/env python3
"""
Shared session protocol for MCP tool implementations.

File: klea_utils/mcp/tools/session.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Protocol


class SessionLike(Protocol):
    """Minimal HTTP session interface needed by the shared tool implementations.

    Kept structural so tests can substitute a fake and so the implementations
    do not depend on a specific HTTP client library.  Matches the subset of
    :class:`httpx.AsyncClient` that the bundled tools use (``get`` for
    downloads, ``stream`` for page fetches).
    """

    async def get(
        self,
        url: str,
        *,
        params: Any | None = None,
        headers: Any | None = None,
        timeout: Any = None,
        follow_redirects: bool = False,
    ) -> Any: ...

    def stream(
        self,
        method: str,
        url: str,
        *,
        params: Any | None = None,
        headers: Any | None = None,
        timeout: Any = None,
        follow_redirects: bool = False,
    ) -> Any: ...
