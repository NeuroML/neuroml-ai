#!/usr/bin/env python3
"""
Shared FastMCP lifespan helpers.

File: klea_utils/mcp/lifespan.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import aiohttp
from fastmcp.server.lifespan import lifespan

logger = logging.getLogger(__name__)


def make_http_session_lifespan(session_key: str = "aiohttp_session"):
    """Create a FastMCP lifespan that provides a shared aiohttp session.

    Tools that need an HTTP session (e.g. klea_utils.mcp.tools.web_fetch)
    read it from ``ctx.lifespan_context[<session_key>]`` in their MCP wrapper.
    Lifespans are composable with the ``|`` operator.

    :param session_key: Lifespan context key under which the session is stored.
    :returns: A FastMCP ``@lifespan``-decorated function.
    """

    @lifespan
    async def _http_session_lifespan(server):
        logger.debug("Creating shared aiohttp session")
        aiohttp_session = aiohttp.ClientSession()
        try:
            yield {session_key: aiohttp_session}
        finally:
            logger.debug("Closing shared aiohttp session")
            await aiohttp_session.close()

    return _http_session_lifespan
