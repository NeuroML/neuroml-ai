#!/usr/bin/env python3
"""
Lifespan for MCP server

File: mcp_pkg/neuroml_mcp/server/app_lifespan.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import httpx
from fastmcp.server.lifespan import lifespan

from ..utils import cleanup_cache_dir, init_cache_dir

logger = logging.getLogger(__name__)

#: Connection-pool tuning for the shared HTTP session.  Generous so a busy
#: multi-user MCP server keeps many warm connections instead of paying
#: TCP/TLS handshakes under bursty load.
_SESSION_LIMITS = httpx.Limits(
    max_connections=100, max_keepalive_connections=100, keepalive_expiry=30.0
)


@lifespan
async def app_lifespan(server):
    """Life span for server"""
    logger.info("MCP Server starting up")

    # add more sessions here as required
    http_session = httpx.AsyncClient(
        limits=_SESSION_LIMITS, timeout=httpx.Timeout(30.0)
    )
    init_cache_dir()

    try:
        yield {"http_session": http_session}
    finally:
        logger.info("MCP Server shutting down")

        await http_session.aclose()
        cleanup_cache_dir()

        logger.info("MCP Server shut down")
