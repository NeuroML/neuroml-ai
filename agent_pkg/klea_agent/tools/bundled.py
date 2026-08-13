#!/usr/bin/env python3
"""
Bundled tools server for Klea Agent.

Provides core tools that run via stdio MCP transport, eliminating the need
for an external MCP server for common operations.  Tool implementations
live in klea_utils (klea_utils.mcp.tools); this module wires the Klea Agent
wrappers (klea_agent.tools.wrappers) onto a FastMCP server.

File: klea_agent/tools/bundled.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastmcp import FastMCP
from klea_utils.mcp.lifespan import make_http_session_lifespan
from klea_utils.mcp.registry import register_tools

from klea_agent.tools import wrappers

bundle_server = FastMCP(
    "KleaAgent",
    instructions="Built-in tools for file operations and web fetching.",
    lifespan=make_http_session_lifespan(),
)

register_tools(bundle_server, [wrappers])


if __name__ == "__main__":
    bundle_server.run(transport="stdio")
