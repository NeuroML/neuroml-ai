#!/usr/bin/env python3
"""
MCP server for answers from docs

File: neuroml_ai/mcp/server/answers.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import inspect
from mcp.server.fastmcp import FastMCP


class MCPServer(object):

    """MCP class for NML RAG"""

    def __init__(self, name: str):
        self._mcp = FastMCP(name, json_response=True)

    @property
    def mcp(self) -> FastMCP:
        """Get the MCP server instance"""
        return self._mcp

    def register_tools(self, modules: list):
        """Register tools from a given module

        :param modules: list of modules with tool function definitions

        """
        for module in modules:
            for fname, fn in inspect.getmembers(module, inspect.isfunction):
                if fname.endswith("_tool"):
                    self.mcp.add_tool(fn)
