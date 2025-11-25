#!/usr/bin/env python3
"""
Test MCP

File: test_mcp.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest

import asyncio
from neuroml_ai.mcp.server.server import MCPServerFactory
from neuroml_ai.mcp.tools import codegen_tools


class TestMCP(unittest.TestCase):
    """Smoke tests for MCP"""

    def test_answers_server(self):
        """Test the Answers MCP server"""
        aserver = MCPServerFactory("codegen")
        aserver.register_tools([codegen_tools])
        mcp = aserver.mcp
        all_tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in all_tools]
        self.assertIn("dummy_tool", tool_names)

if __name__ == "__main__":
    unittest.main()
