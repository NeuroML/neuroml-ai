#!/usr/bin/env python3
"""
Test MCP

File: test_mcp.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest

import asyncio
from neuroml_ai.mcp.server.server import MCPServer
from neuroml_ai.mcp.tools import answer_tools


class TestMCP(unittest.TestCase):
    """Smoke tests for MCP"""

    def test_answers_server(self):
        """Test the Answers MCP server"""
        answers_server = MCPServer("answers")
        answers_server.register_tools([answer_tools])
        mcp = answers_server.mcp
        all_tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in all_tools]
        self.assertIn("dummy_tool", tool_names)

if __name__ == "__main__":
    unittest.main()
