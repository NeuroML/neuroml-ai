#!/usr/bin/env python3
"""
Test MCP

File: test_mcp.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
import unittest
from fastmcp import FastMCP

import asyncio
from neuroml_ai.mcp.tools import codegen_tools
from neuroml_ai.mcp.utils import register_tools


class TestMCP(unittest.TestCase):
    """Smoke tests for MCP"""

    @pytest.mark.skip(reason="Needs reimplementation")
    def test_answers_server(self):
        """Test the Answers MCP server"""
        mcp = FastMCP("nml_codegen", instructions="Dummy", port=8542)
        register_tools(mcp, [codegen_tools])
        mcp.run(transport="streamable-http")

        all_tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in all_tools]
        self.assertIn("dummy_tool", tool_names)

if __name__ == "__main__":
    unittest.main()
