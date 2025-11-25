#!/usr/bin/env python3
"""
MCP server for NeuroML code generation

File: neuroml_ai/mcp/server/answers.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from neuroml_ai.mcp.server.server import MCPServerFactory
from neuroml_ai.mcp.tools import codegen_tools


if __name__ == "__main__":
    answers_server = MCPServerFactory("nml_codegen")
    answers_server.register_tools([codegen_tools])
    mcp = answers_server.mcp
    mcp.run(transport="stdio")
