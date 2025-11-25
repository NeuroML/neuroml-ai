#!/usr/bin/env python3
"""
MCP server for answers from docs

File: neuroml_ai/mcp/server/answers.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from neuroml_ai.mcp.server.server import MCPServer
from neuroml_ai.mcp.tools import answer_tools


if __name__ == "__main__":
    answers_server = MCPServer("answers")
    answers_server.register_tools([answer_tools])
    mcp = answers_server.mcp
    mcp.run(transport="stdio")
