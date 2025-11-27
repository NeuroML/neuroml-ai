#!/usr/bin/env python3
"""
MCP server for NeuroML code generation

File: neuroml_ai/mcp/server/codegen.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from neuroml_ai.mcp.server.factory import MCPServerFactory
from neuroml_ai.mcp.tools import codegen_tools

def main():
    """main runner"""
    answers_server = MCPServerFactory("nml_codegen")
    answers_server.register_tools([codegen_tools])
    mcp = answers_server.mcp
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
