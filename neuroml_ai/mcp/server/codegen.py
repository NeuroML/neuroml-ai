#!/usr/bin/env python3
"""
MCP server for NeuroML code generation

File: neuroml_ai/mcp/server/codegen.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from neuroml_ai.mcp.utils import register_tools
from neuroml_ai.mcp.tools import codegen_tools
from textwrap import dedent
from fastmcp import FastMCP


def main():
    """main runner"""
    usage = dedent(
        """
        NeuroML coding assistant server.

        """
    )
    mcp = FastMCP("nml_codegen", instructions=usage, port=8542)
    register_tools(mcp, [codegen_tools])
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
