#!/usr/bin/env python3
"""
MCP server for NeuroML code generation

File: neuroml_ai/mcp/server/main.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from neuroml_ai.mcp.utils import register_tools
from neuroml_ai.mcp.tools import code_tools
from textwrap import dedent
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse


def create_server():
    """main server creator"""
    usage = dedent(
        """
        NeuroML coding assistant server.

        """
    )
    mcp = FastMCP("nml_ai", instructions=usage, port=8542)
    register_tools(mcp, [code_tools])

    @mcp.custom_route("/health", methods=["GET"])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    @mcp.custom_route("/list", methods=["GET"])
    async def tool_list(request: Request) -> JSONResponse:
        all_tools = await (mcp.get_tools())
        tools_description = [{str(tool.name): str(tool.description)} for name, tool in all_tools.items()]
        resp = {"registered_tools": tools_description}
        return JSONResponse(resp)

    return mcp


if __name__ == "__main__":
    mcp = create_server()
    mcp.run(transport="streamable-http")
