#!/usr/bin/env python3
"""
Tests for the shared MCP tool registry.

File: utils_pkg/tests/test_mcp_registry.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import sys
import types

import pytest
from fastmcp import FastMCP
from klea_utils.mcp.registry import register_tools, tool_meta
from klea_utils.mcp.schemas import ToolInfo

logger = logging.getLogger(__name__)


@tool_meta(ToolInfo(tags={"testing"}))
def sample_tool(param: str) -> str:
    """A tool function."""
    return param


def plain_helper() -> str:
    """A helper that is not a tool."""
    return "helper"


def _private_helper() -> str:
    """A private helper that is not a tool."""
    return "private"


@pytest.mark.asyncio
async def test_register_tools_only_registers_decorated():
    server = FastMCP("test-server")
    register_tools(server, [sys.modules[__name__]])

    tools = await server.list_tools()
    names = [t.name for t in tools]
    logger.debug(f"{names = }")

    assert "sample_tool" in names
    assert "plain_helper" not in names
    assert "_private_helper" not in names


@pytest.mark.asyncio
async def test_register_tools_ignores_imported_decorated_functions():
    """A decorated function attached to a module but defined elsewhere is
    not registered (the __module__ filter)."""
    mod = types.ModuleType("fake_tool_module")

    @tool_meta(ToolInfo(tags={"testing"}))
    def imported_tool(a: str) -> str:
        return a

    mod.__dict__["imported_tool"] = imported_tool

    server = FastMCP("test-server")
    register_tools(server, [mod])

    tools = await server.list_tools()
    names = [t.name for t in tools]
    logger.debug(f"{names = }")

    assert "imported_tool" not in names
