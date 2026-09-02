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


@tool_meta(ToolInfo(tags={"testing"}, checkpaths=["path"]))
def checkpath_tool(path: str) -> str:
    """A tool whose path argument must be permission-checked."""
    return path


@tool_meta(ToolInfo(tags={"testing"}))
def nopath_tool(x: str) -> str:
    """A tool with no path arguments."""
    return x


@tool_meta(ToolInfo(tags={"testing"}, read_only=True, open_world=True))
def annotated_tool(param: str) -> str:
    """A tool carrying standard MCP annotation hints."""
    return param


@tool_meta(ToolInfo(tags={"testing"}, destructive=True))
def destructive_tool(param: str) -> str:
    """A tool marked destructive."""
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


@pytest.mark.asyncio
async def test_register_tools_puts_checkpaths_in_meta():
    """checkpaths declared on ToolInfo must reach clients on the Tool meta."""
    server = FastMCP("test-server")
    register_tools(server, [sys.modules[__name__]])

    tools = await server.list_tools()
    checkpath_tools = [t for t in tools if t.name == "checkpath_tool"]
    assert len(checkpath_tools) == 1
    assert checkpath_tools[0].meta is not None
    assert checkpath_tools[0].meta["checkpaths"] == ["path"]


@pytest.mark.asyncio
async def test_register_tools_without_checkpaths_omits_key():
    """Tools that declare no checkpaths must not carry the key in meta."""
    server = FastMCP("test-server")
    register_tools(server, [sys.modules[__name__]])

    tools = await server.list_tools()
    nopath_tools = [t for t in tools if t.name == "nopath_tool"]
    assert len(nopath_tools) == 1
    assert "checkpaths" not in (nopath_tools[0].meta or {})


@pytest.mark.asyncio
async def test_register_tools_folds_annotation_hints():
    """ToolInfo hint fields must reach clients as standard ToolAnnotations."""
    server = FastMCP("test-server")
    register_tools(server, [sys.modules[__name__]])

    tools = {t.name: t for t in await server.list_tools()}

    annotated = tools["annotated_tool"]
    assert annotated.annotations is not None
    assert annotated.annotations.readOnlyHint is True
    assert annotated.annotations.openWorldHint is True
    assert annotated.annotations.destructiveHint is None

    destructive = tools["destructive_tool"]
    assert destructive.annotations is not None
    assert destructive.annotations.destructiveHint is True
    assert destructive.annotations.readOnlyHint is None
    assert destructive.annotations.openWorldHint is None


@pytest.mark.asyncio
async def test_register_tools_no_annotations_when_unset():
    """A tool declaring no hints must not carry an annotations object."""
    server = FastMCP("test-server")
    register_tools(server, [sys.modules[__name__]])

    tools = {t.name: t for t in await server.list_tools()}

    plain = tools["sample_tool"]
    assert plain.annotations is None
