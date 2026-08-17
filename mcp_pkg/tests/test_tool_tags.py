#!/usr/bin/env python3
"""
Tests of the neuroml_mcp tool tag vocabulary.

File: mcp_pkg/tests/test_tool_tags.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastmcp.client import Client
from neuroml_mcp.server.main import create_server

#: The documented tag vocabulary (see docs/concepts/mcp.rst): scope tags
#: (local/web) plus functional/domain tags.  Every tool carries a domain
#: tag; the ``testing`` tag is gone.  Tags are used for Klea's config
#: filtering only (which domain's tools to expose); behavioral intent
#: (read-only / destructive) is carried by standard MCP tool annotations,
#: not by tags.
_EXPECTED_TAGS = {
    "dummy_code": {"neuroml", "echo"},
    "dummy": {"neuroml", "echo"},
    "list_files": {"neuroml", "local", "files"},
    "run_python_code": {"neuroml", "local", "code"},
    "create_new_NeuroML_model": {"neuroml"},
    "run_lems_simulation": {"neuroml", "local", "code"},
    "get_models_from_neuromldb": {"neuroml", "web", "neuroml-db"},
    "get_repositories_from_open_source_brain": {"neuroml", "web", "osb"},
}


async def _tools_by_name():
    mcp = await create_server()
    async with Client(transport=mcp) as client:
        tools = await client.list_tools()
    return {t.name: t for t in tools}


def _tags(tool) -> set[str]:
    """Read tags as a client sees them (in the tool's fastmcp meta)."""
    meta = tool.meta or {}
    return set(meta.get("fastmcp", {}).get("tags", []))


async def test_no_testing_tag_remains():
    tools = await _tools_by_name()
    for name, tool in tools.items():
        assert "testing" not in _tags(tool), name


async def test_tool_tag_vocabulary():
    tools = await _tools_by_name()
    assert set(tools) == set(_EXPECTED_TAGS)
    for name, expected in _EXPECTED_TAGS.items():
        assert name in tools, f"missing tool: {name}"
        assert _tags(tools[name]) == expected, name


async def test_scope_tags_present_on_io_tools():
    """I/O tools must be marked local or web so deployments can enable only
    the tools they want."""
    tools = await _tools_by_name()
    local = {"list_files", "run_python_code", "run_lems_simulation"}
    web = {"get_models_from_neuromldb", "get_repositories_from_open_source_brain"}
    for name in local:
        assert "local" in _tags(tools[name]), name
    for name in web:
        assert "web" in _tags(tools[name]), name
