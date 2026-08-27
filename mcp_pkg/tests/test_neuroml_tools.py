#!/usr/bin/env python3
"""
Test NeuroML tools

File: mcp_pkg/tests/test_neuroml_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import httpx
import pytest
import pytest_asyncio
from neuroml_mcp.tools.neuroml_tools import (
    get_models_from_neuromldb,
    get_repositories_from_open_source_brain,
)

logger = logging.getLogger(__name__)


class MockContext:
    """Test stub replacing fastmcp.Context"""

    def __init__(self):
        self.lifespan_context = {}

    def set_state(self, key, val):
        self.lifespan_context[key] = val


@pytest_asyncio.fixture
async def http_ctx():
    async with httpx.AsyncClient() as ses:
        ctx = MockContext()
        ctx.set_state("http_session", ses)
        yield ctx


@pytest.mark.asyncio
async def test_get_models_from_neuromldb_download(http_ctx):
    from pathlib import Path

    model = "NMLCL000595"
    res = await get_models_from_neuromldb(
        ctx=http_ctx, search_query=model, num=1, download=True
    )
    logger.debug(f"{res = }")
    # Wrapper now returns ToolResult per MCP spec (isError handling)
    assert not res.is_error
    data = res.structured_content
    assert len(data) == 1

    # Should download model
    assert model in list(data.keys())

    m = data[model]
    resource = m["resource"]
    # ToolResult serialises Path to string via FastMCP; handle both
    path = Path(resource) if isinstance(resource, str) else resource
    assert path.exists()
    assert m["Type"] == "Cell"
    assert m["Publication_Year"] == 2015


@pytest.mark.asyncio
async def test_get_models_from_neuromldb_nodownload(http_ctx):
    model = "NMLCL000595"
    res = await get_models_from_neuromldb(
        ctx=http_ctx, search_query=model, num=1, download=False
    )
    logger.debug(f"{res = }")
    assert not res.is_error
    data = res.structured_content
    assert len(data) == 1

    assert model in list(data.keys())

    m = data[model]
    assert m["resource"] is None
    assert m["Type"] == "Cell"
    assert m["Publication_Year"] == 2015


@pytest.mark.asyncio
async def test_get_repositories_from_open_source_brain(http_ctx):
    # Test basic functionality with a simple search
    search_term = "cerebellum"
    res = await get_repositories_from_open_source_brain(
        ctx=http_ctx,
        search_query=search_term,
        search_data=True,
        search_models=True,
        num=2,
    )
    logger.debug(f"{res = }")

    # Wrapper now returns ToolResult per MCP spec
    assert not res.is_error
    data = res.structured_content
    assert isinstance(data, dict)

    # Should have some results (may be empty depending on search)
    # Just checking it doesn't crash and returns proper structure
    assert "Error" not in data or isinstance(data["Error"], str)


@pytest.mark.asyncio
async def test_get_repositories_from_open_source_brain_no_results(http_ctx):
    # Test with a search term that likely won't return results
    search_term = "nonexistent_search_term_12345"
    res = await get_repositories_from_open_source_brain(
        ctx=http_ctx,
        search_query=search_term,
        search_data=True,
        search_models=True,
        num=1,
    )
    logger.debug(f"{res = }")

    # Wrapper now returns ToolResult per MCP spec
    assert not res.is_error
    data = res.structured_content
    assert isinstance(data, dict)

    # Should not crash, even if no results are found
    assert "Error" not in data or isinstance(data["Error"], str)
