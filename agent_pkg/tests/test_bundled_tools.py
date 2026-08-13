#!/usr/bin/env python3
"""
Tests for the bundled Klea Agent tools server and its wrappers.

File: agent_pkg/tests/test_bundled_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import pytest
from fastmcp import Client
from klea_agent.tools.bundled import bundle_server
from klea_agent.tools.wrappers import list_files, web_fetch

logger = logging.getLogger(__name__)


class MockContext:
    """Test stub replacing fastmcp.Context for direct wrapper calls."""

    def __init__(self, **kwargs):
        self.lifespan_context = {}
        self.lifespan_context.update(kwargs)


class _FakeResponse:
    def __init__(self, text, status=200, content_type="text/html"):
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._text = text

    async def text(self) -> str:
        return self._text


class _FakeGetCM:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        if self._error is not None:
            raise self._error
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def get(self, url, **kwargs):
        return _FakeGetCM(response=self._response, error=self._error)


@pytest.mark.asyncio
async def test_bundled_server_exposes_tools():
    async with Client(transport=bundle_server) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        logger.debug(f"{names = }")
        assert "web_fetch" in names
        assert "list_files" in names


@pytest.mark.asyncio
async def test_web_fetch_missing_session():
    result = await web_fetch(ctx=MockContext(), url="https://example.com")
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert "session" in result["error"].lower()


@pytest.mark.asyncio
async def test_web_fetch_with_session():
    fake = _FakeResponse(
        "<html><body><h1>Hello</h1><script>x()</script></body></html>",
        content_type="text/html; charset=utf-8",
    )
    ctx = MockContext(aiohttp_session=_FakeSession(response=fake))
    result = await web_fetch(ctx=ctx, url="https://example.com")
    logger.debug(f"{result = }")
    assert result["status_code"] == 200
    assert "Hello" in result["content"]
    assert "<h1>" not in result["content"]


@pytest.mark.asyncio
async def test_list_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.md").write_text("")

    result = await list_files(path=str(tmp_path), pattern="*")
    logger.debug(f"{result = }")

    names = {f["path"].split("/")[-1] for f in result["files"]}
    assert "a.py" in names
    assert "b.md" in names
    assert result["error"] == ""
