#!/usr/bin/env python3
"""
Tests for the bundled Klea Agent tools server and its wrappers.

File: agent_pkg/tests/test_bundled_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import httpx
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
    """Minimal httpx-like response used to drive :func:`web_fetch`."""

    def __init__(self, body, status=200, content_type="text/html"):
        self.status_code = status
        self.headers = {"content-type": content_type}
        self._body = body.encode("utf-8") if isinstance(body, str) else body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://example.com")
            response = httpx.Response(
                self.status_code, request=request, content=self._body
            )
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    async def aiter_bytes(self):
        yield self._body


class _FakeStreamCM:
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

    def stream(self, method, url, **kwargs):
        return _FakeStreamCM(response=self._response, error=self._error)


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
    ctx = MockContext(http_session=_FakeSession(response=fake))
    result = await web_fetch(ctx=ctx, url="https://example.com")
    logger.debug(f"{result = }")
    assert result["status_code"] == 200
    assert "Hello" in result["content"]
    assert "<h1>" not in result["content"]


@pytest.mark.asyncio
async def test_list_files(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.md").write_text("")
    monkeypatch.chdir(tmp_path)

    result = await list_files(path=".", pattern="*")
    logger.debug(f"{result = }")

    names = {f["path"].split("/")[-1] for f in result["files"]}
    assert "a.py" in names
    assert "b.md" in names
    assert result["error"] == ""
