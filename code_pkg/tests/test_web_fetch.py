#!/usr/bin/env python3
"""
Tests for the bundled web_fetch tool.

File: code_pkg/tests/test_web_fetch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import httpx
import pytest
from fastmcp import Client

from klea_code.tools.bundled import bundle_server
from klea_code.tools.web_fetch import web_fetch


class _FakeResponse:
    def __init__(
        self,
        text: str,
        status_code: int = 200,
        content_type: str = "text/html",
    ):
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": content_type}

    @property
    def is_error(self) -> bool:
        return self.status_code >= 400


class _FakeAsyncClient:
    def __init__(self, response=None, error=None, **kwargs):
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, headers=None):
        if self._error is not None:
            raise self._error
        return self._response


@pytest.mark.asyncio
async def test_web_fetch_rejects_invalid_url():
    result = await web_fetch(url="not-a-url")
    assert result["content"] == ""
    assert result["status_code"] is None
    assert "http://" in result["error"] or "https://" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_success(monkeypatch):
    fake = _FakeResponse("<html>hello</html>", status_code=200)

    def _client_factory(**kwargs):
        return _FakeAsyncClient(response=fake)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    result = await web_fetch(url="https://example.com")
    assert result["status_code"] == 200
    assert result["content"] == "<html>hello</html>"
    assert result["truncated"] is False
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_truncates(monkeypatch):
    fake = _FakeResponse("abcdefghij", status_code=200)

    def _client_factory(**kwargs):
        return _FakeAsyncClient(response=fake)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    result = await web_fetch(url="https://example.com", max_chars=4)
    assert result["content"] == "abcd"
    assert result["truncated"] is True
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_http_error(monkeypatch):
    fake = _FakeResponse("missing", status_code=404)

    def _client_factory(**kwargs):
        return _FakeAsyncClient(response=fake)

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    result = await web_fetch(url="https://example.com/missing")
    assert result["status_code"] == 404
    assert "404" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_timeout(monkeypatch):
    def _client_factory(**kwargs):
        return _FakeAsyncClient(error=httpx.TimeoutException("timeout"))

    monkeypatch.setattr(httpx, "AsyncClient", _client_factory)

    result = await web_fetch(url="https://example.com", timeout=1.0)
    assert result["content"] == ""
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_bundled_server_exposes_web_fetch():
    async with Client(transport=bundle_server) as client:
        tools = await client.list_tools()
        names = [t.name for t in tools]
        assert "list_files" in names
        assert "web_fetch" in names
