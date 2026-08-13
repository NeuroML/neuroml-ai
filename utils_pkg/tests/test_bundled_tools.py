#!/usr/bin/env python3
"""
Tests for shared MCP tool implementations.

File: utils_pkg/tests/test_bundled_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import aiohttp
import pytest
from klea_utils.mcp.tools.list_files import list_files
from klea_utils.mcp.tools.web_fetch import web_fetch

logger = logging.getLogger(__name__)


class _FakeResponse:
    def __init__(
        self,
        text: str,
        status: int = 200,
        content_type: str = "text/html",
    ):
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
async def test_web_fetch_rejects_invalid_url():
    result = await web_fetch(session=_FakeSession(), url="not-a-url")
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert result["status_code"] is None
    assert "http://" in result["error"] or "https://" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_missing_session():
    result = await web_fetch(session=None, url="https://example.com")
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert result["status_code"] is None
    assert "session" in result["error"].lower()


@pytest.mark.asyncio
async def test_web_fetch_success_strips_html():
    fake = _FakeResponse(
        "<html><body><h1>Hello</h1><script>x()</script></body></html>",
        status=200,
        content_type="text/html; charset=utf-8",
    )
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com",
    )
    logger.debug(f"{result = }")
    assert result["status_code"] == 200
    assert "Hello" in result["content"]
    assert "<h1>" not in result["content"]
    assert "x()" not in result["content"]
    assert result["truncated"] is False
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_plain_text_passthrough():
    fake = _FakeResponse("plain body", status=200, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com/raw.txt",
    )
    logger.debug(f"{result = }")
    assert result["content"] == "plain body"
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_truncates():
    fake = _FakeResponse("abcdefghij", status=200, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com",
        max_chars=4,
    )
    logger.debug(f"{result = }")
    assert result["content"] == "abcd"
    assert result["truncated"] is True
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_http_error():
    fake = _FakeResponse("missing", status=404, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com/missing",
    )
    logger.debug(f"{result = }")
    assert result["status_code"] == 404
    assert "404" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_timeout():
    result = await web_fetch(
        session=_FakeSession(error=TimeoutError()),
        url="https://example.com",
        timeout=1.0,
    )
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_client_error():
    result = await web_fetch(
        session=_FakeSession(error=aiohttp.ClientConnectionError("refused")),
        url="https://example.com",
    )
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert "refused" in result["error"]


def test_list_files_rejects_dotdot():
    result = list_files(path="..")
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert ".." in result["error"]


def test_list_files_basic(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.md").write_text("")
    (tmp_path / "sub").mkdir()

    result = list_files(path=str(tmp_path), pattern="*")

    logger.debug(f"{len(result['files']) = }")
    logger.debug(f"{result['error'] = }")

    names = {f["path"].split("/")[-1] for f in result["files"]}
    assert "a.py" in names
    assert "b.md" in names
    assert "sub" in names
    assert result["error"] == ""
    assert result["truncated"] == "False"


def test_list_files_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("")

    result = list_files(path=str(tmp_path), pattern="*.py", recursive=True)

    logger.debug(f"{len(result['files']) = }")
    logger.debug(f"{result['error'] = }")

    names = {f["path"].split("/")[-1] for f in result["files"]}
    assert "c.py" in names
    assert result["error"] == ""


def test_list_files_truncates(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("")

    result = list_files(path=str(tmp_path), pattern="*.txt", max_results=2)

    logger.debug(f"{len(result['files']) = }")
    logger.debug(f"{result['truncated'] = }")

    assert len(result["files"]) == 2
    assert result["truncated"] == "True"
