#!/usr/bin/env python3
"""
Tests for shared MCP tool implementations.

File: utils_pkg/tests/test_bundled_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import os
import time

import httpx
import klea_utils.api.utils as api_utils
import pytest
from klea_utils.mcp.tools import web_fetch as web_fetch_module
from klea_utils.mcp.tools.download_file import download_file, download_file_to_cache
from klea_utils.mcp.tools.list_files import list_files
from klea_utils.mcp.tools.web_fetch import web_fetch

logger = logging.getLogger(__name__)


class _FakeResponse:
    """Minimal httpx-like response used to drive :func:`web_fetch`."""

    def __init__(
        self,
        body: str | bytes,
        status: int = 200,
        content_type: str = "text/html",
    ):
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

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace")

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
        self.calls: list[tuple[str, str, dict]] = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _FakeStreamCM(response=self._response, error=self._error)

    async def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


class _FlakySession:
    """Session that fails with the given status twice, then succeeds."""

    def __init__(self, body: str = "ok after retries"):
        self._body = body
        self.attempts = 0
        self.calls: list[tuple[str, str, dict]] = []

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        self.attempts += 1
        if self.attempts < 3:
            response = _FakeResponse("down", status=503, content_type="text/plain")
        else:
            response = _FakeResponse(self._body, status=200, content_type="text/plain")
        return _FakeStreamCM(response=response)

    async def get(self, url, **kwargs):
        # Not used by the web_fetch tests this drives; satisfies SessionLike.
        raise AssertionError("_FlakySession.get is not exercised in these tests")


@pytest.fixture(autouse=True)
def _fast_waits(monkeypatch):
    # Neutralise the exponential backoff so tests do not sleep.
    monkeypatch.setattr(api_utils, "wait_random_exponential", lambda **kw: 0.0)


@pytest.fixture(autouse=True)
def _hermetic_user_agents(monkeypatch, request):
    # Pin UA resolution so web_fetch tests never hit the network or disk.
    # The UA-resolution logic itself is covered by dedicated tests below.
    if request.cls is TestResolveUserAgents:
        return
    monkeypatch.setattr(
        web_fetch_module, "_resolve_user_agents", _async_ok([_TEST_USER_AGENT])
    )
    monkeypatch.setattr(web_fetch_module, "_UA_CACHE_PATH", "unused")
    monkeypatch.setattr(web_fetch_module, "_UA_LIST", None)
    monkeypatch.setattr(web_fetch_module, "_UA_RESOLVED_AT", 0.0)


_TEST_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"


def _async_ok(value):
    async def _fn(*args, **kwargs):
        return value

    return _fn


async def _async_fail(*args, **kwargs):
    raise httpx.ConnectError("offline")


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
    assert result["download_truncated"] is False
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
    assert result["download_truncated"] is False
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_download_cap_flags_download_truncated():
    fake = _FakeResponse("abcdefghij", status=200, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com",
        max_download_bytes=4,
    )
    logger.debug(f"{result = }")
    assert result["content"] == "abcd"
    assert result["truncated"] is False
    assert result["download_truncated"] is True
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_web_fetch_http_error_drops_body():
    fake = _FakeResponse("missing", status=404, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com/missing",
    )
    logger.debug(f"{result = }")
    assert result["status_code"] == 404
    assert result["content"] == ""
    assert "404" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_timeout():
    result = await web_fetch(
        session=_FakeSession(error=TimeoutError()),
        url="https://example.com",
        timeout=1.0,
        retries=1,
    )
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert "timed out" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_client_error():
    result = await web_fetch(
        session=_FakeSession(error=httpx.ConnectError("refused")),
        url="https://example.com",
        retries=1,
    )
    logger.debug(f"{result = }")
    assert result["content"] == ""
    assert "refused" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_retries_transient_5xx_then_succeeds():
    session = _FlakySession()
    result = await web_fetch(
        session=session,
        url="https://example.com/flaky",
        retries=5,
    )
    logger.debug(f"{result = }")
    assert result["status_code"] == 200
    assert result["content"] == "ok after retries"
    assert session.attempts == 3


@pytest.mark.asyncio
async def test_web_fetch_retries_exhausted():
    session = _FlakySession()
    result = await web_fetch(
        session=session,
        url="https://example.com/flaky",
        retries=2,
    )
    logger.debug(f"{result = }")
    # Two attempts, still failing with 503.
    assert session.attempts == 2
    assert result["status_code"] == 503
    assert result["content"] == ""
    assert "503" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_uses_browser_user_agent():
    fake = _FakeResponse("hi", status=200, content_type="text/plain")
    session = _FakeSession(response=fake)
    await web_fetch(session=session, url="https://example.com")
    _, _, kwargs = session.calls[0]
    ua = kwargs["headers"]["User-Agent"]
    assert ua == _TEST_USER_AGENT
    assert "Mozilla/5.0" in ua
    assert "klea" not in ua.lower()


@pytest.mark.asyncio
async def test_web_fetch_sends_accept_language():
    fake = _FakeResponse("hi", status=200, content_type="text/plain")
    session = _FakeSession(response=fake)
    await web_fetch(session=session, url="https://example.com")
    _, _, kwargs = session.calls[0]
    assert kwargs["headers"]["Accept-Language"] == "en-US,en;q=0.9"


class _ChallengeSession:
    """First stream answers a Cloudflare challenge, second one succeeds."""

    def __init__(self, body: str = "ok after challenge"):
        self._body = body
        self.calls: list[tuple[str, str, dict]] = []
        self.stream_count = 0

    def stream(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        self.stream_count += 1
        if self.stream_count == 1:
            response = _FakeResponse("challenge", status=403, content_type="text/html")
            response.headers["cf-mitigated"] = "challenge"
        else:
            response = _FakeResponse(self._body, status=200, content_type="text/plain")
        return _FakeStreamCM(response=response)

    async def get(self, url, **kwargs):
        # Not used by the web_fetch test this drives; satisfies SessionLike.
        raise AssertionError("_ChallengeSession.get is not exercised in these tests")


@pytest.mark.asyncio
async def test_web_fetch_cloudflare_challenge_retries_honest_ua():
    session = _ChallengeSession()
    result = await web_fetch(session=session, url="https://example.com")
    logger.debug(f"{result = }")
    assert session.stream_count == 2
    assert result["status_code"] == 200
    assert result["content"] == "ok after challenge"
    # First attempt used the browser UA; retry used the honest client UA.
    browser_ua = session.calls[0][2]["headers"]["User-Agent"]
    honest_ua = session.calls[1][2]["headers"]["User-Agent"]
    assert browser_ua == _TEST_USER_AGENT
    assert honest_ua == web_fetch_module._honest_user_agent()
    assert honest_ua.startswith(web_fetch_module._HONEST_UA_PREFIX)


def test_honest_user_agent_uses_package_version():
    ua = web_fetch_module._honest_user_agent()
    assert ua.startswith("klea-web-fetch/")
    # Version tag resolves to the installed klea_utils version or 'dev'.
    tag = ua.split("/", 1)[1]
    assert tag in ("dev", "0.4.0")


def test_honest_user_agent_fallback_when_no_metadata(monkeypatch):
    import importlib.metadata

    monkeypatch.setattr(web_fetch_module, "_honest_user_agent_cache", None)

    def _no_version(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _no_version)
    ua = web_fetch_module._honest_user_agent()
    assert ua == "klea-web-fetch/dev"


@pytest.mark.asyncio
async def test_web_fetch_ssrf_blocks_private_hosts():
    for url in ("http://127.0.0.1/x", "http://localhost/x", "http://169.254.169.254/x"):
        result = await web_fetch(session=_FakeSession(), url=url)
        logger.debug(f"{url = } -> {result['error'] = }")
        assert result["content"] == ""
        assert result["status_code"] is None
        assert "Blocked request to private/internal address" in result["error"]


@pytest.mark.asyncio
async def test_web_fetch_ssrf_allow_internal_flag():
    fake = _FakeResponse("hi", status=200, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="http://127.0.0.1/x",
        allow_internal_hosts=True,
    )
    assert result["status_code"] == 200
    assert result["content"] == "hi"


@pytest.mark.asyncio
async def test_web_fetch_ssrf_allows_public_host():
    fake = _FakeResponse("public", status=200, content_type="text/plain")
    result = await web_fetch(
        session=_FakeSession(response=fake),
        url="https://example.com",
    )
    assert result["status_code"] == 200
    assert result["content"] == "public"


class TestDownloadFile:
    """Tests of the shared download_file implementation."""

    async def test_download_file_writes_target(self, tmp_path):
        fake = _FakeResponse("file body", status=200, content_type="text/plain")
        session = _FakeSession(response=fake)
        target = tmp_path / "out.txt"
        result = await download_file(
            session=session,
            url="https://example.com/f.txt",
            file_path=target,
            project_root=str(tmp_path),
        )
        assert result == target
        assert target.read_text() == "file body"

    async def test_download_file_missing_session(self, tmp_path):
        result = await download_file(
            session=None, url="https://example.com/f.txt", file_path=tmp_path / "x"
        )
        assert result is None

    async def test_download_file_http_error(self, tmp_path):
        fake = _FakeResponse("missing", status=404, content_type="text/plain")
        session = _FakeSession(response=fake)
        result = await download_file(
            session=session,
            url="https://example.com/f.txt",
            file_path=tmp_path / "x",
            project_root=str(tmp_path),
        )
        assert result is None
        assert not (tmp_path / "x").exists()

    async def test_download_file_to_cache(self, tmp_path):
        fake = _FakeResponse("cached body", status=200, content_type="text/plain")
        session = _FakeSession(response=fake)
        cache_dir = tmp_path / "cache"
        result = await download_file_to_cache(
            session=session,
            url="https://example.com/f.txt",
            cache_dir=cache_dir,
            file_name="f.txt",
        )
        assert result is not None
        assert result == cache_dir / "f.txt"
        assert result.read_text() == "cached body"

    async def test_download_file_denied_outside_project(self, tmp_path):
        fake = _FakeResponse("file body", status=200, content_type="text/plain")
        session = _FakeSession(response=fake)
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        result = await download_file(
            session=session,
            url="https://example.com/f.txt",
            file_path=outside,
            project_root=str(root),
        )
        assert result is None
        assert not outside.exists()


def test_list_files_rejects_dotdot():
    result = list_files(path="..")
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert ".." in result["error"]


def test_list_files_basic(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.md").write_text("")
    (tmp_path / "sub").mkdir()

    result = list_files(path=str(tmp_path), pattern="*", project_root=str(tmp_path))

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

    result = list_files(
        path=str(tmp_path),
        pattern="*.py",
        recursive=True,
        project_root=str(tmp_path),
    )

    logger.debug(f"{len(result['files']) = }")
    logger.debug(f"{result['error'] = }")

    names = {f["path"].split("/")[-1] for f in result["files"]}
    assert "c.py" in names
    assert result["error"] == ""


def test_list_files_truncates(tmp_path):
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("")

    result = list_files(
        path=str(tmp_path), pattern="*.txt", max_results=2, project_root=str(tmp_path)
    )

    logger.debug(f"{len(result['files']) = }")
    logger.debug(f"{result['truncated'] = }")

    assert len(result["files"]) == 2
    assert result["truncated"] == "True"


def test_list_files_max_depth(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b" / "c.py").write_text("")
    (tmp_path / "a" / "top.py").write_text("")

    result = list_files(
        path=str(tmp_path),
        pattern="*.py",
        recursive=True,
        max_depth=1,
        project_root=str(tmp_path),
    )
    names = {f["path"].split("/")[-1] for f in result["files"]}
    logger.debug(f"max_depth=1: {names = }")
    assert names == set()

    result = list_files(
        path=str(tmp_path),
        pattern="*.py",
        recursive=True,
        max_depth=2,
        project_root=str(tmp_path),
    )
    names = {f["path"].split("/")[-1] for f in result["files"]}
    logger.debug(f"max_depth=2: {names = }")
    assert "top.py" in names
    assert "c.py" not in names

    result = list_files(
        path=str(tmp_path),
        pattern="*.py",
        recursive=True,
        max_depth=3,
        project_root=str(tmp_path),
    )
    names = {f["path"].split("/")[-1] for f in result["files"]}
    logger.debug(f"max_depth=3: {names = }")
    assert "top.py" in names
    assert "c.py" in names


def test_list_files_include_directories(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.py").write_text("")

    result = list_files(
        path=str(tmp_path),
        pattern="*",
        include_directories=False,
        project_root=str(tmp_path),
    )
    names = {f["path"].split("/")[-1] for f in result["files"]}
    logger.debug(f"{names = }")
    assert "f.py" in names
    assert "sub" not in names


def test_list_files_include_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "f.py").write_text("")

    result = list_files(
        path=str(tmp_path),
        pattern="*",
        include_files=False,
        project_root=str(tmp_path),
    )
    names = {f["path"].split("/")[-1] for f in result["files"]}
    logger.debug(f"{names = }")
    assert "sub" in names
    assert "f.py" not in names


def test_list_files_symlink_not_recursed(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "target.py").write_text("")
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)

    result = list_files(
        path=str(tmp_path), pattern="*", recursive=True, project_root=str(tmp_path)
    )
    logger.debug(f"{result['files'] = }")
    by_path = {f["path"].split("/")[-1]: f for f in result["files"]}
    assert by_path["link"]["type"] == "link"
    assert by_path["real"]["type"] == "directory"
    assert "target.py" in by_path  # reached via the real dir, not the link
    assert not any(str(f["path"]).endswith("link/target.py") for f in result["files"])

    result = list_files(
        path=str(tmp_path), pattern="*", include_files=False, project_root=str(tmp_path)
    )
    by_path = {f["path"].split("/")[-1]: f for f in result["files"]}
    logger.debug(f"{by_path = }")
    assert by_path["link"]["type"] == "link"
    assert by_path["real"]["type"] == "directory"
    assert "target.py" not in by_path


def test_list_files_denied_outside_project(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "s.txt").write_text("s")
    result = list_files(path=str(outside), pattern="*", project_root=str(root))
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert "denied" in result["error"].lower()


def test_list_files_denied_absolute_escape(tmp_path):
    result = list_files(path="/etc", pattern="*", project_root=str(tmp_path))
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert "denied" in result["error"].lower()


class TestResolveUserAgents:
    """Tests of the UA resolution logic (offline fallback, cache, fetch)."""

    @pytest.fixture(autouse=True)
    def _reset_state(self, monkeypatch, tmp_path):
        monkeypatch.setattr(web_fetch_module, "_UA_LIST", None)
        monkeypatch.setattr(web_fetch_module, "_UA_RESOLVED_AT", 0.0)
        monkeypatch.setattr(
            web_fetch_module, "_UA_CACHE_PATH", tmp_path / "user_agents.json"
        )

    async def test_offline_no_cache_uses_hardcoded_fallback(self, monkeypatch):
        monkeypatch.setattr(web_fetch_module, "_fetch_user_agents", _async_fail)
        agents = await web_fetch_module._resolve_user_agents()
        assert agents == [web_fetch_module._FALLBACK_USER_AGENT]

    async def test_fresh_cache_used_without_fetch(self, monkeypatch):
        cached = ["Mozilla/5.0 (X11; Linux x86_64) CachedAgent/1.0"]
        path = web_fetch_module._UA_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached))
        fetched = False

        async def _fetch():
            nonlocal fetched
            fetched = True
            return ["never used"]

        monkeypatch.setattr(web_fetch_module, "_fetch_user_agents", _fetch)
        agents = await web_fetch_module._resolve_user_agents()
        assert agents == cached
        assert fetched is False

    async def test_fetch_success_writes_cache(self, monkeypatch):
        fetched = ["Mozilla/5.0 (X11; Linux x86_64) FreshAgent/1.0"]
        monkeypatch.setattr(web_fetch_module, "_fetch_user_agents", _async_ok(fetched))
        agents = await web_fetch_module._resolve_user_agents()
        assert agents == fetched
        path = web_fetch_module._UA_CACHE_PATH
        assert path.exists()
        assert json.loads(path.read_text()) == fetched

    async def test_stale_cache_kept_when_refresh_fails(self, monkeypatch):
        stale = ["Mozilla/5.0 (X11; Linux x86_64) StaleAgent/1.0"]
        path = web_fetch_module._UA_CACHE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stale))
        # Make the cache look older than the TTL.
        old_mtime = time.time() - web_fetch_module._UA_TTL_SECONDS - 10
        os.utime(path, (old_mtime, old_mtime))
        monkeypatch.setattr(web_fetch_module, "_fetch_user_agents", _async_fail)
        agents = await web_fetch_module._resolve_user_agents()
        # Stale cache is preferred over the hardcoded fallback.
        assert agents == stale
