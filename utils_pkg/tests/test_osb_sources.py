#!/usr/bin/env python3
"""
Tests for the repository source implementations.

File: utils_pkg/tests/test_osb_sources.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import subprocess

import httpx
import klea_utils.api.utils as api_utils
import pytest
from klea_utils.mcp.tool_impls.repositories import github as github_module
from klea_utils.mcp.tool_impls.repositories.sources import _get_json

logger = logging.getLogger(__name__)

KLEA_REPO_URL = "https://github.com/NeuroML/neuroklea"
KLEA_REPO_BRANCH = "development"

#: Paths that are certain to be tracked and pushed in the klea repo, so the
#: live test only compares these instead of the full file list (which
#: disagrees while local changes are un-pushed).
STABLE_REPO_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "Readme.md",
    "CHANGELOG.md",
    "utils_pkg/AGENTS.md",
    "utils_pkg/setup.cfg",
    "agent_pkg/AGENTS.md",
    "agent_pkg/setup.cfg",
    "rag_pkg/AGENTS.md",
    "rag_pkg/setup.cfg",
    "mcp_pkg/AGENTS.md",
    "mcp_pkg/setup.cfg",
}


class _FakeResponse:
    """Minimal httpx-like response carrying a JSON payload."""

    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://example.com")
            response = httpx.Response(self.status_code, request=request, content=b"{}")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    def json(self):
        return self._payload


class _FakeSession:
    """Serves canned responses per URL and records the calls made.

    ``routes`` maps a URL (without query string) to a :class:`_FakeResponse`
    or to an exception instance that is raised on access.
    """

    def __init__(self, routes: dict):
        self._routes = routes
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        route = self._routes.get(url)
        if route is None:
            raise AssertionError(f"no route registered for {url}")
        if isinstance(route, Exception):
            raise route
        return route

    def stream(self, method, url, **kwargs):
        raise AssertionError("stream is not exercised in repository source tests")


@pytest.fixture(autouse=True)
def _fast_waits(monkeypatch):
    # Neutralise the exponential backoff so retry tests do not sleep.
    monkeypatch.setattr(api_utils, "wait_random_exponential", lambda **kw: 0.0)


@pytest.fixture(autouse=True)
def _hermetic_ssrf(monkeypatch):
    # Mock tests must not perform real DNS resolution for the SSRF guard;
    # the SSRF logic itself is covered by a dedicated test below.
    from klea_utils.mcp.tool_impls.repositories import sources

    monkeypatch.setattr(sources, "check_ssrf", lambda url: None)


@pytest.fixture(autouse=True)
def _no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


GITHUB_API = "https://api.github.com/repos/NeuroML/neuroklea"


def _github_routes(default_branch="development"):
    repo_info = {"default_branch": default_branch}
    branches = [
        {"name": "development", "commit": {"sha": "aaa"}},
        {"name": "main", "commit": {"sha": "bbb"}},
    ]
    tags = [
        {"name": "v1.0", "commit": {"sha": "ccc"}},
        # Duplicate of the development branch name: must appear once.
        {"name": "development", "commit": {"sha": "ddd"}},
    ]
    tree = {
        "tree": [
            {
                "path": "README.md",
                "mode": "100644",
                "type": "blob",
                "size": 12,
                "sha": "e1",
            },
            {"path": "src", "mode": "040000", "type": "tree", "sha": "e2"},
            {
                "path": "src/lib.py",
                "mode": "100644",
                "type": "blob",
                "size": 34,
                "sha": "e3",
            },
            {
                "path": "deployments/huggingface",
                "mode": "160000",
                "type": "commit",
                "size": None,
                "sha": "e4",
            },
        ]
    }
    return {
        GITHUB_API: _FakeResponse(repo_info),
        f"{GITHUB_API}/branches": _FakeResponse(branches),
        f"{GITHUB_API}/tags": _FakeResponse(tags),
        f"{GITHUB_API}/git/trees/development": _FakeResponse(tree),
        f"{GITHUB_API}/git/trees/main": _FakeResponse(
            {
                "tree": [
                    {
                        "path": "other.md",
                        "mode": "100644",
                        "type": "blob",
                        "size": 5,
                        "sha": "f1",
                    }
                ]
            }
        ),
    }


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/NeuroML/neuroklea", ("NeuroML", "neuroklea")),
        (
            "https://github.com/NeuroML/neuroklea/tree/development",
            ("NeuroML", "neuroklea"),
        ),
        ("http://www.github.com/owner/repo", ("owner", "repo")),
    ],
)
def test_parse_github_url_valid(url, expected):
    assert github_module._parse_github_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "https://gitlab.com/owner/repo",
        "https://github.com/single",
        "ftp://github.com/owner/repo",
    ],
)
def test_parse_github_url_invalid(url):
    from klea_utils.mcp.tool_impls.repositories.errors import RepositorySourceError

    with pytest.raises(RepositorySourceError):
        github_module._parse_github_url(url)


async def test_github_list_versions_merges_branches_and_tags():
    session = _FakeSession(_github_routes())
    result = await github_module.github_list_versions(session, KLEA_REPO_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["source"] == "github"
    # development appears as both a branch and a tag; listed once.
    assert result["versions"] == ["development", "main", "v1.0"]


async def test_github_list_files_with_version_lists_blobs_only():
    session = _FakeSession(_github_routes())
    result = await github_module.github_list_files(
        session, KLEA_REPO_URL, version="development"
    )
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["version"] == "development"
    paths = {f["path"] for f in result["files"]}
    assert paths == {"README.md", "src/lib.py"}
    by_path = {f["path"]: f for f in result["files"]}
    assert by_path["README.md"]["name"] == "README.md"
    assert by_path["README.md"]["size"] == 12
    assert (
        by_path["README.md"]["download_url"]
        == "https://raw.githubusercontent.com/NeuroML/neuroklea/development/README.md"
    )
    assert (
        by_path["src/lib.py"]["download_url"]
        == "https://raw.githubusercontent.com/NeuroML/neuroklea/development/src/lib.py"
    )


async def test_github_list_files_uses_default_branch():
    session = _FakeSession(_github_routes(default_branch="main"))
    result = await github_module.github_list_files(session, KLEA_REPO_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["version"] == "main"
    assert {f["path"] for f in result["files"]} == {"other.md"}


async def test_github_list_files_token_header(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    session = _FakeSession(_github_routes())
    await github_module.github_list_files(session, KLEA_REPO_URL, version="development")
    _, kwargs = session.calls[0]
    assert "Authorization" in kwargs["headers"]
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"


async def test_github_list_files_no_token():
    session = _FakeSession(_github_routes())
    await github_module.github_list_files(session, KLEA_REPO_URL, version="development")
    _, kwargs = session.calls[0]
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["headers"]["User-Agent"].startswith("klea-web-fetch/")


async def test_github_list_files_http_error():
    routes = {
        f"{GITHUB_API}/git/trees/development": _FakeResponse({}, status=404),
    }
    session = _FakeSession(routes)
    result = await github_module.github_list_files(
        session, KLEA_REPO_URL, version="development"
    )
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert "HTTP 404" in result["error"]


async def test_github_list_files_ssrf_denied(monkeypatch):
    from klea_utils.mcp.tool_impls.repositories import sources

    monkeypatch.setattr(
        sources,
        "check_ssrf",
        lambda url: "Blocked request to private/internal address: 10.0.0.1",
    )
    session = _FakeSession(_github_routes())
    result = await github_module.github_list_files(
        session, KLEA_REPO_URL, version="development"
    )
    assert result["files"] == []
    assert "Blocked request" in result["error"]
    assert session.calls == []


async def test_get_json_transient_error_retries_then_fails():
    class _Flaky:
        def __init__(self):
            self.attempts = 0

        async def get(self, url, **kwargs):
            self.attempts += 1
            raise httpx.ConnectError("offline")

        def stream(self, method, url, **kwargs):
            raise AssertionError("stream not used")

    flaky = _Flaky()
    with pytest.raises(Exception) as excinfo:
        await _get_json(flaky, "https://api.github.com/x", retries=3)
    assert flaky.attempts == 3
    assert "Request to https://api.github.com/x failed" in str(excinfo.value)


def _local_tracked_files() -> set[str]:
    """Return the git-tracked file paths in the repository root.

    Excludes gitlink (submodule) entries (mode 160000); GitHub's
    git/trees API reports submodules as type "commit", which the
    implementation filters out.
    """
    root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
    root = root.decode("utf-8").strip()
    staged = subprocess.check_output(["git", "-C", root, "ls-files", "--stage"])
    files = set()
    for line in staged.decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        mode = parts[0].split()[0]
        if mode == "160000":
            continue
        files.add(parts[1])
    return files


async def test_github_live_klea_repo_stable_files_match():
    """The stable files of the klea repo must match between GitHub and local.

    Fetches the file list of the ``development`` branch and compares the
    curated :data:`STABLE_REPO_FILES` against ``git ls-files`` in the
    repository root.  The comparison is limited to those well-known files so
    the test does not depend on every local change being pushed.
    """
    local_files = _local_tracked_files()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await github_module.github_list_files(
                client, KLEA_REPO_URL, version=KLEA_REPO_BRANCH
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"GitHub API unavailable: {exc}")

    if result["error"]:
        pytest.skip(f"GitHub API error: {result['error']}")

    remote_files = {f["path"] for f in result["files"]}
    local_stable = local_files & STABLE_REPO_FILES
    remote_stable = remote_files & STABLE_REPO_FILES
    logger.debug(f"{len(local_stable) = }\n{len(remote_stable) = }")
    missing_remote = sorted(local_stable - remote_stable)
    missing_local = sorted(remote_stable - local_stable)
    logger.debug(f"{missing_remote = }\n{missing_local = }")
    assert local_stable == remote_stable
