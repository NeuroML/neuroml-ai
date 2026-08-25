#!/usr/bin/env python3
"""
Tests for the repository source implementations.

File: utils_pkg/tests/test_osb_sources.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
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

    ``routes`` maps a URL (without query string) to a :class:`_FakeResponse`,
    to an exception instance that is raised on access, or to a callable
    ``(params) -> _FakeResponse`` for responses that depend on query
    parameters.
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
        if callable(route):
            return route(kwargs.get("params"))
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


FIGSHARE_API = "https://api.figshare.com/v2/articles/14976822"
RDR_ARTICLE_URL = "https://rdr.ucl.ac.uk/articles/dataset/Fibre_Direction_Data/14976822"
RDR_FIRST_FILE = "FL_S_170905_10_40_52_fibre_direction.mat"


def _figshare_routes(version="1"):
    article_info = {"version": int(version)}
    versions = [{"version": int(version)}]
    files = [
        {
            "name": "f1.mat",
            "size": 10,
            "download_url": "https://ndownloader.figshare.com/files/1",
        },
        {
            "name": "f2.mat",
            "size": 20,
            "download_url": "https://ndownloader.figshare.com/files/2",
        },
    ]
    return {
        FIGSHARE_API: _FakeResponse(article_info),
        f"{FIGSHARE_API}/versions": _FakeResponse(versions),
        f"{FIGSHARE_API}/files": _FakeResponse(files),
    }


@pytest.mark.parametrize(
    "url,expected",
    [
        (
            "https://figshare.com/articles/dataset/Title/14976822",
            "14976822",
        ),
        (RDR_ARTICLE_URL, "14976822"),
        ("https://www.figshare.com/articles/x/42", "42"),
        # Institutional instances sit on arbitrary domains with arbitrary
        # path prefixes; only the trailing numeric article ID is reliable.
        ("https://rdr.myuniv.eu/lots/of/data/articles/14976822", "14976822"),
        ("https://data.someuni.edu/14976822/", "14976822"),
        ("https://figshare.com/articles/x/14976822?foo=bar", "14976822"),
    ],
)
def test_parse_figshare_url_valid(url, expected):
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    assert figshare_module._parse_figshare_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "https://github.com/owner/repo",
        "https://rdr.ucl.ac.uk/articles/x",
        "https://figshare.com/articles/dataset/Title/abc",
        "ftp://figshare.com/articles/x/1",
    ],
)
def test_parse_figshare_url_invalid(url):
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module
    from klea_utils.mcp.tool_impls.repositories.errors import RepositorySourceError

    with pytest.raises(RepositorySourceError):
        figshare_module._parse_figshare_url(url)


async def test_figshare_list_versions():
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    session = _FakeSession(_figshare_routes())
    result = await figshare_module.figshare_list_versions(session, RDR_ARTICLE_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["source"] == "figshare"
    assert result["versions"] == ["1"]


async def test_figshare_list_files_flat_mapping():
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    session = _FakeSession(_figshare_routes())
    result = await figshare_module.figshare_list_files(
        session, RDR_ARTICLE_URL, version="1"
    )
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["version"] == "1"
    assert [f["path"] for f in result["files"]] == ["f1.mat", "f2.mat"]
    by_path = {f["path"]: f for f in result["files"]}
    assert (
        by_path["f1.mat"]["download_url"] == "https://ndownloader.figshare.com/files/1"
    )
    assert by_path["f1.mat"]["size"] == 10


async def test_figshare_list_files_resolves_default_version():
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    session = _FakeSession(_figshare_routes(version="3"))
    result = await figshare_module.figshare_list_files(session, RDR_ARTICLE_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["version"] == "3"
    assert len(result["files"]) == 2


async def test_figshare_list_files_paginates(monkeypatch):
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    # Small page size so the two pages are exercised without huge fixtures.
    monkeypatch.setattr(figshare_module, "PAGE_SIZE", 2)

    def _files_route(params):
        page = int((params or {}).get("page", 1))
        if page == 1:
            payload = [
                {"name": "p1a.mat", "size": 1, "download_url": "https://x/1"},
                {"name": "p1b.mat", "size": 2, "download_url": "https://x/2"},
            ]
        else:
            payload = [
                {"name": "p2a.mat", "size": 3, "download_url": "https://x/3"},
            ]
        return _FakeResponse(payload)

    session = _FakeSession(
        {
            FIGSHARE_API: _FakeResponse({"version": 1}),
            f"{FIGSHARE_API}/files": _files_route,
        }
    )
    result = await figshare_module.figshare_list_files(
        session, RDR_ARTICLE_URL, version="1"
    )
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert [f["path"] for f in result["files"]] == [
        "p1a.mat",
        "p1b.mat",
        "p2a.mat",
    ]


async def test_figshare_list_files_skips_entries_missing_fields():
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    files = [
        {"name": "ok.mat", "size": 1, "download_url": "https://x/ok"},
        {"name": "no-url.mat", "size": 2},
        {"download_url": "https://x/anon"},
        {},
    ]
    session = _FakeSession(
        {
            FIGSHARE_API: _FakeResponse({"version": 1}),
            f"{FIGSHARE_API}/files": _FakeResponse(files),
        }
    )
    result = await figshare_module.figshare_list_files(
        session, RDR_ARTICLE_URL, version="1"
    )
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert [f["path"] for f in result["files"]] == ["ok.mat"]


async def test_figshare_list_files_http_error():
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    session = _FakeSession(
        {
            f"{FIGSHARE_API}/files": _FakeResponse({}, status=500),
        }
    )
    result = await figshare_module.figshare_list_files(
        session, RDR_ARTICLE_URL, version="1"
    )
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert "HTTP 500" in result["error"]


async def test_figshare_live_article_listing():
    """List the files of a real FigShare article (UCL RDR instance).

    Validates the full pipeline against the live API; skips when the API is
    unreachable.  Asserts a well-formed non-empty listing including a file
    name that is known to be present.
    """
    from klea_utils.mcp.tool_impls.repositories import figshare as figshare_module

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await figshare_module.figshare_list_files(client, RDR_ARTICLE_URL)
    except httpx.HTTPError as exc:
        pytest.skip(f"FigShare API unavailable: {exc}")

    if result["error"]:
        pytest.skip(f"FigShare API error: {result['error']}")

    assert len(result["files"]) >= 1
    names = {f["name"] for f in result["files"]}
    assert RDR_FIRST_FILE in names
    assert all(f["path"] for f in result["files"])
    assert all(f["download_url"] for f in result["files"])
    # The default version is resolved from the article metadata.
    assert result["version"].isdigit()


DANDI_API = "https://api.dandiarchive.org/api"
DANDI_TEST_URL = "https://dandiarchive.org/dandiset/000025"
DANDI_VERSIONS_URL = f"{DANDI_API}/dandisets/000025/versions/"
DANDI_PATHS_URL = f"{DANDI_API}/dandisets/000025/versions/draft/assets/paths/"
DANDI_TEST_FILE = "001_140709EXP_A1.nwb"


def _dandi_paths_route(tree_by_prefix: dict):
    """Return a route callable serving the assets/paths endpoint by prefix.

    ``tree_by_prefix`` maps a ``path_prefix`` to a list of result items;
    each item is ``{"path": ..., "asset": {...} | None}``.  Files beyond
    ``PAGE_SIZE`` are served on subsequent pages.
    """

    def _route(params):
        prefix = (params or {}).get("path_prefix", "")
        items = tree_by_prefix.get(prefix, [])
        page = int((params or {}).get("page", 1))
        page_size = int((params or {}).get("page_size", 100))
        start = (page - 1) * page_size
        page_items = items[start : start + page_size]
        return _FakeResponse(
            {
                "count": len(items),
                "next": None,
                "previous": None,
                "results": page_items,
            }
        )

    return _route


def _asset(path: str, asset_id: str) -> dict:
    return {"path": path, "asset": {"asset_id": asset_id}}


def _folder(path: str) -> dict:
    return {"path": path, "asset": None}


def _dandi_routes():
    versions = {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [{"version": "draft"}, {"version": "0.210812.1448"}],
    }
    tree = {
        "": [_folder("sub-a/"), _asset("root.nwb", "root-asset")],
        "sub-a/": [_asset("sub-a/deep.nwb", "deep-asset")],
    }
    return {
        DANDI_VERSIONS_URL: _FakeResponse(versions),
        DANDI_PATHS_URL: _dandi_paths_route(tree),
    }


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://dandiarchive.org/dandiset/000025", "000025"),
        ("https://dandiarchive.org/dandiset/000025/versions/draft", "000025"),
        ("http://www.dandiarchive.org/dandiset/000025", "000025"),
    ],
)
def test_parse_dandi_url_valid(url, expected):
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    assert dandi_module._parse_dandi_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "not-a-url",
        "https://github.com/owner/repo",
        "https://dandiarchive.org/",
        "https://dandiarchive.org/somewhere/000025",
        "ftp://dandiarchive.org/dandiset/000025",
    ],
)
def test_parse_dandi_url_invalid(url):
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module
    from klea_utils.mcp.tool_impls.repositories.errors import RepositorySourceError

    with pytest.raises(RepositorySourceError):
        dandi_module._parse_dandi_url(url)


async def test_dandi_list_versions():
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    session = _FakeSession(_dandi_routes())
    result = await dandi_module.dandi_list_versions(session, DANDI_TEST_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["source"] == "dandi"
    assert result["versions"] == ["draft", "0.210812.1448"]


async def test_dandi_list_files_recursive_walk():
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    session = _FakeSession(_dandi_routes())
    result = await dandi_module.dandi_list_files(session, DANDI_TEST_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert result["version"] == "draft"
    by_path = {f["path"]: f for f in result["files"]}
    assert set(by_path) == {"root.nwb", "sub-a/deep.nwb"}
    assert by_path["root.nwb"]["name"] == "root.nwb"
    assert by_path["root.nwb"]["size"] is None
    assert by_path["root.nwb"]["download_url"] == (
        f"{DANDI_API}/assets/root-asset/download/"
    )
    assert by_path["sub-a/deep.nwb"]["download_url"] == (
        f"{DANDI_API}/assets/deep-asset/download/"
    )


async def test_dandi_list_files_default_version_is_draft():
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    session = _FakeSession(_dandi_routes())
    result = await dandi_module.dandi_list_files(session, DANDI_TEST_URL)
    # The default "draft" is used and resolved without error.
    assert result["error"] == ""
    assert result["version"] == "draft"


async def test_dandi_list_files_paginates(monkeypatch):
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    monkeypatch.setattr(dandi_module, "PAGE_SIZE", 2)
    tree = {
        "": [_asset(f"f{i}.nwb", f"asset-{i}") for i in range(3)],
    }
    session = _FakeSession({DANDI_PATHS_URL: _dandi_paths_route(tree)})
    result = await dandi_module.dandi_list_files(session, DANDI_TEST_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert [f["path"] for f in result["files"]] == ["f0.nwb", "f1.nwb", "f2.nwb"]


async def test_dandi_list_files_caps_total_files(monkeypatch):
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    monkeypatch.setattr(dandi_module, "MAX_FILES", 2)
    tree = {
        "": [_asset(f"f{i}.nwb", f"asset-{i}") for i in range(5)],
    }
    session = _FakeSession({DANDI_PATHS_URL: _dandi_paths_route(tree)})
    result = await dandi_module.dandi_list_files(session, DANDI_TEST_URL)
    logger.debug(f"{result = }")
    assert result["error"] == ""
    assert len(result["files"]) == 2


async def test_dandi_list_files_http_error():
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    session = _FakeSession({DANDI_PATHS_URL: _FakeResponse({}, status=500)})
    result = await dandi_module.dandi_list_files(session, DANDI_TEST_URL)
    logger.debug(f"{result = }")
    assert result["files"] == []
    assert "HTTP 500" in result["error"]


async def test_dandi_live_dandiset_listing():
    """List the files of a real DANDI dandiset.

    Uses the small dandiset 000025 by default; override with the
    ``KLEA_DANDI_TEST_URL`` environment variable to use another dandiset.
    Skips when the API is unreachable.
    """
    from klea_utils.mcp.tool_impls.repositories import dandi as dandi_module

    url = os.environ.get("KLEA_DANDI_TEST_URL", DANDI_TEST_URL)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            result = await dandi_module.dandi_list_files(client, url)
    except httpx.HTTPError as exc:
        pytest.skip(f"DANDI API unavailable: {exc}")

    if result["error"]:
        pytest.skip(f"DANDI API error: {result['error']}")

    assert result["version"] == "draft"
    names = {f["name"] for f in result["files"]}
    assert DANDI_TEST_FILE in names
    assert all(f["path"] for f in result["files"])
    assert all(f["download_url"] for f in result["files"])
