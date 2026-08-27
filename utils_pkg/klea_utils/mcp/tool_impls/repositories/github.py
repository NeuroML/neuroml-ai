#!/usr/bin/env python3
"""
GitHub repository source implementation.

File: klea_utils/mcp/tool_impls/repositories/github.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
from typing import Any
from urllib.parse import urlparse

from klea_utils.mcp.tool_impls.session import SessionLike

from .errors import RepositorySourceError
from .sources import _get_json

logger = logging.getLogger(__name__)

#: GitHub REST API base for the ``/repos`` collection.
GITHUB_API_BASE = "https://api.github.com/repos"
#: Base URL for direct raw file downloads.
RAW_BASE = "https://raw.githubusercontent.com"
#: Page size for the branches/tags listing endpoints.
PAGE_SIZE = 100


def _parse_github_url(url: str) -> tuple[str, str]:
    """Extract ``(owner, repo)`` from a GitHub repository URL.

    Extra path segments (e.g. ``tree/development``) are ignored; only the
    ``owner/repo`` part is used.

    :raises RepositorySourceError: when the URL does not name a GitHub
        repository.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RepositorySourceError(f"{url} is not a valid GitHub URL")
    if (parsed.hostname or "").lower() not in ("github.com", "www.github.com"):
        raise RepositorySourceError(f"{url} is not a GitHub URL")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise RepositorySourceError(
            f"{url} does not name a GitHub repository (owner/repo)"
        )
    return parts[0], parts[1]


def _github_headers() -> dict[str, str]:
    """Headers for GitHub API requests, honouring ``GITHUB_TOKEN``.

    The token is optional and only used to lift the unauthenticated API
    rate limit; no secret is required for public repositories.
    """
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_url(owner: str, repo: str) -> str:
    return f"{GITHUB_API_BASE}/{owner}/{repo}"


async def _default_branch(session: SessionLike | None, owner: str, repo: str) -> str:
    """Resolve the repository's default branch name."""
    info = await _get_json(session, _api_url(owner, repo), headers=_github_headers())
    default = info.get("default_branch")
    if not default:
        raise RepositorySourceError(f"No default branch for {owner}/{repo}")
    return default


async def github_list_versions(session: SessionLike | None, url: str) -> dict[str, Any]:
    """List the available versions (branches and tags) of a GitHub repository.

    A GitHub version is a git branch or a tag; both are merged into a
    single list.  When a name exists as both a branch and a tag, it is
    listed once.

    Use when:
        - Discovering which branches/tags a GitHub repository offers before
          listing its files.

    Args:
        url: GitHub repository URL (https://github.com/<owner>/<repo>).

    Returns:
        Dictionary with source, url, versions, and an empty files list.
    """
    versions: list[str] = []
    error = ""
    try:
        owner, repo = _parse_github_url(url)
        api = _api_url(owner, repo)
        headers = _github_headers()
        branches = await _get_json(
            session, f"{api}/branches", params={"per_page": PAGE_SIZE}, headers=headers
        )
        tags = await _get_json(
            session, f"{api}/tags", params={"per_page": PAGE_SIZE}, headers=headers
        )
        for item in branches + tags:
            name = item.get("name")
            if name and name not in versions:
                versions.append(name)
        logger.info(f"Listed {len(versions)} versions for {owner}/{repo}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list GitHub versions for {url}: {exc}")

    return {
        "source": "github",
        "url": url,
        "version": None,
        "versions": versions,
        "files": [],
        "error": error,
    }


async def github_list_files(
    session: SessionLike | None,
    url: str,
    version: str | None = None,
) -> dict[str, Any]:
    """List the files in a GitHub repository at a given version.

    The ``version`` is a git branch or a tag.  When a name exists as both
    a branch and a tag, a branch is assumed (git ref resolution
    precedence).  If ``version`` is omitted, the repository's default
    branch is used.

    Use when:
        - Getting the file list of a GitHub repository so files can be
          downloaded.

    Args:
        url: GitHub repository URL (https://github.com/<owner>/<repo>).
        version: Branch or tag to list.  Defaults to the default branch.

    Returns:
        Dictionary with source, url, version, files (path, name,
        download_url, size), and error.
    """
    files: list[dict[str, Any]] = []
    error = ""
    try:
        owner, repo = _parse_github_url(url)
        api = _api_url(owner, repo)
        headers = _github_headers()
        if version is None:
            version = await _default_branch(session, owner, repo)
            logger.debug(f"Using default branch {version} for {owner}/{repo}")

        tree = await _get_json(
            session,
            f"{api}/git/trees/{version}",
            params={"recursive": "1"},
            headers=headers,
        )
        for item in tree.get("tree", []):
            if item.get("type") != "blob":
                # Folders (type "tree") and submodules (type "commit") are
                # not downloadable files.
                continue
            path = item.get("path", "")
            if not path:
                continue
            files.append(
                {
                    "path": path,
                    "name": path.rsplit("/", 1)[-1],
                    "download_url": f"{RAW_BASE}/{owner}/{repo}/{version}/{path}",
                    "size": item.get("size"),
                }
            )
        logger.info(f"Listed {len(files)} files for {owner}/{repo}@{version}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list GitHub files for {url}: {exc}")

    return {
        "source": "github",
        "url": url,
        "version": version,
        "versions": None,
        "files": files,
        "error": error,
    }
