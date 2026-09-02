#!/usr/bin/env python3
"""
DANDI Archive repository source implementation.

File: klea_utils/mcp/tool_impls/repositories/dandi.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from collections import deque
from typing import Any
from urllib.parse import urlparse

from klea_utils.mcp.tool_impls.session import SessionLike

from .errors import RepositorySourceError
from .sources import _get_json

logger = logging.getLogger(__name__)

#: DANDI Archive REST API base URL.
DANDI_API_BASE = "https://api.dandiarchive.org/api"
#: The default version label, used when none is specified.
DRAFT_VERSION = "draft"
#: Page size for the ``assets/paths`` endpoint.
PAGE_SIZE = 100
#: Cap on the number of pages fetched per folder, to bound runaway loops.
MAX_PAGES = 100
#: Cap on the total number of files collected, to bound runaway recursion.
MAX_FILES = 10_000
#: Hosts that serve DANDI Archive.
DANDI_HOSTS = ("dandiarchive.org", "www.dandiarchive.org")


def _parse_dandi_url(url: str) -> str:
    """Extract the dandiset ID from a DANDI Archive URL.

    DANDI URLs have the stable shape ``https://dandiarchive.org/dandiset/<id>``
    (optionally followed by ``/versions/<version>``); the ID is the first
    path segment after ``dandiset``.

    :raises RepositorySourceError: when the URL does not name a dandiset.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RepositorySourceError(f"{url} is not a valid DANDI URL")
    if (parsed.hostname or "").lower() not in DANDI_HOSTS:
        raise RepositorySourceError(f"{url} is not a DANDI URL")
    parts = [p for p in parsed.path.split("/") if p]
    try:
        index = parts.index("dandiset")
    except ValueError:
        raise RepositorySourceError(f"{url} does not name a DANDI dandiset")
    if index + 1 >= len(parts) or not parts[index + 1]:
        raise RepositorySourceError(f"{url} does not name a DANDI dandiset")
    return parts[index + 1]


def _assets_paths_url(dandiset_id: str, version: str) -> str:
    return f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/{version}/assets/paths/"


async def dandi_list_versions(session: SessionLike | None, url: str) -> dict[str, Any]:
    """List the available versions of a DANDI dandiset.

    The list includes the working ``draft`` version as well as published
    versions.

    Use when:
        - Discovering which versions a DANDI dandiset offers before listing its
          files.

    Args:
        url: DANDI dandiset URL (https://dandiarchive.org/dandiset/<id>).

    Returns:
        Dictionary with source, url, versions, and an empty files list.
    """
    versions: list[str] = []
    error = ""
    try:
        dandiset_id = _parse_dandi_url(url)
        contents = await _get_json(
            session, f"{DANDI_API_BASE}/dandisets/{dandiset_id}/versions/"
        )
        versions = [v["version"] for v in contents.get("results", [])]
        logger.info(f"Listed {len(versions)} versions for dandiset {dandiset_id}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list DANDI versions for {url}: {exc}")

    return {
        "source": "dandi",
        "url": url,
        "version": None,
        "versions": versions,
        "files": [],
        "error": error,
    }


async def dandi_list_files(
    session: SessionLike | None,
    url: str,
    version: str | None = None,
) -> dict[str, Any]:
    """List the files of a DANDI dandiset at a given version.

    The file tree is walked recursively via the ``assets/paths`` endpoint:
    entries with an ``asset`` are files, entries without one are folders that
    are descended into.  When ``version`` is omitted, the ``draft`` version
    is used.

    Use when:
    - Getting the file list of a DANDI dandiset so files can be downloaded.

    Args:
        url: DANDI dandiset URL (https://dandiarchive.org/dandiset/<id>).
        version: Version to list (e.g. ``draft`` or a published version).
            Defaults to ``draft``.

    Returns:
        Dictionary with source, url, version, files (path, name,
        download_url, size), and error.
    """
    files: list[dict[str, Any]] = []
    error = ""
    try:
        dandiset_id = _parse_dandi_url(url)
        if version is None:
            version = DRAFT_VERSION
            logger.debug(f"Using default version {version} for {dandiset_id}")

        paths_url = _assets_paths_url(dandiset_id, version)
        #: Pending folder path prefixes to walk (BFS).
        queue: deque[str] = deque([""])
        truncated = False

        while queue and not truncated:
            prefix = queue.popleft()
            page = 1
            while page <= MAX_PAGES and not truncated:
                contents = await _get_json(
                    session,
                    paths_url,
                    params={
                        "path_prefix": prefix,
                        "page": page,
                        "page_size": PAGE_SIZE,
                    },
                )
                results = contents.get("results", [])
                if not results:
                    break
                for item in results:
                    if len(files) >= MAX_FILES:
                        logger.warning(
                            f"Reached the DANDI file cap of {MAX_FILES}; "
                            "truncating the listing"
                        )
                        truncated = True
                        break
                    asset = item.get("asset")
                    path = item.get("path", "")
                    if asset:
                        asset_id = asset.get("asset_id", "")
                        if not asset_id or not path:
                            continue
                        files.append(
                            {
                                "path": path,
                                "name": path.rstrip("/").rsplit("/", 1)[-1],
                                "download_url": (
                                    f"{DANDI_API_BASE}/assets/{asset_id}/download/"
                                ),
                                "size": item.get("aggregate_size"),
                            }
                        )
                    elif path:
                        # A folder: descend into it.
                        queue.append(path)
                if len(results) < PAGE_SIZE:
                    break
                page += 1

        logger.info(f"Listed {len(files)} files for dandiset {dandiset_id}@{version}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list DANDI files for {url}: {exc}")

    return {
        "source": "dandi",
        "url": url,
        "version": version,
        "versions": None,
        "files": files,
        "error": error,
    }
