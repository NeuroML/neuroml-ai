#!/usr/bin/env python3
"""
FigShare repository source implementation.

File: klea_utils/mcp/tool_impls/repositories/figshare.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any
from urllib.parse import urlparse

from klea_utils.mcp.tool_impls.session import SessionLike

from .errors import RepositorySourceError
from .sources import _get_json

logger = logging.getLogger(__name__)

#: FigShare API v2 base URL.  Even for institutional FigShare instances
#: (e.g. ``rdr.ucl.ac.uk``) the article IDs and the API endpoint are shared.
FIGSHARE_API_BASE = "https://api.figshare.com/v2"
#: Page size for the article files endpoint (the API maximum is 1000).
PAGE_SIZE = 1000
#: Cap on the number of pages fetched, to bound runaway loops.
MAX_PAGES = 100


def _parse_figshare_url(url: str) -> str:
    """Extract the article ID from a FigShare article URL.

    FigShare instances run on many institutional domains and their public
    URLs have no standard shape beyond ending in the numeric article ID, so
    any HTTP(S) URL is accepted and the trailing path segment must be
    numeric.  The API endpoint is the shared ``api.figshare.com`` regardless
    of the instance host, so only the article ID matters.

    :raises RepositorySourceError: when the URL does not end in a numeric
        article ID.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RepositorySourceError(f"{url} is not a valid FigShare URL")
    parts = [p for p in parsed.path.split("/") if p]
    if not parts or not parts[-1].isdigit():
        raise RepositorySourceError(f"{url} does not name a FigShare article")
    return parts[-1]


def _api_url(article_id: str) -> str:
    return f"{FIGSHARE_API_BASE}/articles/{article_id}"


async def figshare_list_versions(
    session: SessionLike | None, url: str
) -> dict[str, Any]:
    """List the available versions of a FigShare article.

    Use when:
        - Discovering which versions a FigShare article offers before listing
          its files.

    Args:
        url: FigShare article URL (e.g. https://figshare.com/articles/dataset/<title>/<article_id>).

    Returns:
        Dictionary with source, url, versions, and an empty files list.
    """
    versions: list[str] = []
    error = ""
    try:
        article_id = _parse_figshare_url(url)
        contents = await _get_json(session, f"{_api_url(article_id)}/versions")
        versions = [str(v["version"]) for v in contents]
        logger.info(f"Listed {len(versions)} versions for article {article_id}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list FigShare versions for {url}: {exc}")

    return {
        "source": "figshare",
        "url": url,
        "version": None,
        "versions": versions,
        "files": [],
        "error": error,
    }


async def figshare_list_files(
    session: SessionLike | None,
    url: str,
    version: str | None = None,
) -> dict[str, Any]:
    """List the files of a FigShare article.

    FigShare serves the same file list for every version of an article (the
    files endpoint is not versioned); the ``version`` argument is accepted
    for a uniform API but only labels the result.  When ``version`` is
    omitted, the article's current version is reported.

    Use when:
        - Getting the file list of a FigShare article so files can be
          downloaded.

    Args:
        url: FigShare article URL (e.g. https://figshare.com/articles/dataset/<title>/<article_id>).
        version: Version label for the result.  Defaults to the article's
            current version.

    Returns:
        Dictionary with source, url, version, files (path, name,
        download_url, size), and error.
    """
    files: list[dict[str, Any]] = []
    error = ""
    try:
        article_id = _parse_figshare_url(url)
        if version is None:
            info = await _get_json(session, _api_url(article_id))
            version = str(info.get("version"))
            logger.debug(f"Using article version {version} for {article_id}")

        page = 1
        while page <= MAX_PAGES:
            contents = await _get_json(
                session,
                f"{_api_url(article_id)}/files",
                params={"page": page, "page_size": PAGE_SIZE},
            )
            if not contents:
                break
            for afile in contents:
                name = afile.get("name", "")
                download_url = afile.get("download_url", "")
                if not name or not download_url:
                    continue
                files.append(
                    {
                        # FigShare has no folder structure; the flat name is
                        # the relative path.
                        "path": name,
                        "name": name,
                        "download_url": download_url,
                        "size": afile.get("size"),
                    }
                )
            if len(contents) < PAGE_SIZE:
                break
            page += 1
        logger.info(f"Listed {len(files)} files for article {article_id}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list FigShare files for {url}: {exc}")

    return {
        "source": "figshare",
        "url": url,
        "version": version,
        "versions": None,
        "files": files,
        "error": error,
    }
