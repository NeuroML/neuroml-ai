#!/usr/bin/env python3
"""
BioModels repository source implementation.

File: klea_utils/mcp/tool_impls/repositories/biomodels.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import re
from typing import Any
from urllib.parse import urlparse

from klea_utils.mcp.tool_impls.session import SessionLike

from .errors import RepositorySourceError
from .sources import _get_json

logger = logging.getLogger(__name__)

#: Canonical BioModels API base.  The legacy ``www.ebi.ac.uk/biomodels``
#: host redirects here, so all API calls go to this base regardless of the
#: host used in the model URL.
BIOMODELS_API_BASE = "https://www.biomodels.org"
#: Hosts that serve BioModels model pages.
BIOMODELS_HOSTS = ("biomodels.org", "www.biomodels.org")
LEGACY_BIOMODELS_HOSTS = ("ebi.ac.uk", "www.ebi.ac.uk")


def _parse_biomodels_url(url: str) -> str:
    """Extract the model ID from a BioModels model URL.

    Accepts the current ``biomodels.org`` host and the legacy
    ``www.ebi.ac.uk/biomodels`` host (which redirects to it).  The model ID
    is the last path segment starting with ``MODEL`` (submitted models) or
    ``BIOMD`` (curated models).

    :raises RepositorySourceError: when the URL does not name a BioModels
        model.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RepositorySourceError(f"{url} is not a valid BioModels URL")
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host in BIOMODELS_HOSTS:
        parts = [p for p in path.split("/") if p]
    elif host in LEGACY_BIOMODELS_HOSTS:
        if not path.startswith("/biomodels"):
            raise RepositorySourceError(f"{url} is not a BioModels URL")
        parts = [p for p in path.split("/") if p]
        if parts and parts[0].lower() == "biomodels":
            parts = parts[1:]
    else:
        raise RepositorySourceError(f"{url} is not a BioModels URL")

    model_id = ""
    for part in reversed(parts):
        if re.fullmatch(r"(MODEL|BIOMD)\w+", part):
            model_id = part
            break
    if not model_id:
        raise RepositorySourceError(f"{url} does not name a BioModels model")
    return model_id


async def _model_info(session: SessionLike | None, model_id: str) -> dict[str, Any]:
    """Fetch the JSON model record for *model_id*."""
    return await _get_json(
        session,
        f"{BIOMODELS_API_BASE}/{model_id}",
        params={"format": "json"},
    )


async def biomodels_list_versions(
    session: SessionLike | None, url: str
) -> dict[str, Any]:
    """List the available revisions (versions) of a BioModels model.

    Use when:
    - Discovering which revisions a BioModels model offers before listing
      its files.

    Args:
        url: BioModels model URL (e.g.
            https://www.biomodels.org/MODEL0912160000).

    Returns:
        Dictionary with source, url, versions, and an empty files list.
    """
    versions: list[str] = []
    error = ""
    try:
        model_id = _parse_biomodels_url(url)
        info = await _model_info(session, model_id)
        revisions = info.get("history", {}).get("revisions", [])
        versions = [str(r["version"]) for r in revisions]
        logger.info(f"Listed {len(versions)} versions for model {model_id}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list BioModels versions for {url}: {exc}")

    return {
        "source": "biomodels",
        "url": url,
        "version": None,
        "versions": versions,
        "files": [],
        "error": error,
    }


async def biomodels_list_files(
    session: SessionLike | None,
    url: str,
    version: str | None = None,
) -> dict[str, Any]:
    """List the files of a BioModels model at a given revision.

    The file list combines the ``main`` and ``additional`` file groups of
    the model record.  When ``version`` is omitted, the latest revision is
    used.

    Use when:
    - Getting the file list of a BioModels model so files can be downloaded.

    Args:
        url: BioModels model URL (e.g.
            https://www.biomodels.org/MODEL0912160000).
        version: Revision number (as a string) to list.  Defaults to the
            latest revision.

    Returns:
        Dictionary with source, url, version, files (path, name,
        download_url, size), and error.
    """
    files: list[dict[str, Any]] = []
    error = ""
    try:
        model_id = _parse_biomodels_url(url)
        if version is None:
            info = await _model_info(session, model_id)
            revisions = info.get("history", {}).get("revisions", [])
            if not revisions:
                raise RepositorySourceError(
                    f"No revisions found for BioModels model {model_id}"
                )
            version = str(max(r["version"] for r in revisions))
            logger.debug(f"Using latest revision {version} for {model_id}")

        contents = await _get_json(
            session,
            f"{BIOMODELS_API_BASE}/model/files/{model_id}.{version}",
            params={"format": "json"},
        )
        for afile in contents.get("additional", []) + contents.get("main", []):
            name = afile.get("name", "")
            if not name:
                continue
            size = afile.get("fileSize")
            if size is not None:
                try:
                    size = int(size)
                except (TypeError, ValueError):
                    pass
            files.append(
                {
                    # BioModels has no folder structure; the flat name is
                    # the relative path.
                    "path": name,
                    "name": name,
                    "download_url": (
                        f"{BIOMODELS_API_BASE}/model/download/{model_id}.{version}"
                        f"?filename={name}"
                    ),
                    "size": size,
                }
            )
        logger.info(f"Listed {len(files)} files for model {model_id}@{version}")
    except RepositorySourceError as exc:
        error = str(exc)
        logger.warning(f"Failed to list BioModels files for {url}: {exc}")

    return {
        "source": "biomodels",
        "url": url,
        "version": version,
        "versions": None,
        "files": files,
        "error": error,
    }
