#!/usr/bin/env python3
"""
Client for the models.dev model catalog.

Fetches ``https://models.dev/api.json`` (a ~3MB JSON mapping of provider ->
model -> properties) and exposes the per-model token limits used to bound
LLM output token reservations.  This is needed because some providers
(e.g. HuggingFace) reserve the whole context window as output when no
max-token parameter is set, which leads to spurious usage limits and rate
limiting.

The catalog is fetched lazily on first use, kept in memory for the
process lifetime (``lru_cache``), and mirrored to an on-disk cache
(``{user_cache_dir}/klea/models-dev.json``) with a one-day TTL so that
restarts do not need to re-download it.

Providers without a catalog entry (local ollama, unknown custom
endpoints) and models missing from the catalog resolve to ``None`` so
callers can fall back gracefully instead of failing.

File: klea_utils/models_catalog.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import httpx
from platformdirs import PlatformDirs

logger = logging.getLogger(__name__)

#: Default models.dev catalog URL.  Overridable via the KLEA_MODELS_DEV_URL
#: environment variable (e.g. for offline mirrors or enterprise proxies).
DEFAULT_MODELS_DEV_URL = "https://models.dev/api.json"

#: Time-to-live for the on-disk catalog cache, in seconds (1 day).
DISK_CACHE_TTL_SECONDS = 24 * 60 * 60

#: File name of the on-disk catalog cache, under the platformdirs cache dir.
DISK_CACHE_FILE = "models-dev.json"

#: HTTP timeout for fetching the catalog.  Short: this is only hit on the
#: first use after the disk cache expires.
FETCH_TIMEOUT_SECONDS = 15.0

#: Klea provider id -> models.dev catalog provider key.  Providers mapped
#: to ``None`` have no catalog entry (local ollama) and are handled as
#: "no information".  ``custom`` resolves to ``openai`` to mirror
#: ``LLMModel.build_config()``, which maps ``custom:`` model strings to
#: ``model_provider="openai"``.  Unknown providers are passed through
#: as-is and simply miss if the catalog does not contain them.
_MODELS_DEV_PROVIDER_KEYS: dict[str, str | None] = {
    "huggingface": "huggingface",
    "openai": "openai",
    "anthropic": "anthropic",
    "google_genai": "google",
    "google": "google",
    "custom": "openai",
    "ollama": None,
}


class ModelLimits(NamedTuple):
    """Token limits for a single model from the catalog.

    All fields are optional: the catalog always carries ``context`` and
    ``output``, while ``input`` is only defined for a subset of models.
    """

    context: int | None = None
    input: int | None = None
    output: int | None = None


def _catalog_provider_key(provider: str) -> str | None:
    """Map a Klea provider id to its models.dev catalog key.

    :param provider: Klea provider id (``huggingface``, ``openai``, ...)
    :returns: The models.dev catalog key, or ``None`` if the provider is
        known to have no catalog entry.
    """
    key = provider.lower()
    if key in _MODELS_DEV_PROVIDER_KEYS:
        return _MODELS_DEV_PROVIDER_KEYS[key]
    # Pass through unknown providers verbatim; lookups for them will
    # simply miss and return None.
    return key


def _catalog_url() -> str:
    """Return the models.dev catalog URL (env override or default)."""
    return os.getenv("KLEA_MODELS_DEV_URL", DEFAULT_MODELS_DEV_URL)


def _disk_cache_path() -> Path:
    """Return the on-disk cache path for the models.dev catalog.

    Shared by all Klea packages (``~/.cache/klea/models-dev.json`` on
    Linux) so a single download serves rag, code, and mcp processes.
    """
    return Path(PlatformDirs("klea").user_cache_dir) / DISK_CACHE_FILE


def _load_catalog_from_disk(cache_path: Path, allow_stale: bool = False) -> dict | None:
    """Load the catalog from the on-disk cache.

    :param cache_path: Path to the cache file.
    :param allow_stale: When ``False`` (default), return ``None`` for
        missing or expired caches.  When ``True``, also return expired
        caches (used as an offline fallback).
    :returns: The catalog dict, or ``None`` if it could not be loaded.
    """
    try:
        if not cache_path.exists():
            return None
        age_seconds = time.time() - cache_path.stat().st_mtime
        if not allow_stale and age_seconds > DISK_CACHE_TTL_SECONDS:
            logger.debug(
                "models.dev catalog cache is stale (%ds old), refetching",
                int(age_seconds),
            )
            return None
        with open(cache_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("models.dev disk cache %s is not a JSON object", cache_path)
            return None
        logger.debug("Loaded models.dev catalog from disk cache: %s", cache_path)
        return data
    except Exception as e:
        logger.warning("Failed to read models.dev disk cache %s: %s", cache_path, e)
        return None


def _write_catalog_to_disk(cache_path: Path, data: dict) -> None:
    """Write the catalog to the on-disk cache (best-effort)."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        logger.debug("Wrote models.dev catalog to disk cache: %s", cache_path)
    except Exception as e:
        logger.warning("Failed to write models.dev disk cache %s: %s", cache_path, e)


def _fetch_catalog() -> dict:
    """Fetch the models.dev catalog from the network.

    Blocking with a short timeout: called only on first use after the
    disk cache expires.  Raises on network or parse failure.

    :returns: The catalog dict.
    """
    url = _catalog_url()
    logger.info("Fetching models.dev catalog from %s", url)
    resp = httpx.get(url, timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise TypeError("models.dev catalog is not a JSON object")
    return data


@lru_cache(maxsize=1)
def _catalog() -> dict:
    """Return the models.dev catalog, loading from cache or network.

    Cached in memory for the process lifetime (``lru_cache``), so the
    catalog is downloaded at most once per process.  A fresh disk cache
    short-circuits the network; on a network failure a stale disk copy
    is used as a fallback before propagating the error.

    :returns: The catalog dict.
    :raises: The underlying network/parse error if no disk copy exists.
    """
    cache_path = _disk_cache_path()

    data = _load_catalog_from_disk(cache_path)
    if data is not None:
        return data

    try:
        data = _fetch_catalog()
    except Exception as e:
        logger.warning("Failed to fetch models.dev catalog: %s", e)
        stale = _load_catalog_from_disk(cache_path, allow_stale=True)
        if stale is not None:
            logger.info("Falling back to stale models.dev catalog from disk")
            return stale
        raise

    _write_catalog_to_disk(cache_path, data)
    return data


def get_model_limits(provider: str, model_name: str) -> ModelLimits | None:
    """Return the token limits for a provider + model, or ``None``.

    Returns ``None`` when the provider has no catalog entry (local
    ollama, custom endpoints), the model is missing from the catalog, or
    the catalog could not be fetched.  Callers should treat ``None`` as
    "no information available".

    :param provider: Klea provider id (e.g. ``"huggingface"``).
    :param model_name: Model identifier, e.g. ``"gpt-4o"`` or
        ``"Qwen/Qwen3-Coder-30B-A3B-Instruct"``.
    :returns: ``ModelLimits`` with whatever fields the catalog defines.
    """
    catalog_key = _catalog_provider_key(provider)
    if catalog_key is None:
        return None

    try:
        catalog = _catalog()
    except Exception as e:
        logger.warning("models.dev catalog unavailable: %s", e)
        return None

    provider_entry = catalog.get(catalog_key)
    if not isinstance(provider_entry, dict):
        return None
    model_entry = provider_entry.get("models", {}).get(model_name)
    if not isinstance(model_entry, dict):
        return None
    limit = model_entry.get("limit")
    if not isinstance(limit, dict):
        return None

    return ModelLimits(
        context=limit.get("context") if isinstance(limit.get("context"), int) else None,
        input=limit.get("input") if isinstance(limit.get("input"), int) else None,
        output=limit.get("output") if isinstance(limit.get("output"), int) else None,
    )


def get_model_output_limit(provider: str, model_name: str) -> int | None:
    """Return the max output token limit for a provider + model.

    Convenience wrapper around :func:`get_model_limits`.

    :param provider: Klea provider id.
    :param model_name: Model identifier.
    :returns: The model's ``limit.output``, or ``None`` if unknown.
    """
    limits = get_model_limits(provider, model_name)
    return limits.output if limits else None


def get_model_context_limit(provider: str, model_name: str) -> int | None:
    """Return the context window size for a provider + model.

    Convenience wrapper around :func:`get_model_limits`.

    :param provider: Klea provider id.
    :param model_name: Model identifier.
    :returns: The model's ``limit.context``, or ``None`` if unknown.
    """
    limits = get_model_limits(provider, model_name)
    return limits.context if limits else None
