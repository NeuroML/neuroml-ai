#!/usr/bin/env python3
"""
Shared SSE streaming client for Klea frontends.

Provides both an async generator (for NiceGUI and TUI) and a synchronous
generator (for Streamlit) that consume the ``/query/stream`` SSE endpoint
and yield parsed event dicts.

File: klea_utils/api/sse.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
from collections.abc import AsyncGenerator, Generator

import httpx

from ..llm import parse_model_name
from ..plogging import setup_logger

logger = setup_logger(__name__)


async def stream_events(
    query: str,
    session_id: str,
    server_url: str,
) -> AsyncGenerator[dict, None]:
    """POST to ``/query/stream`` and yield parsed SSE event dicts.

    Each yielded dict has at least a ``"type"`` key.  Known types::

        progress    {"type": "progress", "node": "<label>"}
        info        {"type": "info", "node": "<label>", "data": {...}}
        debug       {"type": "debug", "node": "<label>", "data": {...}}
        token       {"type": "token", "content": "<chunk>", "node": "<label>"}
        complete    {"type": "complete", "message_for_user": "<text>"}
        error       {"type": "error", "message": "<text>"}

    This async generator is intended for NiceGUI and TUI frontends.

    :param query: User's query string.
    :param session_id: Opaque session identifier.
    :param server_url: Base URL of the backend API server.
    """
    url = f"{server_url}/query/stream"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            url,
            json={"query": query, "session_id": session_id},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    if line.strip():
                        logger.warning("Skipping non-data line: %s", line[:80])
                    continue
                # Strip the SSE "data: " prefix (6 chars) to get raw JSON.
                yield json.loads(line[6:])


# TODO: if more fetch-json-from-endpoint functions are added, consider
# extracting the async/sync boilerplate into _fetch_json / _fetch_json_sync
# helpers in utils.py to avoid repetition.
async def fetch_active_models(
    server_url: str,
    session_id: str,
) -> dict[str, dict[str, str]]:
    """Fetch the resolved model config per role for a session.

    Calls ``GET /models/session/{session_id}/active`` and returns the
    merged default + override config dict.

    :param server_url: Base URL of the backend API server.
    :param session_id: Opaque session identifier.
    :returns: ``{"chat": {"model": "...", "provider": "..."}, "guard": ..., "embedding": ...}``
    """
    url = f"{server_url}/models/session/{session_id}/active"
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                logger.warning(
                    "Failed to fetch active models: HTTP %s from %s",
                    resp.status_code,
                    url,
                )
                return {}
            data: dict[str, dict[str, str]] = resp.json()
            logger.debug("Active models for session %s: %s", session_id, data)
            return data
        except Exception as e:
            logger.warning(
                "Failed to fetch active models from %s: %s",
                url,
                e,
            )
            return {}


def fetch_active_models_sync(
    server_url: str,
    session_id: str,
) -> dict[str, dict[str, str]]:
    """Synchronous counterpart of :func:`fetch_active_models`.

    Intended for Streamlit and TUI frontends.

    :param server_url: Base URL of the backend API server.
    :param session_id: Opaque session identifier.
    """
    url = f"{server_url}/models/session/{session_id}/active"
    with httpx.Client(timeout=5) as client:
        try:
            resp = client.get(url)
            if resp.status_code != 200:
                logger.warning(
                    "Failed to fetch active models: HTTP %s from %s",
                    resp.status_code,
                    url,
                )
                return {}
            data: dict[str, dict[str, str]] = resp.json()
            logger.debug("Active models for session %s: %s", session_id, data)
            return data
        except Exception as e:
            logger.warning(
                "Failed to fetch active models from %s: %s",
                url,
                e,
            )
            return {}


def format_model_info(info: dict[str, dict[str, str]]) -> str:
    """Build a compact one-line model summary from active models config.

    Strips provider prefixes and joins roles, e.g.::

        Chat:deepseek-v4-flash | Guard:llama-guard3 | Embedding:bge-m3

    :param info: The dict returned by ``fetch_active_models`` /
        ``fetch_active_models_sync``.
    :returns: Empty string if no models are configured.
    """
    parts: list[str] = []
    for role, cfg in info.items():
        raw = cfg.get("model", "")
        if raw:
            parsed = parse_model_name(raw)
            name_short = parsed.model_name if parsed.model_name else raw
        else:
            name_short = "?"
        parts.append(f"{role.capitalize()}: {name_short}")
    result = " | ".join(parts)
    logger.debug("Formatted model info: %s", result)
    return result


def stream_events_sync(
    query: str,
    session_id: str,
    server_url: str,
) -> Generator[dict, None, None]:
    """Synchronous counterpart of :func:`stream_events`.

    Intended for the Streamlit frontend only.

    Streamlit runs the script top-to-bottom on every interaction, and
    ``st.write_stream`` consumes a **synchronous** generator.  Calling
    ``asyncio.run()`` inside a Streamlit re-run context creates nested
    event-loop problems, so the SSE client must be sync here.

    Async frontends (NiceGUI, TUI) should use :func:`stream_events`
    instead.

    :param query: User's query string.
    :param session_id: Opaque session identifier.
    :param server_url: Base URL of the backend API server.
    """
    url = f"{server_url}/query/stream"
    with httpx.Client(timeout=None) as client:
        with client.stream(
            "POST",
            url,
            json={"query": query, "session_id": session_id},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    if line.strip():
                        logger.warning("Skipping non-data line: %s", line[:80])
                    continue
                # Strip the SSE "data: " prefix (6 chars) to get raw JSON.
                yield json.loads(line[6:])
