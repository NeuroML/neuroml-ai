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
import logging
from collections.abc import AsyncGenerator, Generator

import httpx

logger = logging.getLogger(__name__)


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
