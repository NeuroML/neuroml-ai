#!/usr/bin/env python3
"""
Shared chat endpoint factory for Klea packages.

File: klea_utils/api/chat.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..plogging import setup_logger

logger = setup_logger(__name__)


class ChatPayload(BaseModel):
    query: str
    chat_id: str
    user_id: str = ""


def create_chat_router() -> APIRouter:
    """Create an APIRouter with ``/query`` and ``/query/stream`` endpoints.

    The router reads the graph instance from ``request.app.state.graph``
    (set by :func:`klea_utils.api.app.make_app`).
    Chat-session configs are kept in ``app.state.chat_sessions`` keyed by
    ``{user_id}:{chat_id}``.
    """
    router = APIRouter()

    def _storage_key(payload: ChatPayload) -> str:
        return f"{payload.user_id}:{payload.chat_id}"

    @router.post("/query")
    async def query(request: Request, payload: ChatPayload):
        # Lazy: BaseLangGraph is the base class for all graphs
        from klea_utils.graph.base import BaseLangGraph, model_overrides_ctx

        graph: BaseLangGraph = request.app.state.graph
        key = _storage_key(payload)
        thread_id = f"user_{payload.user_id}:chat_{payload.chat_id}"
        chat_sessions = request.app.state.chat_sessions

        chat_sessions.setdefault(key, {}).setdefault("models", {})
        model_overrides_ctx.set(chat_sessions[key]["models"])

        try:
            result = await graph.run_graph_invoke(payload.query, thread_id)
        except Exception as e:
            detail = f"{e}\n{traceback.format_exc()}"
            logger.error(detail)
            raise HTTPException(status_code=500, detail=detail)

        return {"result": result}

    @router.post("/query/stream")
    async def query_stream(request: Request, payload: ChatPayload):
        from klea_utils.graph.base import BaseLangGraph, model_overrides_ctx

        graph: BaseLangGraph = request.app.state.graph
        key = _storage_key(payload)
        thread_id = f"user_{payload.user_id}:chat_{payload.chat_id}"
        chat_sessions = request.app.state.chat_sessions

        chat_sessions.setdefault(key, {}).setdefault("models", {})
        model_overrides_ctx.set(chat_sessions[key]["models"])

        async def event_stream():
            try:
                async for event in graph.run_graph_astream_events(
                    payload.query, thread_id
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                detail = f"{e}\n{traceback.format_exc()}"
                logger.error(detail)
                error_event = json.dumps({"type": "error", "message": str(e)})
                yield f"data: {error_event}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
