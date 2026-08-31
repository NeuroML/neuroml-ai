#!/usr/bin/env python3
"""
Shared chat endpoint factory for Klea packages.

File: klea_utils/api/chat.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import copy
import json
import logging
import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from klea_utils.api.sessions_db import SessionStore

logger = logging.getLogger(__name__)


class ChatPayload(BaseModel):
    query: str = Field(..., min_length=1)
    chat_id: str = Field(..., pattern=r"^[^:]+$")
    user_id: str = Field(default="", pattern=r"^[^:]*$")


def create_chat_router() -> APIRouter:
    """Create an APIRouter with ``/query`` and ``/query/stream`` endpoints.

    The router reads the graph instance from ``request.app.state.graph``
    (set by :func:`klea_utils.api.app.make_app`).
    Chat-session data is stored via the ``SessionStore`` on
    ``app.state.chat_sessions``.
    """
    router = APIRouter()

    @router.post("/query")
    async def query(request: Request, payload: ChatPayload):
        # Lazy: BaseLangGraph is the base class for all graphs
        from klea_utils.graph.base import BaseLangGraph, model_overrides_ctx

        if (
            not getattr(request.app.state, "is_ready", False)
            or not getattr(request.app.state, "graph", None)
            or getattr(request.app.state.graph, "graph", None) is None
        ):
            raise HTTPException(status_code=503, detail="Service not ready")

        graph: BaseLangGraph = request.app.state.graph
        store: SessionStore = request.app.state.chat_sessions
        thread_id = f"user_{payload.user_id}:chat_{payload.chat_id}"

        store.create_chat(payload.user_id, payload.chat_id)
        overrides = store.get_overrides(payload.user_id, payload.chat_id)
        token = model_overrides_ctx.set(copy.deepcopy(overrides or {}))
        try:
            result = await graph.run_graph_invoke(payload.query, thread_id)
            message = result if isinstance(result, str) else str(result)
            store.add_message(payload.user_id, payload.chat_id, "user", payload.query)
            store.add_message(payload.user_id, payload.chat_id, "assistant", message)
        except ValueError as e:
            logger.warning(f"Bad request: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            logger.warning(f"Service not ready: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            logger.error(f"{e}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            model_overrides_ctx.reset(token)

        return {"result": message}

    @router.post("/query/stream")
    async def query_stream(request: Request, payload: ChatPayload):
        from klea_utils.graph.base import BaseLangGraph, model_overrides_ctx

        if (
            not getattr(request.app.state, "is_ready", False)
            or not getattr(request.app.state, "graph", None)
            or getattr(request.app.state.graph, "graph", None) is None
        ):
            raise HTTPException(status_code=503, detail="Service not ready")

        graph: BaseLangGraph = request.app.state.graph
        store: SessionStore = request.app.state.chat_sessions
        thread_id = f"user_{payload.user_id}:chat_{payload.chat_id}"

        store.create_chat(payload.user_id, payload.chat_id)
        overrides = store.get_overrides(payload.user_id, payload.chat_id)

        async def event_stream():
            token = model_overrides_ctx.set(copy.deepcopy(overrides or {}))
            query = payload.query
            user_id = payload.user_id
            chat_id = payload.chat_id
            try:
                async for event in graph.run_graph_astream_events(query, thread_id):
                    t = event.get("type")
                    if t == "complete":
                        store.add_message(user_id, chat_id, "user", query)
                        store.add_message(
                            user_id,
                            chat_id,
                            "assistant",
                            event.get("message_for_user", ""),
                        )
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"{e}\n{traceback.format_exc()}")
                error_event = json.dumps(
                    {
                        "type": "error",
                        "message": str(e),
                        "error_type": type(e).__name__,
                        "node": "",
                    }
                )
                yield f"data: {error_event}\n\n"
            finally:
                model_overrides_ctx.reset(token)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
