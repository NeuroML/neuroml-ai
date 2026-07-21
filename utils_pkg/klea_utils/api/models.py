#!/usr/bin/env python3
"""
Per-session model configuration endpoints for runtime model switching.

File: klea_utils/api/models.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from fastapi import APIRouter, Request
from pydantic import BaseModel

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SessionModelConfigPayload(BaseModel):
    model: str
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    provider: str | None = None
    huggingfacehub_api_token: str | None = None


def create_models_router() -> APIRouter:
    """Create an APIRouter for per-session model configuration.

    ``GET /session/{session_id}/models``
        Returns overrides for the session (or global defaults as fallback).

    ``POST /session/{session_id}/models/{role}``
        Stores per-session model overrides in ``app.state.sessions``.
    """
    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("/session/{session_id}")
    async def get_session_models(session_id: str, request: Request):
        sessions = request.app.state.sessions
        return sessions.get(session_id, {})

    @router.post("/session/{session_id}/{role}")
    async def set_session_model(
        session_id: str, role: str, payload: SessionModelConfigPayload, request: Request
    ):
        sessions = request.app.state.sessions
        data = sessions.setdefault(session_id, {})
        data[role] = payload.model_dump(exclude_none=True)
        return {
            "status": "ok",
            "session_id": session_id,
            "role": role,
            "model": payload.model,
        }

    return router
