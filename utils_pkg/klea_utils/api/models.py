#!/usr/bin/env python3
"""
Per-session model configuration endpoints for runtime model switching.

File: klea_utils/api/models.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..plogging import setup_logger

logger = setup_logger(__name__)


class SessionModelConfigPayload(BaseModel):
    model: str
    api_key: str | None = None
    base_url: str | None = None
    provider: str | None = None


def create_models_router() -> APIRouter:
    """Create an APIRouter for per-session model configuration.

    ``GET /models/session/{session_id}/overrides``
        Returns stored overrides for a session.

    ``GET /models/session/{session_id}/active``
        Returns resolved model config (defaults merged with overrides).

    ``POST /models/session/{session_id}/overrides/{role}``
        Stores per-session model overrides in ``app.state.sessions``.
    """
    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("/session/{session_id}/overrides")
    async def get_session_model_overrides(session_id: str, request: Request):
        sessions = request.app.state.sessions
        return sessions.get(session_id, {}).get("models", {})

    @router.get("/session/{session_id}/active")
    async def get_session_active_models(session_id: str, request: Request):
        """Return the resolved model config per role (defaults + session overrides).

        Reads the graph's ``llm_models`` dict for defaults and merges any
        per-session overrides on top.  The frontend can use this to display
        the effective model configuration for a session.
        """
        # Lazy: BaseLangGraph is the base class for all graphs
        from klea_utils.graph.base import BaseLangGraph

        graph: BaseLangGraph = request.app.state.graph
        defaults: dict[str, dict[str, str]] = {}
        for role, entry in graph.llm_models.items():
            cfg: dict[str, str] = {"model": entry.model_name or ""}
            if entry.parsed_model:
                cfg["provider"] = entry.parsed_model.provider or ""
            defaults[role] = cfg

        # Add the embedding model (fixed, not session-configurable)
        if graph.embedding_model:
            defaults["embedding"] = {"model": graph.embedding_model}

        overrides = request.app.state.sessions.get(session_id, {}).get("models", {})
        for role, override in overrides.items():
            if role in defaults:
                defaults[role]["model"] = override.get("model", defaults[role]["model"])
                if override.get("provider"):
                    defaults[role]["provider"] = override["provider"]
                if override.get("api_key"):
                    defaults[role]["api_key"] = f"...{override['api_key'][-4:]}"
                if override.get("base_url"):
                    defaults[role]["base_url"] = override["base_url"]
            else:
                defaults[role] = override

        return defaults

    @router.post("/session/{session_id}/overrides/{role}")
    async def set_session_model_override(
        session_id: str, role: str, payload: SessionModelConfigPayload, request: Request
    ):
        sessions = request.app.state.sessions
        data = sessions.setdefault(session_id, {}).setdefault("models", {})
        data[role] = payload.model_dump(exclude_none=True)
        return {
            "status": "ok",
            "session_id": session_id,
            "role": role,
            "model": payload.model,
        }

    return router
