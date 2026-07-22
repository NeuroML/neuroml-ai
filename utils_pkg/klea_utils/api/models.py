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


class ChatModelConfigPayload(BaseModel):
    model: str
    api_key: str | None = None
    base_url: str | None = None
    provider: str | None = None
    user_id: str = ""


def create_models_router() -> APIRouter:
    """Create an APIRouter for per-chat model configuration.

    ``GET /chat/{user_id}/{chat_id}/models/overrides``
        Returns stored model overrides for a chat.

    ``GET /chat/{user_id}/{chat_id}/models/active``
        Returns resolved model config (defaults merged with overrides).

    ``POST /chat/{user_id}/{chat_id}/models/overrides/{role}``
        Stores per-chat model overrides in ``app.state.chat_sessions``.
    """
    router = APIRouter(prefix="/chat", tags=["models"])

    @router.get("/{user_id}/{chat_id}/models/overrides")
    async def get_chat_model_overrides(user_id: str, chat_id: str, request: Request):
        chat_sessions = request.app.state.chat_sessions
        key = f"{user_id}:{chat_id}"
        return chat_sessions.get(key, {}).get("models", {})

    @router.get("/{user_id}/{chat_id}/models/active")
    async def get_chat_active_models(user_id: str, chat_id: str, request: Request):
        """Return the resolved model config per role (defaults + chat overrides).

        Reads the graph's ``llm_models`` dict for defaults and merges any
        per-chat overrides on top.
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

        # Add the embedding model (fixed, not chat-configurable)
        if graph.embedding_model:
            defaults["embedding"] = {"model": graph.embedding_model}

        key = f"{user_id}:{chat_id}"
        overrides = request.app.state.chat_sessions.get(key, {}).get("models", {})
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

    @router.post("/{user_id}/{chat_id}/models/overrides/{role}")
    async def set_chat_model_override(
        user_id: str,
        chat_id: str,
        role: str,
        payload: ChatModelConfigPayload,
        request: Request,
    ):
        chat_sessions = request.app.state.chat_sessions
        key = f"{user_id}:{chat_id}"
        data = chat_sessions.setdefault(key, {}).setdefault("models", {})
        data[role] = payload.model_dump(exclude_none=True)
        return {
            "status": "ok",
            "chat_id": chat_id,
            "role": role,
            "model": payload.model,
        }

    return router
