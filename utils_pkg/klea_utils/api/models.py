#!/usr/bin/env python3
"""
Model management endpoints for runtime model switching.

File: klea_utils/api/models.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import traceback

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelConfigPayload(BaseModel):
    model: str
    temperature: float | None = None
    api_key: str | None = None
    base_url: str | None = None
    provider: str | None = None
    huggingfacehub_api_token: str | None = None


def create_models_router() -> APIRouter:
    """Create an APIRouter with ``GET /models`` and ``POST /models/{role}``.

    Reads/writes model entries from ``request.app.state.graph.llm_models``.
    """
    router = APIRouter(prefix="/models", tags=["models"])

    @router.get("/")
    async def list_models(request: Request):
        from klea_utils.graph.base import BaseLangGraph

        graph: BaseLangGraph = request.app.state.graph
        return {
            role: {
                "model": entry.instance.model_id
                if hasattr(entry.instance, "model_id")
                else str(entry.instance)
            }
            for role, entry in graph.llm_models.items()
        }

    @router.post("/{role}")
    async def set_model(role: str, payload: ModelConfigPayload, request: Request):
        from klea_utils.graph.base import BaseLangGraph

        graph: BaseLangGraph = request.app.state.graph
        if role not in graph.llm_models:
            raise HTTPException(404, f"Unknown model role: {role}")
        overrides = payload.model_dump(exclude={"model"}, exclude_none=True)
        try:
            graph.update_model(role, payload.model, **overrides)
        except Exception as e:
            detail = f"{e}\n{traceback.format_exc()}"
            logger.error(detail)
            raise HTTPException(400, detail=str(e))
        return {"status": "ok", "role": role, "model": payload.model}

    return router
