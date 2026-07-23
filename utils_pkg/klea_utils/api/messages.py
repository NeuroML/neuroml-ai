#!/usr/bin/env python3
"""
Message history endpoints for chat sessions.

File: klea_utils/api/messages.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, HTTPException, Request

from klea_utils.api.sessions_db import SessionStore

from ..plogging import setup_logger

logger = setup_logger(__name__)


def create_messages_router() -> APIRouter:
    """Create an APIRouter for chat message history.

    ``GET /chat/{user_id}/{chat_id}/messages``
        Return all messages for a chat, oldest first.
    """
    router = APIRouter(prefix="/chat", tags=["messages"])

    @router.get("/{user_id}/{chat_id}/messages")
    async def get_messages(user_id: str, chat_id: str, request: Request):
        store: SessionStore = request.app.state.chat_sessions
        if not store.get_chat(user_id, chat_id):
            logger.warning("get_messages(%s, %s): chat not found", user_id, chat_id)
            raise HTTPException(status_code=404, detail="Chat not found")
        msgs = store.get_messages(user_id, chat_id)
        logger.debug("get_messages(%s, %s): %d message(s)", user_id, chat_id, len(msgs))
        return msgs

    return router
