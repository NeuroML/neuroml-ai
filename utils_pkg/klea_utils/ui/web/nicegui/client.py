#!/usr/bin/env python3
"""
Server API client for the NiceGUI frontend.

All functions accept the *server_url* as their first argument so the
caller can point them at any running Klea backend without shared state.

File: klea_utils/ui/web/nicegui/client.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from datetime import datetime

import httpx

from ....plogging import setup_logger
from .state import chats, ensure_chat

logger = setup_logger(__name__)


async def hydrate_chats(server_url: str, user_id: str, current_chat_id: str) -> None:
    """Fetch chats and messages from the server and populate the local state.

    Chats are only created server-side by ``/query/stream``, so if the
    server has no data for this user yet the local store stays empty.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            logger.debug("GET /chat/%s", user_id)
            resp = await client.get(f"{server_url}/chat/{user_id}")
            logger.debug("status=%d", resp.status_code)
            if resp.status_code == 200:
                chats_from_server = resp.json()
                logger.debug(
                    "server returned %d chat(s): %s",
                    len(chats_from_server),
                    [c.get("chat_id") for c in chats_from_server],
                )
                for chat_data in chats_from_server:
                    chat_id = chat_data["chat_id"]
                    key = f"{user_id}:{chat_id}"
                    ensure_chat(user_id, chat_id)
                    chats[key]["name"] = chat_data.get("title", chat_id)
                    chats[key]["created"] = chat_data.get("created_at", 0)

            current_key = f"{user_id}:{current_chat_id}" if current_chat_id else None
            if current_key and current_key in chats:
                resp = await client.get(
                    f"{server_url}/chat/{user_id}/{current_chat_id}/messages"
                )
                if resp.status_code == 200:
                    session = ensure_chat(user_id, current_chat_id)
                    session["messages"] = [
                        (
                            msg["content"],
                            datetime.fromtimestamp(msg["created_at"])
                            .astimezone()
                            .strftime("%X"),
                            msg["role"] == "user",
                        )
                        for msg in resp.json()
                    ]
    except Exception as e:
        logger.warning("Failed to hydrate chats from server: %s", e)


async def create_chat_on_server(server_url: str, user_id: str, chat_id: str) -> None:
    """POST a new chat to the server so it persists."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            title = chats.get(f"{user_id}:{chat_id}", {}).get("name", chat_id)
            resp = await client.post(
                f"{server_url}/chat/{user_id}",
                json={"chat_id": chat_id, "title": title},
            )
            if resp.status_code == 200:
                chat_data = resp.json()
                ensure_chat(user_id, chat_id)
                chats[f"{user_id}:{chat_id}"]["name"] = chat_data.get("title", chat_id)
    except Exception as e:
        logger.warning("Failed to create chat on server: %s", e)


async def delete_chat_on_server(server_url: str, user_id: str, chat_id: str) -> None:
    """DELETE the chat on the server."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(f"{server_url}/chat/{user_id}/{chat_id}")
    except Exception as e:
        logger.warning("Failed to delete chat on server: %s", e)


async def rename_chat_on_server(
    server_url: str, user_id: str, chat_id: str, title: str
) -> None:
    """PATCH the chat title on the server."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.patch(
                f"{server_url}/chat/{user_id}/{chat_id}",
                json={"title": title},
            )
    except Exception as e:
        logger.warning("Failed to rename chat on server: %s", e)
