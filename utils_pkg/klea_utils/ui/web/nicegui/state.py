#!/usr/bin/env python3
"""
In-memory chat state for the NiceGUI frontend.

Keyed by ``{user_id}:{chat_id}`` so that colliding chat_ids across
different users do not interfere.

File: klea_utils/ui/web/nicegui/state.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from datetime import datetime

from ....plogging import setup_logger

logger = setup_logger(__name__)

# Per-chat data store.  External modules may read/write this dict
# directly for performance; the helper functions below cover the
# common create-or-get and sorted-lookup cases.
#
# Using plain dicts rather than Pydantic BaseModel because this is a
# simple in-memory frontend cache (not an API contract) and NiceGUI
# naturally works with dict access.  The saved schema is documented
# inline in ensure_chat() below.
chats: dict[str, dict] = {}


def ensure_chat(user_id: str, chat_id: str) -> dict:
    """Return the chat session dict for *user_id* / *chat_id*, creating it if missing.

    Each chat session dict has the following keys::

        name                Human-readable display name (auto-generated)
        created             ``datetime.timestamp()`` of creation (float).
        pinned              Whether the chat session is pinned to the top of the list.
        messages            List of ``(text, stamp, is_user)`` tuples where
                            *is_user* is ``True`` for user messages and
                            ``False`` for bot / system messages.
        inspector_entries   List of dicts with info/debug events for the most
                            recent query in this chat session.
        inspector_expanded  Set of indices into *inspector_entries* that are
                            currently expanded in the UI.
        state_sections      Dict of ``{node_label: section_data}`` for the status
                            pane, ordered by first insertion (per node label).
        model_info          Dict of active model config per role
                            (from ``fetch_active_models``).
        token_usage         Numeric token totals accumulated for this in-memory
                            chat session.
    """
    key = f"{user_id}:{chat_id}"
    if key not in chats:
        now = datetime.now().astimezone()
        logger.debug("creating %s", key)
        chats[key] = {
            "name": chat_id.replace("-", " ").title(),
            "created": now.timestamp(),
            "pinned": False,
            "messages": [],
            "inspector_entries": [],
            "inspector_expanded": set(),
            "state_sections": {},
            "model_info": {},
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        }
    else:
        logger.debug("found existing %s", key)
    return chats[key]


def get_chats_sorted(user_id: str) -> list[tuple[str, dict]]:
    """Return (chat_id, data) pairs for *user_id*, pinned first, then by creation desc.

    Filters by the user_id prefix so that in a multi-browser scenario
    (same NiceGUI process) each user only sees their own chats.
    """
    prefix = f"{user_id}:"
    all_keys = list(chats.keys())
    matched = [k for k in all_keys if k.startswith(prefix)]
    logger.debug(
        "user_id=%s prefix=%r chats_keys=%s matched=%s",
        user_id,
        prefix,
        all_keys,
        matched,
    )
    items = [(k.split(":", 1)[1], v) for k, v in chats.items() if k.startswith(prefix)]
    items.sort(key=lambda x: (not x[1]["pinned"], -x[1]["created"]))
    return items
