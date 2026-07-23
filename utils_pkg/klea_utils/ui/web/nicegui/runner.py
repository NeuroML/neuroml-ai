#!/usr/bin/env python3
"""
NiceGUI runner for Klea web interfaces.

This module implements the main UI logic for the NiceGUI web interface,
including the 3-column layout, chat functionality, and inspector panel.

File: klea_utils/ui/web/nicegui/runner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import uuid
from datetime import datetime

import coolname
import httpx
from nicegui import app, background_tasks, ui
from nicegui.events import GenericEventArguments

from klea_utils.api.sse import (
    fetch_active_models,
    format_model_info,
    stream_events,
)
from klea_utils.api.utils import check_api_is_ready

from ....plogging import setup_logger
from .widgets import ChatBubble

logger = setup_logger(__name__)

# Per-chat data store.
# Keyed by ``{user_id}:{chat_id}`` so that colliding chat_ids across
# different users do not interfere.  Without the user_id prefix,
# two users whose sessions happen to share a chat_id (possible with
# a finite slug pool) would overwrite each other's data.
_chats: dict[str, dict] = {}


def _ensure_chat(user_id: str, chat_id: str) -> dict:
    """Return the session dict for *user_id* / *chat_id*, creating it if missing.

    Each session dict has the following keys::

        name                Human-readable display name (auto-generated)
        created             ``datetime.timestamp()`` of creation (float).
        pinned              Whether the session is pinned to the top of the list.
        messages            List of ``(text, stamp, is_user)`` tuples where
                            *is_user* is ``True`` for user messages and
                            ``False`` for bot / system messages.
        inspector_entries   List of dicts with info/debug events for the most
                            recent query in this session.
        inspector_expanded  Set of indices into *inspector_entries* that are
                            currently expanded in the UI.
    """
    key = f"{user_id}:{chat_id}"
    if key not in _chats:
        now = datetime.now()
        _chats[key] = {
            "name": chat_id.replace("-", " ").title(),
            "created": now.timestamp(),
            "pinned": False,
            "messages": [],
            "inspector_entries": [],
            "inspector_expanded": set(),
        }
    return _chats[key]


async def _hydrate_chats(server_url: str, user_id: str, current_chat_id: str) -> None:
    """Fetch chats and messages from the server and populate ``_chats``."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Fetch chat list
            resp = await client.get(f"{server_url}/chat/{user_id}")
            if resp.status_code == 200:
                for chat_data in resp.json():
                    chat_id = chat_data["chat_id"]
                    key = f"{user_id}:{chat_id}"
                    _ensure_chat(user_id, chat_id)
                    _chats[key]["name"] = chat_data.get("title", chat_id)
                    _chats[key]["created"] = chat_data.get("created_at", 0)

            # Ensure current chat exists on server
            current_key = f"{user_id}:{current_chat_id}"
            if current_key not in _chats:
                resp = await client.post(
                    f"{server_url}/chat/{user_id}",
                    json={"chat_id": current_chat_id},
                )
                if resp.status_code == 200:
                    chat_data = resp.json()
                    _ensure_chat(user_id, current_chat_id)
                    _chats[current_key]["name"] = chat_data.get(
                        "title", current_chat_id
                    )

            # Fetch messages for current chat
            resp = await client.get(
                f"{server_url}/chat/{user_id}/{current_chat_id}/messages"
            )
            if resp.status_code == 200:
                session = _ensure_chat(user_id, current_chat_id)
                for msg in resp.json():
                    session["messages"].append(
                        (
                            msg["content"],
                            datetime.fromtimestamp(msg["created_at"]).strftime("%X"),
                            msg["role"] == "user",
                        )
                    )
    except Exception as e:
        logger.warning("Failed to hydrate chats from server: %s", e)


def _get_chats_sorted() -> list[tuple[str, dict]]:
    """Return (chat_id, data) pairs, pinned first, then by creation desc."""
    items = [(k.split(":", 1)[1], v) for k, v in _chats.items()]
    items.sort(key=lambda x: (not x[1]["pinned"], -x[1]["created"]))
    return items


def setup_layout(
    chat_id: str,
    server_url: str,
    user_id: str = "",
    title: str = "Klea",
    subtitle: str = "",
    disclaimer: str = "",
    footer_text: str = 'Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
    model_info: str = "",
) -> None:
    """Build the full page UI: header, drawers, chat area, and footer.

    User messages appear as right-aligned bubbles (grey background);
    system / bot messages are left-aligned, full-width and transparent,
    matching the Gemini/ChatGPT model without avatars.

    Layout (left to right)::

        [left_drawer | center_column | right_drawer]

    The left drawer uses Quasar's *mini* mode to provide a
    ChatGPT-style rail that shows only icons when collapsed and
    full text when expanded.

    :param chat_id: Chat conversation identifier.
    :param server_url: Base URL of the backend API server.
    :param user_id: Opaque persistent user identifier.
    :param title: Bold application title in the header bar.
    :param subtitle: Optional smaller text shown next to *title*
        in the header.
    :param disclaimer: Optional text shown below the chat input.
    :param footer_text: HTML content for the footer bar.
    :param model_info: Compact model summary string (from ``format_model_info``).
    """
    # --- CSS overrides ---
    # Make q-page a flex container so the nicegui-content can flex-fill
    # the available page height, which in turn lets the center column
    # grow and pin the input row to the bottom.
    ui.add_css(".q-page { display: flex; flex-direction: column; }")
    ui.add_css(
        ".nicegui-content { display: flex; flex-direction: column; flex: 1; min-height: 0; }"
    )
    # Collapse long bot messages to 4 lines with an expand / collapse toggle.
    ui.add_css(".msg-collapsed { max-height: 6em; overflow: hidden; }")
    ui.add_css(".msg-expanded { max-height: none; }")
    ui.add_css(
        ".inspector-entry > summary { list-style: none; display: flex; align-items: center; gap: 0.25rem; }"
    )
    ui.add_css(
        ".inspector-entry > summary::before { content: '\\25B6'; font-size: 0.65rem; margin-right: 0.35rem; transition: transform 0.15s; }"
    )
    ui.add_css(".inspector-entry[open] > summary::before { content: '\\25BC'; }")
    ui.add_css(
        ".inspector-details summary { list-style: none; display: flex; align-items: center; gap: 0.25rem; }"
    )
    ui.add_css(
        ".inspector-details summary::before { content: '\\25B6'; font-size: 0.6rem; margin-right: 0.35rem; }"
    )
    ui.add_css(".inspector-details[open] summary::before { content: '\\25BC'; }")
    ui.add_css(
        ".inspector-details .md-div { overflow: hidden !important; height: auto !important; }"
    )
    ui.add_css(
        ".inspector-details code { white-space: pre-wrap !important; word-break: break-all !important; }"
    )
    ui.add_css(
        ".q-tooltip { max-width: 350px !important; overflow: visible !important; white-space: nowrap !important; }"
    )

    # --- Persistent dark mode ---
    dark = ui.dark_mode()
    if "dark_mode" not in app.storage.user:
        app.storage.user["dark_mode"] = False
    dark.bind_value(app.storage.user, "dark_mode")

    mini_state = True
    # Mutable containers so refreshable functions can pick up changes.
    _current_chat_id = [chat_id]
    toggle_icon_ref: list = [None]

    _expanded: set[int] = set()

    @ui.refreshable
    def _chat_messages() -> None:
        """Render the message list for the currently active session.

        User messages appear as right-aligned bubbles with a grey
        background (``sent=True``).  System / bot messages appear as
        left-aligned, full-width, transparent text (``sent=False``),
        matching the Gemini/ChatGPT model where only user input has a
        visible bubble.
        """
        session = _chats.get(f"{user_id}:{_current_chat_id[0]}")
        msgs = session["messages"] if session else []
        if msgs:
            for idx, (text, stamp, is_user) in enumerate(msgs):
                collapsed = idx not in _expanded
                ChatBubble(
                    text=text,
                    stamp=stamp,
                    is_user=is_user,
                    collapsed=collapsed,
                    idx=idx,
                    on_copy=lambda t=text: ui.run_javascript(
                        f"navigator.clipboard.writeText({json.dumps(t)})"
                    ),
                    on_expand=lambda i=idx: (
                        (_expanded.discard(i) if i in _expanded else _expanded.add(i))
                        or _chat_messages.refresh()
                    ),
                )
        ui.run_javascript(
            "document.querySelector('.chat-scroll-area')?.scrollTo(0, 999999)"
        )

    def _switch_chat(chat_id: str) -> None:
        """Switch the active chat without a page reload."""
        app.storage.user["chat_id"] = chat_id
        _current_chat_id[0] = chat_id
        logger.debug("Switched to chat %s", chat_id)
        _chat_messages.refresh()
        _render_chat_list.refresh()
        _inspector_panel.refresh()

    async def _delete_chat_on_server(chat_id: str) -> None:
        """DELETE the chat on the server."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.delete(f"{server_url}/chat/{user_id}/{chat_id}")
        except Exception as e:
            logger.warning("Failed to delete chat on server: %s", e)

    def _delete_chat(chat_id: str) -> None:
        """Remove a session from the store and server.

        If the currently active session is deleted, the next available
        session becomes active (or a fresh one is created).
        """
        background_tasks.create(_delete_chat_on_server(chat_id))
        _chats.pop(f"{user_id}:{chat_id}", None)
        if _current_chat_id[0] == chat_id:
            remaining = _get_chats_sorted()
            if remaining:
                _switch_chat(remaining[0][0])
            else:
                new_chat_id = coolname.generate_slug(2)
                _ensure_chat(user_id, new_chat_id)
                _switch_chat(new_chat_id)
        else:
            _render_chat_list.refresh()

    def _toggle_pin(chat_id: str) -> None:
        """Flip the pinned flag for a session and refresh the list."""
        session = _ensure_chat(user_id, chat_id)
        session["pinned"] = not session["pinned"]
        _render_chat_list.refresh()

    def _toggle_left_drawer():
        """Switch the left drawer between mini (rail) and full width.

        When *mini_state* is ``True`` the drawer shows a 64 px narrow
        rail with only item icons; otherwise it expands to ``w-80``
        (320 px) showing icons and labels.  The toggle-button icon
        changes direction to hint at the available action.
        """
        nonlocal mini_state
        mini_state = not mini_state
        if mini_state:
            left_drawer.props("mini")
            toggle_icon_ref[0].name = "keyboard_double_arrow_right"
        else:
            left_drawer.props(remove="mini")
            toggle_icon_ref[0].name = "keyboard_double_arrow_left"

    async def _create_chat_on_server(chat_id: str) -> None:
        """POST a new chat to the server so it persists."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{server_url}/chat/{user_id}",
                    json={"chat_id": chat_id},
                )
                if resp.status_code == 200:
                    chat_data = resp.json()
                    _ensure_chat(user_id, chat_id)
                    _chats[f"{user_id}:{chat_id}"]["name"] = chat_data.get(
                        "title", chat_id
                    )
        except Exception as e:
            logger.warning("Failed to create chat on server: %s", e)

    def _new_chat():
        """Create a fresh session and switch to it."""
        chat_id = coolname.generate_slug(2)
        _ensure_chat(user_id, chat_id)
        _switch_chat(chat_id)
        background_tasks.create(_create_chat_on_server(chat_id))

    def _rename_chat(chat_id: str) -> None:
        """Open a dialog to rename a chat and persist on server."""
        chat = _ensure_chat(user_id, chat_id)
        dialog = ui.dialog()

        async def _save():
            chat["name"] = inp.value
            dialog.close()
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.patch(
                        f"{server_url}/chat/{user_id}/{chat_id}",
                        json={"title": inp.value},
                    )
            except Exception as e:
                logger.warning("Failed to rename chat on server: %s", e)
            _render_chat_list.refresh()

        with dialog:
            with ui.card():
                ui.label("Rename chat").classes("text-lg font-bold")
                inp = ui.input(value=chat["name"]).on("keydown.enter", _save)
                with ui.row().classes("w-full justify-end"):
                    ui.button("Cancel", on_click=dialog.close)
                    ui.button("Save", on_click=_save).props("unelevated color=primary")
        dialog.open()

    @ui.refreshable
    def _render_chat_list():
        """Render the sorted list of session entries in the left drawer.

        Each entry shows the session name (bold for the active session),
        a tooltip with the creation timestamp, and a three-dot context
        menu for rename / pin / delete.  Refresh this to pick up newly
        created sessions without a page reload.
        """
        for chat_id, sdata in _get_chats_sorted():
            is_current = chat_id == _current_chat_id[0]
            with (
                ui.item(
                    on_click=lambda s=chat_id: _switch_chat(s),
                )
                .props("dense")
                .classes("w-full")
                .on("dblclick", lambda s=chat_id: _rename_chat(s))
            ):
                with ui.item_section().props("avatar"):
                    ui.icon("push_pin" if sdata["pinned"] else "history")
                with ui.item_section():
                    label_cls = "text-xs font-bold" if is_current else "text-xs"
                    ui.label(sdata["name"]).classes(label_cls)
                    ui.tooltip(
                        "Created: "
                        + datetime.fromtimestamp(sdata["created"]).strftime(
                            "%a %d %b %Y at %X"
                        )
                    )
                # Three-dot context menu (right-aligned).
                with ui.item_section().props("side"):
                    with (
                        ui.button(icon="more_vert")
                        .props("flat dense round")
                        .on("click.stop", lambda: None)
                    ):
                        with ui.menu():
                            with ui.menu_item(
                                on_click=lambda s=chat_id: _rename_chat(s)
                            ):
                                with ui.item_section().props("avatar"):
                                    ui.icon("edit")
                                with ui.item_section():
                                    ui.label("Rename")
                            with ui.menu_item(
                                on_click=lambda s=chat_id: _toggle_pin(s)
                            ):
                                with ui.item_section().props("avatar"):
                                    ui.icon("push_pin")
                                with ui.item_section():
                                    ui.label("Unpin" if sdata["pinned"] else "Pin")
                            with ui.menu_item(
                                on_click=lambda s=chat_id: _delete_chat(s)
                            ):
                                with ui.item_section().props("avatar"):
                                    ui.icon("delete")
                                with ui.item_section():
                                    ui.label("Delete")

    # ---- Header ----
    with ui.header().classes("items-center"):
        ui.label(title).classes("text-xl font-bold")
        if subtitle:
            ui.label(subtitle).classes("text-sm text-grey-4 ml-2 mr-2")
        ui.space()

        def _toggle_dark():
            """Flip the dark-mode flag in ``app.storage.user``."""
            dark.value = not dark.value

        ui.button(icon="dark_mode", on_click=_toggle_dark).props(
            "flat color=white round"
        )

    # ---- Left drawer (rail mode by default) ----
    with (
        ui.left_drawer(value=True)
        .props("mini")
        .classes("w-80 overflow-x-hidden p-2") as left_drawer
    ):
        # Items use QItem + QItemSection(avatar) so that Quasar
        # automatically hides the label when the drawer is in mini mode.
        with ui.item(on_click=_new_chat).props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                ui.icon("add")
                ui.tooltip("Start a new conversation")
            with ui.item_section():
                ui.label("New Chat")
        with (
            ui.item(on_click=lambda: right_drawer.toggle())
            .props("dense")
            .classes("w-full")
        ):
            with ui.item_section().props("avatar"):
                ui.icon("info")
                ui.tooltip("View pipeline details in the inspection pane")
            with ui.item_section():
                ui.label("Inspector")
        ui.separator()

        # Session list header
        with ui.item().props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                ui.icon("chat")
            with ui.item_section():
                ui.label("Chats").classes("text-sm font-bold")

        _render_chat_list()

        ui.space()
        # Toggle button at the bottom of the drawer.
        with ui.item(on_click=_toggle_left_drawer).props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                toggle_icon_ref[0] = ui.icon("keyboard_double_arrow_right")
                ui.tooltip("Expand or collapse the sidebar")
            with ui.item_section():
                ui.label("").classes("text-xs")

    def _toggle_inspector_entry(idx: int) -> None:
        """Toggle the expanded/collapsed state of an inspector entry."""
        session = _chats.get(f"{user_id}:{_current_chat_id[0]}")
        if not session:
            return
        expanded = session.setdefault("inspector_expanded", set())
        if idx in expanded:
            expanded.discard(idx)
        else:
            expanded.add(idx)

    # ---- Right drawer (inspector, hidden by default) ----
    with (
        ui.right_drawer(value=False)
        .props("width=420")
        .classes("overflow-y-auto") as right_drawer
    ):
        ui.label("Inspector").classes("text-sm font-bold mb-0")
        ui.separator().classes("my-0.5")

        if model_info:
            with ui.row().classes("items-center no-wrap my-0.5"):
                ui.icon("settings").classes("text-sm")
                with ui.label(model_info).classes("text-xs truncate"):
                    ui.tooltip("Current models per role")
            ui.separator().classes("my-0.5")

        @ui.refreshable
        def _inspector_panel() -> None:
            session = _chats.get(f"{user_id}:{_current_chat_id[0]}")
            if not session:
                logger.debug(
                    "No session found for %s, inspector empty", _current_chat_id[0]
                )
                return
            entries = session.get("inspector_entries", [])
            logger.debug(
                "Rendering inspector panel: %d entries for session %s",
                len(entries),
                _current_chat_id[0],
            )
            if not entries:
                ui.label("Debug events will appear here").classes(
                    "text-sm text-gray-500"
                )
                return
            for idx, entry in enumerate(entries):
                heading = entry.get("heading", "")
                timing = entry.get("timing_seconds", None)
                expanded = idx in session["inspector_expanded"]
                with (
                    ui.element("details")
                    .props("open" if expanded else "")
                    .classes("inspector-entry mb-2 w-full")
                ):
                    with (
                        ui.element("summary")
                        .classes("text-xs font-bold cursor-pointer w-full")
                        .on("click", lambda i=idx: _toggle_inspector_entry(i))
                    ):
                        with ui.row().classes("w-full flex-nowrap items-center"):
                            ui.label(heading)
                            if timing:
                                ui.label(f"({timing:.1f}s)").classes(
                                    "text-xs text-grey-5"
                                )
                    ui.label(entry.get("summary", "")).classes(
                        "text-xs text-grey-6 mb-1 w-full"
                    )
                    details = entry.get("details", {})
                    if details:
                        with ui.element("details").classes(
                            "inspector-details text-xs text-grey-5 cursor-pointer w-full"
                        ):
                            with ui.element("summary").classes("text-xs w-full"):
                                ui.label("View details")
                            ui.code(
                                json.dumps(details, indent=2), language="json"
                            ).classes("text-xs")

        _inspector_panel()

    # ---- Center: chat messages + input (pinned to bottom) ----
    with (
        ui.column()
        .classes("w-full px-8")
        .style("flex: 1; min-height: 0; display: flex; flex-direction: column;")
    ):
        with ui.scroll_area().classes("w-full grow chat-scroll-area"):
            _chat_messages()
            _stream_container = ui.column().classes("w-full")

        with ui.row().classes("w-full no-wrap items-end py-4"):
            text = (
                ui.textarea(placeholder="Start a conversation")
                .props("rounded outlined input-class=mx-3 autogrow")
                .classes("flex-grow")
            )

            async def _do_stream(query: str) -> None:
                """Stream the RAG pipeline progress and final answer."""
                session = _ensure_chat(user_id, _current_chat_id[0])
                logger.debug(
                    "Clearing inspector entries for session %s", _current_chat_id[0]
                )
                session["inspector_entries"] = []
                session["inspector_expanded"] = set()
                _inspector_panel.refresh()
                with _stream_container:
                    pg_row = ui.row().classes("w-full items-center gap-2 p-2")
                    with pg_row:
                        ui.spinner(type="dots").classes("w-4 h-4")
                        pg_label = ui.label("").classes("text-xs text-grey-5 italic")

                full_response = ""
                try:
                    async for event in stream_events(
                        query, _current_chat_id[0], server_url, user_id=user_id
                    ):
                        t = event.get("type", "?")
                        if t == "progress":
                            pg_label.set_text(f"{event.get('node', '')}")
                        elif t == "debug":
                            data = event.get("data", {})
                            session["inspector_entries"].append(
                                {
                                    "type": t,
                                    "node": event.get("node", ""),
                                    "heading": data.get("heading", ""),
                                    "summary": data.get("summary", ""),
                                    "details": data.get("details", {}),
                                    "timing_seconds": data.get("timing_seconds", None),
                                }
                            )
                            _inspector_panel.refresh()
                        elif t == "complete":
                            pg_row.delete()
                            full_response = event.get("message_for_user", full_response)
                            _ensure_chat(user_id, _current_chat_id[0])[
                                "messages"
                            ].append(
                                (full_response, datetime.now().strftime("%X"), False)
                            )
                            _chat_messages.refresh()
                            break
                        elif t == "error":
                            pg_row.delete()
                            _ensure_chat(user_id, _current_chat_id[0])[
                                "messages"
                            ].append(
                                (
                                    f"Error: {event.get('message', 'Unknown error')}",
                                    datetime.now().strftime("%X"),
                                    False,
                                )
                            )
                            _chat_messages.refresh()
                            break
                except httpx.RequestError as e:
                    pg_row.delete()
                    _ensure_chat(user_id, _current_chat_id[0])["messages"].append(
                        (f"Connection error: {e}", datetime.now().strftime("%X"), False)
                    )
                    _chat_messages.refresh()

            def send() -> None:
                """Append the current input text as a user message."""
                if not text.value.strip():
                    return
                stamp = datetime.now().strftime("%X")
                query = text.value
                text.value = ""
                _ensure_chat(user_id, _current_chat_id[0])["messages"].append(
                    (query, stamp, True)
                )
                _chat_messages.refresh()
                _render_chat_list.refresh()

                background_tasks.create(_do_stream(query))

            with text.add_slot("append"):
                with ui.button(icon="send", on_click=send).props(
                    "flat dense round color=primary"
                ):
                    ui.tooltip("Enter to send, Shift+Enter for newline")

        # Plain Enter sends the message and prevents the default newline
        def handle_enter(e: GenericEventArguments):
            if e.args.get("shiftKey"):
                text.value += "\n"
            else:
                send()

        text.on("keydown.enter.exact.prevent", handle_enter)
        # Clicking the send icon inside the textarea also sends.
        text.on("click:append", send)

        if disclaimer:
            ui.label(disclaimer).classes("text-xs text-grey-5 pb-2 w-full text-center")

    # ---- Footer ----
    with ui.footer().classes("bg-grey-3 dark:bg-grey-9 text-xs py-1"):
        ui.html(footer_text).classes("w-full text-center text-grey-6")


def run_nicegui_app(
    title: str,
    server_url: str,
    subtitle: str = "",
    disclaimer: str = "",
    footer_text: str = 'Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
    debug: bool = False,
) -> None:
    """Start the NiceGUI web server with the Klea chat interface.

    This function is the main entry point for the NiceGUI frontend.
    It registers a ``@ui.page("/")`` handler that builds the full UI
    via :func:`setup_layout` and then starts the NiceGUI server.

    :param title: Application title (displayed in the header and
        browser tab).
    :param server_url: Base URL of the backend API server
        (e.g. ``http://127.0.0.1:8005``).
    :param subtitle: Optional smaller text shown next to *title*
        in the header.
    :param disclaimer: Optional text shown below the chat input.
    :param footer_text: HTML content for the footer bar.
    :param debug: When ``True``, enable NiceGUI's file-watch hot
        reload (``reload=True``).  Set to ``False`` in production.
    """

    @ui.page("/", response_timeout=30)
    async def main_page():
        """Build the main page after ensuring the WebSocket is connected.

        ``ui.run_javascript`` (used inside ``_chat_messages``) only
        works after the client has connected, so we await
        ``ui.context.client.connected()`` first.
        """
        await ui.context.client.connected()

        # The page builder runs exactly once -- build the appropriate page
        # directly rather than showing a loading spinner and then swapping.
        # The ``response_timeout=30`` on the page decorator gives the health
        # check time to complete before the client sees a timeout.
        try:
            await check_api_is_ready(f"{server_url}/health/ready")
        except Exception:
            ui.add_css(".nicegui-content { display: flex; flex: 1; }")
            with ui.column().classes("w-full h-full items-center justify-center gap-4"):
                ui.icon("cloud_off", size="4rem").classes("text-grey-5")
                ui.label("Backend unavailable").classes("text-xl text-grey-7")
                ui.label("Please check that the Klea server is running.").classes(
                    "text-grey-5"
                )
            return

        if "user_id" not in app.storage.user:
            app.storage.user["user_id"] = str(uuid.uuid4())

        user_id = app.storage.user["user_id"]

        if "chat_id" not in app.storage.user:
            app.storage.user["chat_id"] = coolname.generate_slug(2)

        chat_id = app.storage.user["chat_id"]

        await _hydrate_chats(server_url, user_id, chat_id)

        # Fetch model info (non-fatal — just won't show in header on failure)
        active_models = await fetch_active_models(server_url, user_id, chat_id)
        model_info = format_model_info(active_models)

        setup_layout(
            chat_id=chat_id,
            user_id=user_id,
            server_url=server_url,
            title=title,
            subtitle=subtitle,
            disclaimer=disclaimer,
            footer_text=footer_text,
            model_info=model_info,
        )

    ui.run(
        port=7860,
        host="0.0.0.0",
        title=title,
        show=False,
        reload=debug,
        storage_secret="klea-nicegui-secret-change-me",
    )
