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
import logging
import uuid
from datetime import datetime

import httpx
from nicegui import app, background_tasks, ui
from nicegui.events import GenericEventArguments

from klea_utils.api.sse import stream_events
from klea_utils.api.utils import check_api_is_ready

from .widgets import ChatBubble

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Per-session data store.
# Keyed by session_id.
_sessions: dict[str, dict] = {}


def _ensure_session(session_id: str) -> dict:
    """Return the session dict for *session_id*, creating it if missing.

    Each session dict has the following keys::

        name                Human-readable display name (auto-generated from
                            the creation timestamp).
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
    if session_id not in _sessions:
        now = datetime.now()
        _sessions[session_id] = {
            "name": now.strftime("%a %d %b %Y at %X"),
            "created": now.timestamp(),
            "pinned": False,
            "messages": [],
            "inspector_entries": [],
            "inspector_expanded": set(),
        }
    return _sessions[session_id]


def _get_sessions_sorted() -> list[tuple[str, dict]]:
    """Return (session_id, data) pairs, pinned first, then by creation desc."""
    items = list(_sessions.items())
    items.sort(key=lambda x: (not x[1]["pinned"], -x[1]["created"]))
    return items


def setup_layout(
    session_id: str,
    server_url: str,
    title: str = "Klea",
    subtitle: str = "",
    disclaimer: str = "",
    footer_text: str = 'Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
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

    :param session_id: Opaque session identifier persisted in
        ``app.storage.user``.
    :param server_url: Base URL of the backend API server.
    :param title: Bold application title in the header bar.
    :param subtitle: Optional smaller text shown next to *title*
        in the header.
    :param disclaimer: Optional text shown below the chat input.
    :param footer_text: HTML content for the footer bar.
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

    # --- Persistent dark mode ---
    dark = ui.dark_mode()
    if "dark_mode" not in app.storage.user:
        app.storage.user["dark_mode"] = False
    dark.bind_value(app.storage.user, "dark_mode")

    mini_state = True
    # Mutable containers so refreshable functions can pick up changes.
    _current_session = [session_id]
    toggle_icon_ref = [None]

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
        session = _sessions.get(_current_session[0])
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
                        _expanded.discard(i) if i in _expanded else _expanded.add(i),
                        _chat_messages.refresh(),
                    )[1],
                )
        ui.run_javascript(
            "document.querySelector('.chat-scroll-area')?.scrollTo(0, 999999)"
        )

    def _switch_session(sid: str) -> None:
        """Switch the active session without a page reload."""
        app.storage.user["session_id"] = sid
        _current_session[0] = sid
        logger.debug("Switched to session %s", sid)
        _chat_messages.refresh()
        _render_session_list.refresh()
        _inspector_panel.refresh()

    def _delete_session(sid: str) -> None:
        """Remove a session from the store.

        If the currently active session is deleted, the next available
        session becomes active (or a fresh one is created).
        """
        _sessions.pop(sid, None)
        if _current_session[0] == sid:
            remaining = _get_sessions_sorted()
            if remaining:
                _switch_session(remaining[0][0])
            else:
                new_sid = str(uuid.uuid4())
                _ensure_session(new_sid)
                _switch_session(new_sid)
        else:
            _render_session_list.refresh()

    def _toggle_pin(sid: str) -> None:
        """Flip the pinned flag for a session and refresh the list."""
        session = _ensure_session(sid)
        session["pinned"] = not session["pinned"]
        _render_session_list.refresh()

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

    def _new_session():
        """Create a fresh session and switch to it."""
        sid = str(uuid.uuid4())
        _ensure_session(sid)
        _switch_session(sid)

    def _rename_session(sid: str) -> None:
        """Open a dialog to rename a session."""
        session = _ensure_session(sid)
        dialog = ui.dialog()

        def _save():
            session["name"] = inp.value
            dialog.close()
            _render_session_list.refresh()

        with dialog:
            with ui.card():
                ui.label("Rename session").classes("text-lg font-bold")
                inp = ui.input(value=session["name"]).on("keydown.enter", _save)
                with ui.row().classes("w-full justify-end"):
                    ui.button("Cancel", on_click=dialog.close)
                    ui.button("Save", on_click=_save).props("unelevated color=primary")
        dialog.open()

    @ui.refreshable
    def _render_session_list():
        """Render the sorted list of session entries in the left drawer.

        Each entry shows the session name (bold for the active session),
        a tooltip with the creation timestamp, and a three-dot context
        menu for rename / pin / delete.  Refresh this to pick up newly
        created sessions without a page reload.
        """
        for sid, sdata in _get_sessions_sorted():
            is_current = sid == _current_session[0]
            with (
                ui.item(
                    on_click=lambda s=sid: _switch_session(s),
                )
                .props("dense")
                .classes("w-full")
                .on("dblclick", lambda s=sid: _rename_session(s))
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
                                on_click=lambda s=sid: _rename_session(s)
                            ):
                                with ui.item_section().props("avatar"):
                                    ui.icon("edit")
                                with ui.item_section():
                                    ui.label("Rename")
                            with ui.menu_item(on_click=lambda s=sid: _toggle_pin(s)):
                                with ui.item_section().props("avatar"):
                                    ui.icon("push_pin")
                                with ui.item_section():
                                    ui.label("Unpin" if sdata["pinned"] else "Pin")
                            with ui.menu_item(
                                on_click=lambda s=sid: _delete_session(s)
                            ):
                                with ui.item_section().props("avatar"):
                                    ui.icon("delete")
                                with ui.item_section():
                                    ui.label("Delete")

    # ---- Header ----
    with ui.header().classes("items-center"):
        if subtitle:
            ui.label(subtitle).classes("text-sm text-grey-4 mr-2")
        ui.label(title).classes("text-xl font-bold")
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
        with ui.item(on_click=_new_session).props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                ui.icon("add")
            with ui.item_section():
                ui.label("New Session")
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
                ui.label("Sessions").classes("text-sm font-bold")

        _render_session_list()

        ui.space()
        # Toggle button at the bottom of the drawer.
        with ui.item(on_click=_toggle_left_drawer).props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                toggle_icon_ref[0] = ui.icon("keyboard_double_arrow_right")
            with ui.item_section():
                ui.label("").classes("text-xs")

    def _toggle_inspector_entry(idx: int) -> None:
        """Toggle the expanded/collapsed state of an inspector entry."""
        session = _sessions.get(_current_session[0])
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
        ui.separator().classes("mb-2")

        @ui.refreshable
        def _inspector_panel() -> None:
            session = _sessions.get(_current_session[0])
            if not session:
                logger.debug(
                    "No session found for %s, inspector empty", _current_session[0]
                )
                return
            entries = session.get("inspector_entries", [])
            logger.debug(
                "Rendering inspector panel: %d entries for session %s",
                len(entries),
                _current_session[0],
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
                session = _ensure_session(_current_session[0])
                logger.debug(
                    "Clearing inspector entries for session %s", _current_session[0]
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
                        query, _current_session[0], server_url
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
                            _ensure_session(_current_session[0])["messages"].append(
                                (full_response, datetime.now().strftime("%X"), False)
                            )
                            _chat_messages.refresh()
                            break
                        elif t == "error":
                            pg_row.delete()
                            _ensure_session(_current_session[0])["messages"].append(
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
                    _ensure_session(_current_session[0])["messages"].append(
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
                _ensure_session(_current_session[0])["messages"].append(
                    (query, stamp, True)
                )
                _chat_messages.refresh()
                _render_session_list.refresh()

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

        if "session_id" not in app.storage.user:
            app.storage.user["session_id"] = str(uuid.uuid4())

        session_id = app.storage.user["session_id"]

        setup_layout(
            session_id=session_id,
            server_url=server_url,
            title=title,
            subtitle=subtitle,
            disclaimer=disclaimer,
            footer_text=footer_text,
        )

    ui.run(
        port=7860,
        host="0.0.0.0",
        title=title,
        show=False,
        reload=debug,
        storage_secret="klea-nicegui-secret-change-me",
    )
