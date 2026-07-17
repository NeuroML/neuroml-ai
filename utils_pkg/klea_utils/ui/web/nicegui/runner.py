#!/usr/bin/env python3
"""
NiceGUI runner for Klea web interfaces.

This module implements the main UI logic for the NiceGUI web interface,
including the 3-column layout, chat functionality, and inspector panel.

File: klea_utils/ui/web/nicegui/runner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import uuid
from datetime import datetime

from nicegui import app, ui
from nicegui.events import GenericEventArguments

# Avatar for system / bot messages (uses static robohash URL).
SYSTEM_AVATAR = "https://robohash.org/klea-system?bgset=bg1"

# Per-session message store.
# Keyed by session_id so that switching sessions preserves history.
# Tuple format: (user_id, avatar_url, text, timestamp).
_messages: dict[str, list[tuple[str, str, str, str]]] = {}


def _ensure_messages(
    session_id: str,
) -> list[tuple[str, str, str, str]]:
    """Return the message list for *session_id*, creating it if missing."""
    if session_id not in _messages:
        _messages[session_id] = []
    return _messages[session_id]


@ui.refreshable
def _chat_messages(session_id: str, own_id: str, avatar_url: str) -> None:
    """Render the message list for the given session.

    This function is decorated with ``@ui.refreshable`` so that
    ``_chat_messages.refresh()`` can be called after a new message
    is appended to the store.  The entire list is rebuilt on every
    refresh, which means the scroll position is lost -- the
    ``run_javascript`` call at the end scrolls back to the bottom.

    :param session_id: Session whose messages to display.
    :param own_id: Current user's UUID (compared against each
        message's *user_id* to determine ``sent`` styling).
    :param avatar_url: Avatar URL for the current user's messages.
    """
    messages = _ensure_messages(session_id)
    if messages:
        for user_id, msg_avatar, text, stamp in messages:
            is_sent = own_id == user_id
            ui.chat_message(
                text=text,
                stamp=stamp,
                avatar=msg_avatar,
                sent=is_sent,
            ).classes("w-full" if not is_sent else "w-11/12")
    else:
        ui.chat_message(
            "Welcome! Type a message below to start chatting.",
            stamp="now",
            avatar=SYSTEM_AVATAR,
        ).classes("w-full")
    # Scroll to bottom after rendering so new messages are visible.
    ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")


def setup_layout(
    session_id: str,
    own_id: str,
    avatar_url: str,
    title: str = "Klea",
    subtitle: str = "",
    disclaimer: str = "",
    footer_text: str = 'Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
) -> None:
    """Build the full page UI: header, drawers, chat area, and footer.

    Layout (left to right):

        [left_drawer | center_column | right_drawer]

    The left drawer uses Quasar's *mini* mode to provide a
    ChatGPT-style rail that shows only icons when collapsed and
    full text when expanded.

    :param session_id: Opaque session identifier persisted in
        ``app.storage.user``.
    :param own_id: Per-page-instance UUID used to distinguish
        sent vs. received messages.
    :param avatar_url: Robohash URL for the current user's avatar.
    :param title: Bold application title in the header bar.
    :param subtitle: Optional smaller text shown next to *title*
        in the header.
    :param disclaimer: Optional text shown below the chat input.
    :param footer_text: HTML content for the footer bar.
    """
    # --- Global CSS overrides ---
    # Remove bubble background from system/bot messages so they appear
    # inline with the page background.  User messages keep their default
    # Quasar styling for visual distinction.
    ui.add_css(
        ".q-message-text--received { background: none !important; box-shadow: none !important; }"
    )

    # --- Persistent dark mode ---
    dark = ui.dark_mode()
    if "dark_mode" not in app.storage.user:
        app.storage.user["dark_mode"] = False
    dark.bind_value(app.storage.user, "dark_mode")

    mini_state = True
    # Mutable container to capture the toggle-icon element reference.
    # The icon is created inside the drawer ``with`` block (after this
    # variable is defined), so we store it in a single-element list to
    # allow assignment from the enclosing scope.
    toggle_icon_ref = [None]

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
        .classes("w-80 overflow-x-hidden") as left_drawer
    ):
        # Items use QItem + QItemSection(avatar) so that Quasar
        # automatically hides the label when the drawer is in mini mode.
        with ui.item(on_click=lambda: print("New session clicked")):
            with ui.item_section().props("avatar"):
                ui.icon("add")
            with ui.item_section():
                ui.label("New Session")
        with ui.item(on_click=lambda: right_drawer.toggle()):
            with ui.item_section().props("avatar"):
                ui.icon("info")
            with ui.item_section():
                ui.label("Inspector")
        ui.separator()
        with ui.item().props("dense"):
            with ui.item_section().props("avatar"):
                ui.icon("chat")
            with ui.item_section():
                ui.label("Sessions").classes("text-lg font-bold")
        with ui.item().props("dense"):
            with ui.item_section().props("avatar"):
                ui.icon("history")
            with ui.item_section():
                ui.label(f"Session ID: {session_id[:8]}...").classes(
                    "text-sm text-gray-500"
                )
        ui.space()
        # Toggle button at the bottom of the drawer.
        with ui.item(on_click=_toggle_left_drawer):
            with ui.item_section().props("avatar"):
                toggle_icon_ref[0] = ui.icon("keyboard_double_arrow_right")

    # ---- Right drawer (inspector, hidden by default) ----
    with ui.right_drawer(value=False).classes("w-80") as right_drawer:
        ui.label("Inspector").classes("text-lg font-bold mb-4")
        ui.separator()
        ui.label("Info and debug events will appear here").classes(
            "text-sm text-gray-500"
        )

    # ---- Center: chat messages + input ----
    with ui.column().classes("w-full h-full px-48"):
        # Scrollable message history (grows to fill the column).
        with ui.scroll_area().classes("w-full grow"):
            _chat_messages(session_id, own_id, avatar_url)

        # Message input row (pinned to the bottom of the center column).
        with ui.row().classes("w-full no-wrap items-center py-4"):
            text = (
                ui.textarea(placeholder="Type your message...")
                .props("rounded outlined input-class=mx-3 autogrow")
                .classes("flex-grow")
            )

            def send() -> None:
                """Append the current input text as a user message."""
                if not text.value.strip():
                    return
                stamp = datetime.now().strftime("%X")
                _ensure_messages(session_id).append(
                    (own_id, avatar_url, text.value, stamp)
                )
                text.value = ""
                _chat_messages.refresh()

            ui.button("Send", on_click=send).props("unelevated color=primary")

        # Plain Enter sends the message and prevents the default newline
        # insertion.  The .exact modifier ensures this fires only when NO
        # modifier keys (Shift, Ctrl, Alt, Meta) are pressed, so
        # Shift+Enter / Ctrl+Enter fall through to the default textarea
        # behaviour (newline insertion).
        def handle_enter(e: GenericEventArguments):
            # e.args contains the client-side JavaScript event properties
            if e.args.get("shiftKey"):
                # The user pressed Shift + Enter.
                # We manually append a newline character because .prevent stopped it.
                text.value += "\n"
            else:
                # The user pressed Enter alone.
                send()

        text.on("keydown.enter.exact.prevent", handle_enter)

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
        (e.g. ``http://127.0.0.1:8005``).  Currently unused in the
        UI itself, but reserved for future SSE streaming.
    :param subtitle: Optional smaller text shown next to *title*
        in the header.
    :param disclaimer: Optional text shown below the chat input.
    :param footer_text: HTML content for the footer bar.
    :param debug: When ``True``, enable NiceGUI's file-watch hot
        reload (``reload=True``).  Set to ``False`` in production.
    """
    # (server_url is reserved for future SSE streaming)

    @ui.page("/")
    async def main_page():
        """Build the main page after ensuring the WebSocket is connected.

        ``ui.run_javascript`` (used inside ``_chat_messages``) only
        works after the client has connected, so we await
        ``ui.context.client.connected()`` first.
        """
        await ui.context.client.connected()

        if "session_id" not in app.storage.user:
            app.storage.user["session_id"] = str(uuid.uuid4())

        session_id = app.storage.user["session_id"]
        own_id = str(uuid.uuid4())
        # Robohash generates a unique avatar from the UUID.
        avatar_url = f"https://robohash.org/{own_id}?bgset=bg2"

        setup_layout(
            session_id=session_id,
            own_id=own_id,
            avatar_url=avatar_url,
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
