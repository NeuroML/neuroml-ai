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

_messages: dict[str, list[tuple[str, str, str, str]]] = {}


def _ensure_messages(
    session_id: str,
) -> list[tuple[str, str, str, str]]:
    if session_id not in _messages:
        _messages[session_id] = []
    return _messages[session_id]


@ui.refreshable
def _chat_messages(session_id: str, own_id: str, avatar_url: str) -> None:
    messages = _ensure_messages(session_id)
    if messages:
        for user_id, msg_avatar, text, stamp in messages:
            ui.chat_message(
                text=text,
                stamp=stamp,
                avatar=msg_avatar,
                sent=own_id == user_id,
            ).classes("w-full")
    else:
        ui.chat_message(
            "Welcome! Type a message below to start chatting.",
            name="System",
            stamp="now",
        ).classes("w-full")
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
    dark = ui.dark_mode()
    if "dark_mode" not in app.storage.user:
        app.storage.user["dark_mode"] = False
    dark.bind_value(app.storage.user, "dark_mode")

    mini_state = True
    toggle_icon_ref = [None]

    def _toggle_left_drawer():
        nonlocal mini_state
        mini_state = not mini_state
        if mini_state:
            left_drawer.props("mini")
            toggle_icon_ref[0].name = "keyboard_double_arrow_right"
        else:
            left_drawer.props(remove="mini")
            toggle_icon_ref[0].name = "keyboard_double_arrow_left"

    with ui.header().classes("items-center"):
        if subtitle:
            ui.label(subtitle).classes("text-sm text-grey-4 mr-2")
        ui.label(title).classes("text-xl font-bold")
        ui.space()

        def _toggle_dark():
            dark.value = not dark.value

        ui.button(icon="dark_mode", on_click=_toggle_dark).props(
            "flat color=white round"
        )

    with (
        ui.left_drawer(value=True)
        .props("mini")
        .classes("w-80 overflow-x-hidden") as left_drawer
    ):
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
        with ui.item(on_click=_toggle_left_drawer):
            with ui.item_section().props("avatar"):
                toggle_icon_ref[0] = ui.icon("keyboard_double_arrow_right")

    with ui.right_drawer(value=False).classes("w-80") as right_drawer:
        ui.label("Inspector").classes("text-lg font-bold mb-4")
        ui.separator()
        ui.label("Info and debug events will appear here").classes(
            "text-sm text-gray-500"
        )

    with ui.column().classes("w-full h-full px-48"):
        with ui.scroll_area().classes("w-full grow"):
            _chat_messages(session_id, own_id, avatar_url)

        with ui.row().classes("w-full no-wrap items-center py-4"):
            text = (
                ui.input(placeholder="Type your message...")
                .props("rounded outlined input-class=mx-3")
                .classes("flex-grow")
            )

            def send() -> None:
                if not text.value.strip():
                    return
                stamp = datetime.now().strftime("%X")
                _ensure_messages(session_id).append(
                    (own_id, avatar_url, text.value, stamp)
                )
                text.value = ""
                _chat_messages.refresh()

            text.on("keydown.enter", send)
            ui.button("Send", on_click=send).props("unelevated color=primary")

        if disclaimer:
            ui.label(disclaimer).classes("text-xs text-grey-5 pb-2 w-full text-center")

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
    @ui.page("/")
    async def main_page():
        await ui.context.client.connected()

        if "session_id" not in app.storage.user:
            app.storage.user["session_id"] = str(uuid.uuid4())

        session_id = app.storage.user["session_id"]
        own_id = str(uuid.uuid4())
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
