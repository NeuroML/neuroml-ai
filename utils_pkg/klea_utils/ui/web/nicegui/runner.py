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
from .client import (
    create_chat_on_server,
    delete_chat_on_server,
    hydrate_chats,
    rename_chat_on_server,
)
from .state import chats, ensure_chat, get_chats_sorted
from .widgets import ChatBubble

logger = setup_logger(__name__)


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
    # Status pane styling — uses disclosure triangles (same pattern as inspector)
    ui.add_css(
        ".status-entry > summary { list-style: none; display: flex; align-items: center; gap: 0.25rem; }"
    )
    ui.add_css(
        ".status-entry > summary::before { content: '\\25B6'; font-size: 0.65rem; margin-right: 0.35rem; transition: transform 0.15s; }"
    )
    ui.add_css(".status-entry[open] > summary::before { content: '\\25BC'; }")
    ui.add_css(
        ".status-details summary { list-style: none; display: flex; align-items: center; gap: 0.25rem; }"
    )
    ui.add_css(
        ".status-details summary::before { content: '\\25B6'; font-size: 0.6rem; margin-right: 0.35rem; }"
    )
    ui.add_css(".status-details[open] summary::before { content: '\\25BC'; }")
    ui.add_css(
        ".status-details code { white-space: pre-wrap !important; word-break: break-all !important; }"
    )
    ui.add_css(
        ".status-entry .md-div { overflow: hidden !important; height: auto !important; }"
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

    def _render_chat_area() -> None:
        """Rebuild the scroll-area content (welcome or messages).

        Uses explicit clear+rebuild instead of ``@ui.refreshable``
        to avoid issues with the welcome-to-empty-chat transition.
        """
        current = _current_chat_id[0]
        logger.debug(
            "current=%s msgs=%d",
            current,
            len(chats.get(f"{user_id}:{current}", {}).get("messages", []))
            if current
            else 0,
        )
        _chat_area.clear()
        with _chat_area:
            if not current:
                with (
                    ui.column()
                    .classes("w-full h-full items-center justify-center gap-4")
                    .style("flex: 1; display: flex;")
                ):
                    ui.label("Start a conversation").classes("text-xl text-grey-5")
                    ui.label("Type your message below to begin").classes(
                        "text-sm text-grey-5"
                    )
            else:
                current_chat = chats.get(f"{user_id}:{current}")
                msgs = current_chat["messages"] if current_chat else []
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
                            (
                                _expanded.discard(i)
                                if i in _expanded
                                else _expanded.add(i)
                            )
                            or _render_chat_area()
                        ),
                    )
        _scroll_to_bottom()

    def _scroll_to_bottom() -> None:
        """Scroll chat area to the bottom.

        Uses Quasar's ``setScrollPosition`` (via NiceGUI's
        ``scroll_to(pixels=99999)``) instead of raw JavaScript because
        NiceGUI batches UI updates and JS into the same WebSocket packet
        — by the time a ``setTimeout`` or ``requestAnimationFrame``
        callback fires the new DOM may not be laid out yet, so
        ``scrollTop = scrollHeight`` or ``scrollIntoView`` land at the
        wrong position.

        ``setScrollPosition`` is Quasar's own scroll API on
        ``QScrollArea``; it coordinates with its internal layout cycle
        so the scroll lands correctly after the content updates are
        painted.  The large pixel value is safe — Quasar clamps it to
        the actual scrollable extent.
        """
        logger.debug("attempting scroll for chat=%s", _current_chat_id[0])
        _scroll_area.scroll_to(pixels=99999)

    def _switch_chat(chat_id: str) -> None:
        """Switch the active chat without a page reload."""
        app.storage.user["chat_id"] = chat_id
        _current_chat_id[0] = chat_id
        logger.debug("chat_id=%s user_id=%s", chat_id, user_id)
        _render_chat_area()
        _render_chat_list.refresh()
        _status_pane.refresh()

    def _delete_chat(chat_id: str) -> None:
        """Remove a chat session from the store and server.

        If the currently active chat session is deleted, the next available
        chat session becomes active (or a new one is created).
        """
        logger.debug(
            "deleting chat_id=%s user_id=%s (current=%s)",
            chat_id,
            user_id,
            _current_chat_id[0],
        )
        background_tasks.create(delete_chat_on_server(server_url, user_id, chat_id))
        chats.pop(f"{user_id}:{chat_id}", None)
        if _current_chat_id[0] == chat_id:
            remaining = get_chats_sorted(user_id)
            if remaining:
                _switch_chat(remaining[0][0])
            else:
                _current_chat_id[0] = ""
                app.storage.user["chat_id"] = ""
                _render_chat_area()
                _render_chat_list.refresh()
                _status_pane.refresh()
        else:
            _render_chat_list.refresh()

    def _toggle_pin(chat_id: str) -> None:
        """Flip the pinned flag for a chat and refresh the list."""
        current_chat = ensure_chat(user_id, chat_id)
        current_chat["pinned"] = not current_chat["pinned"]
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

    def _new_chat():
        """Create a new chat and switch to it."""
        chat_id = coolname.generate_slug(2)
        logger.debug("creating chat_id=%s", chat_id)
        ensure_chat(user_id, chat_id)
        _switch_chat(chat_id)
        background_tasks.create(create_chat_on_server(server_url, user_id, chat_id))

    def _rename_chat(chat_id: str) -> None:
        """Open a dialog to rename a chat and persist on server."""
        chat = ensure_chat(user_id, chat_id)
        dialog = ui.dialog()

        async def _save():
            chat["name"] = inp.value
            dialog.close()
            await rename_chat_on_server(server_url, user_id, chat_id, inp.value)
            _render_chat_list.refresh()

        with dialog, ui.card():
            ui.label("Rename chat").classes("text-lg font-bold")
            inp = ui.input(value=chat["name"]).on("keydown.enter", _save)
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close)
                ui.button("Save", on_click=_save).props("unelevated color=primary")
        dialog.open()

    @ui.refreshable
    def _render_chat_list():
        """Render the sorted list of chat sessions in the left drawer.

        Each entry shows the chat name (bold for the active chat),
        a tooltip with the creation timestamp, and a three-dot context
        menu for rename / pin / delete.  Refresh this to pick up newly
        created chat sessions without a page reload.
        """
        # Underscore suffix avoids shadowing setup_layout's chat_id parameter
        for chat_id_, sdata in get_chats_sorted(user_id):
            is_current = chat_id_ == _current_chat_id[0]
            with (
                ui.item(
                    on_click=lambda s=chat_id_: _switch_chat(s),
                )
                .props("dense")
                .classes("w-full")
                .on("dblclick", lambda s=chat_id_: _rename_chat(s))
            ):
                with ui.item_section().props("avatar"):
                    ui.icon("push_pin" if sdata["pinned"] else "history")
                with ui.item_section():
                    label_cls = "text-xs font-bold" if is_current else "text-xs"
                    ui.label(sdata["name"]).classes(label_cls)
                    ui.tooltip(
                        "Created: "
                        + datetime.fromtimestamp(sdata["created"])
                        .astimezone()
                        .strftime("%a %d %b %Y at %X")
                    )
                # Three-dot context menu (right-aligned).
                with (
                    ui.item_section().props("side"),
                    ui.button(icon="more_vert")
                    .props("flat dense round")
                    .on("click.stop", lambda: None),
                    ui.menu(),
                ):
                    with ui.menu_item(on_click=lambda s=chat_id_: _rename_chat(s)):
                        with ui.item_section().props("avatar"):
                            ui.icon("edit")
                        with ui.item_section():
                            ui.label("Rename")
                    with ui.menu_item(on_click=lambda s=chat_id_: _toggle_pin(s)):
                        with ui.item_section().props("avatar"):
                            ui.icon("push_pin")
                        with ui.item_section():
                            ui.label("Unpin" if sdata["pinned"] else "Pin")
                    with ui.menu_item(on_click=lambda s=chat_id_: _delete_chat(s)):
                        with ui.item_section().props("avatar"):
                            ui.icon("delete")
                        with ui.item_section():
                            ui.label("Delete")

    def _delete_all_data():
        """Show a confirmation dialog before deleting the user session."""
        logger.debug("opening delete-user-session dialog for user_id=%s", user_id)
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label("Delete user session?").classes("text-lg font-bold")
            ui.label(
                "This will permanently delete all your chats, messages, and "
                "checkpoints from the server. This cannot be undone."
            ).classes("text-sm")
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=dialog.close)
                ui.button("Delete", on_click=lambda: _confirm_delete_all(dialog)).props(
                    "unelevated color=negative"
                )
        dialog.open()

    async def _confirm_delete_all(dialog: ui.dialog):
        """DELETE all server data, reset in-memory state, and generate a new user ID."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.delete(f"{server_url}/chat/{user_id}")
                if resp.status_code == 200:
                    logger.debug(
                        "delete_user_session succeeded for user_id=%s", user_id
                    )
        except Exception as e:
            logger.warning("Failed to delete user session: %s", e)
            dialog.close()
            return

        logger.debug("clearing in-memory chats for user_id=%s", user_id)
        chat_key_prefix = f"{user_id}:"
        for key in list(chats.keys()):
            if key.startswith(chat_key_prefix):
                chats.pop(key, None)

        _current_chat_id[0] = ""
        app.storage.user["chat_id"] = ""
        old_id = user_id
        app.storage.user["user_id"] = str(uuid.uuid4())
        logger.debug(
            "reset local state: new user_id=%s (was %s)",
            app.storage.user["user_id"],
            old_id,
        )
        _render_chat_area()
        _render_chat_list.refresh()
        _status_pane.refresh()
        dialog.close()

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
        .props("width=320")
        .classes("overflow-x-hidden p-2") as left_drawer
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

        # Session list header (no icon — plain text signals a section heading)
        with ui.item().props("dense").classes("w-full"), ui.item_section():
            ui.label("Chats").classes("text-sm font-bold")

        _render_chat_list()

        ui.space()

        with ui.item(on_click=_delete_all_data).props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                ui.icon("delete")
                ui.tooltip("Delete all data for this user session")
            with ui.item_section():
                ui.label("Delete user session").classes("text-xs")

        # Toggle button at the bottom of the drawer.
        with ui.item(on_click=_toggle_left_drawer).props("dense").classes("w-full"):
            with ui.item_section().props("avatar"):
                toggle_icon_ref[0] = ui.icon("keyboard_double_arrow_right")
                ui.tooltip("Expand or collapse the sidebar")
            with ui.item_section():
                ui.label("").classes("text-xs")

    def _toggle_inspector_entry(idx: int) -> None:
        """Toggle the expanded/collapsed state of an inspector entry for the active chat."""
        current_chat = chats.get(f"{user_id}:{_current_chat_id[0]}")
        if not current_chat:
            return
        expanded = current_chat.setdefault("inspector_expanded", set())
        if idx in expanded:
            expanded.discard(idx)
        else:
            expanded.add(idx)

    # ---- Right drawer (status pane, hidden by default) ----
    with (
        ui.right_drawer(value=False)
        .props("width=420")
        .classes("overflow-y-auto") as right_drawer
    ):
        ui.label("Status").classes("text-sm font-bold mb-0")
        ui.separator().classes("my-0.5")

        if model_info:
            with ui.row().classes("items-center no-wrap my-0.5"):
                ui.icon("settings").classes("text-sm")
                with ui.label(model_info).classes("text-xs truncate"):
                    ui.tooltip("Current models per role")
            ui.separator().classes("my-0.5")

        @ui.refreshable
        def _status_pane() -> None:
            """Render state sections for the active chat in the right drawer."""
            current_chat = chats.get(f"{user_id}:{_current_chat_id[0]}")
            if not current_chat:
                if _current_chat_id[0]:
                    logger.debug(
                        "No chat found for %s, status pane empty", _current_chat_id[0]
                    )
                return
            sections = current_chat.get("state_sections", {})
            if not sections:
                ui.label("State updates will appear here").classes(
                    "text-sm text-gray-500"
                )
                return
            for idx, (node_label, section) in enumerate(sections.items()):
                with (
                    ui.element("details")
                    .props("open")
                    .classes("status-entry mb-2 w-full")
                ):
                    with (
                        ui.element("summary").classes(
                            "text-xs font-bold cursor-pointer w-full"
                        ),
                        ui.row().classes("w-full flex-nowrap items-center"),
                    ):
                        ui.label(section.get("heading", node_label))
                    display = section.get("display", "")
                    if display:
                        ui.markdown(display).classes("text-xs w-full")
                    summary = section.get("summary", "")
                    if summary and not display:
                        ui.label(summary).classes("text-xs text-grey-6 mb-1 w-full")
                    details = section.get("details", {})
                    if details:
                        with ui.element("details").classes(
                            "status-details text-xs text-grey-5 cursor-pointer w-full"
                        ):
                            with ui.element("summary").classes("text-xs w-full"):
                                ui.label("View details")
                            ui.code(
                                json.dumps(details, indent=2), language="json"
                            ).classes("text-xs")

        _status_pane()

        # --- Debug section (commented out — location TBD) ---
        # @ui.refreshable
        # def _debug_panel() -> None:
        #     """Render debug/info entries (old inspector) — unused pending frontend design."""
        #     current_chat = chats.get(f"{user_id}:{_current_chat_id[0]}")
        #     if not current_chat:
        #         return
        #     entries = current_chat.get("inspector_entries", [])
        #     if not entries:
        #         return
        #     ui.separator().classes("my-1")
        #     ui.label("Debug").classes("text-xs font-bold mb-0")
        #     for idx, entry in enumerate(entries):
        #         heading = entry.get("heading", "")
        #         timing = entry.get("timing_seconds", None)
        #         expanded = idx in current_chat["inspector_expanded"]
        #         with (
        #             ui.element("details")
        #             .props("open" if expanded else "")
        #             .classes("inspector-entry mb-2 w-full")
        #         ):
        #             with (
        #                 ui.element("summary")
        #                 .classes("text-xs font-bold cursor-pointer w-full")
        #                 .on("click", lambda i=idx: _toggle_inspector_entry(i))
        #             ):
        #                 with ui.row().classes("w-full flex-nowrap items-center"):
        #                     ui.label(heading)
        #                     if timing:
        #                         ui.label(f"({timing:.1f}s)").classes(
        #                             "text-xs text-grey-5"
        #                         )
        #             ui.label(entry.get("summary", "")).classes(
        #                 "text-xs text-grey-6 mb-1 w-full"
        #             )
        #             details = entry.get("details", {})
        #             if details:
        #                 with ui.element("details").classes(
        #                     "inspector-details text-xs text-grey-5 cursor-pointer w-full"
        #                 ):
        #                     with ui.element("summary").classes("text-xs w-full"):
        #                         ui.label("View details")
        #                     ui.code(
        #                         json.dumps(details, indent=2), language="json"
        #                     ).classes("text-xs")
        #
        # _debug_panel()

    # ---- Center: chat messages + input (pinned to bottom) ----
    with (
        ui.column()
        .classes("w-full px-8")
        .style("flex: 1; min-height: 0; display: flex; flex-direction: column;")
    ):
        with ui.scroll_area().classes("w-full grow chat-scroll-area") as _scroll_area:
            _chat_area = ui.column().classes("w-full")
            _render_chat_area()
            _stream_container = ui.column().classes("w-full")

        with ui.row().classes("w-full no-wrap items-end py-4"):
            text = (
                ui.textarea(placeholder="Start a conversation")
                .props("rounded outlined input-class=mx-3 autogrow")
                .classes("flex-grow")
            )

            async def _do_stream(query: str, chat_id: str) -> None:
                """Stream the RAG pipeline progress, store the final answer in the chat dict, and update the UI."""
                current_chat = ensure_chat(user_id, chat_id)
                logger.debug("Clearing inspector entries for chat %s", chat_id)
                current_chat["inspector_entries"] = []
                current_chat["inspector_expanded"] = set()
                current_chat["state_sections"] = {}
                _status_pane.refresh()
                with _stream_container:
                    pg_row = ui.row().classes("w-full items-center gap-2 p-2")
                    with pg_row:
                        ui.spinner(type="dots").classes("w-4 h-4")
                        pg_label = ui.label("").classes("text-xs text-grey-5 italic")

                full_response = ""
                try:
                    async for event in stream_events(
                        query, chat_id, server_url, user_id=user_id
                    ):
                        t = event.get("type", "?")
                        if t == "progress":
                            pg_label.set_text(f"{event.get('node', '')}")
                        elif t == "debug":
                            data = event.get("data", {})
                            current_chat["inspector_entries"].append(
                                {
                                    "type": t,
                                    "node": event.get("node", ""),
                                    "heading": data.get("heading", ""),
                                    "summary": data.get("summary", ""),
                                    "details": data.get("details", {}),
                                    "timing_seconds": data.get("timing_seconds", None),
                                }
                            )
                        elif t == "state":
                            data = event.get("data", {})
                            node = event.get("node", "")
                            current_chat["state_sections"][node] = {
                                "heading": data.get("heading", ""),
                                "display": data.get("display", ""),
                                "summary": data.get("summary", ""),
                                "details": data.get("details", {}),
                            }
                            _status_pane.refresh()
                        elif t == "complete":
                            pg_row.delete()
                            full_response = event.get("message_for_user", full_response)
                            logger.debug(
                                "chat=%s response_len=%d",
                                chat_id,
                                len(full_response),
                            )
                            ensure_chat(user_id, chat_id)["messages"].append(
                                (
                                    full_response,
                                    datetime.now().astimezone().strftime("%X"),
                                    False,
                                )
                            )
                            _render_chat_area()
                            break
                        elif t == "error":
                            pg_row.delete()
                            ensure_chat(user_id, chat_id)["messages"].append(
                                (
                                    f"Error: {event.get('message', 'Unknown error')}",
                                    datetime.now().astimezone().strftime("%X"),
                                    False,
                                )
                            )
                            _render_chat_area()
                            break
                except httpx.RequestError as e:
                    pg_row.delete()
                    ensure_chat(user_id, chat_id)["messages"].append(
                        (
                            f"Connection error: {e}",
                            datetime.now().astimezone().strftime("%X"),
                            False,
                        )
                    )
                    _render_chat_area()

            def send() -> None:
                """Append the current input text as a user message. Creates a new chat if none is active."""
                if not text.value.strip():
                    return
                stamp = datetime.now().astimezone().strftime("%X")
                query = text.value
                text.value = ""

                current = _current_chat_id[0]
                logger.debug("current=%s query_len=%d", current, len(query))
                if not current:
                    current = coolname.generate_slug(2)
                    _current_chat_id[0] = current
                    app.storage.user["chat_id"] = current
                    ensure_chat(user_id, current)
                    background_tasks.create(
                        create_chat_on_server(server_url, user_id, current)
                    )
                    _render_chat_list.refresh()

                ensure_chat(user_id, current)["messages"].append((query, stamp, True))
                _render_chat_area()
                _render_chat_list.refresh()

                background_tasks.create(_do_stream(query, current))

            with (
                text.add_slot("append"),
                ui.button(icon="send", on_click=send).props(
                    "flat dense round color=primary"
                ),
            ):
                ui.tooltip("Enter to send, Shift+Enter for newline")

        # Plain Enter sends the message and prevents the default newline
        def handle_enter(e: GenericEventArguments):
            """Send on Enter, insert newline on Shift+Enter."""
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

        ``ui.run_javascript`` (used inside ``_render_chat_area``) only
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
            logger.debug("NEW user_id=%s", app.storage.user["user_id"])
        else:
            logger.debug("EXISTING user_id=%s", app.storage.user["user_id"])

        user_id = app.storage.user["user_id"]

        chat_id = ""

        logger.debug("user_id=%s chat_id=%s", user_id, chat_id)
        logger.debug("before hydrate: chats keys=%s", list(chats.keys()))
        await hydrate_chats(server_url, user_id)
        logger.debug("after hydrate: chats keys=%s", list(chats.keys()))

        model_info = ""
        if chat_id:
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
