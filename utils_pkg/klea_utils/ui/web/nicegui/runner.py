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
from typing import Any

import coolname
import httpx
from nicegui import app, background_tasks, core, ui
from nicegui.events import GenericEventArguments
from nicegui.storage import request_contextvar

from klea_utils.api.sse import (
    fetch_active_models,
    stream_events,
)
from klea_utils.api.utils import check_api_is_ready
from klea_utils.llm import parse_model_name
from klea_utils.ui.linkify import linkify_md

from .client import (
    clear_model_override,
    create_chat_on_server,
    delete_chat_on_server,
    hydrate_chats,
    rename_chat_on_server,
    set_model_override,
)
from .state import chats, ensure_chat, get_chats_sorted
from .widgets import ChatBubble

logger = logging.getLogger(__name__)


async def _ensure_user_storage():
    """Return ``app.storage.user`` recreating it if stale.

    When the ``.nicegui/storage-user-*.json`` file is missing but the
    browser still sends the old session cookie, ``app.storage.user``
    raises ``AssertionError``. We warn and recreate the backing
    ``FilePersistentDict`` for that ``session_id`` so the page can
    continue with a fresh ``user_id`` instead of 500.
    """
    try:
        return app.storage.user
    except AssertionError as e:
        request = request_contextvar.get()
        session_id = request.session.get("id", "unknown") if request else "unknown"
        logger.warning(
            f"stale nicegui session {session_id = } missing storage, recreating: {e}"
        )
        if request is not None:
            await core.app.storage._create_user_storage(session_id)
        return app.storage.user


def _user_storage_or_none():
    """Return ``app.storage.user`` or ``None`` if stale (no await)."""
    try:
        return app.storage.user
    except AssertionError as e:
        request = request_contextvar.get()
        session_id = request.session.get("id", "unknown") if request else "unknown"
        logger.warning(f"stale nicegui session {session_id = } at storage access: {e}")
        return None


def _safe_set_user(key: str, value: Any) -> None:
    """Set ``app.storage.user[key]`` if storage is available, else warn."""
    store = _user_storage_or_none()
    if store is not None:
        store[key] = value
    else:
        logger.warning(f"skipping persistent set {key}={value!r} due to stale storage")


def setup_layout(
    chat_id: str,
    server_url: str,
    user_id: str = "",
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

    :param chat_id: Chat conversation identifier.
    :param server_url: Base URL of the backend API server.
    :param user_id: Opaque persistent user identifier.
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
    # GitHub-style alerts (rendered from ``> [!WARNING]`` etc. by the markdown2
    # 'alerts' extra) used for the fallback / best-effort warnings in bubbles.
    # No default style defined by nicegui for these extras
    ui.add_css(
        ".nicegui-markdown div.alert { "
        "padding: 0.4rem 0.75rem; "
        "border-left: 4px solid #d29922; "
        "border-radius: 0.25rem; "
        "background: rgba(210, 153, 34, 0.12); "
        "margin: 0.5rem 0; "
        "}"
    )
    ui.add_css(
        ".nicegui-markdown div.alert em { font-style: normal; font-weight: 600; }"
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
        ".q-tooltip { max-width: 350px !important; overflow: visible !important; white-space: nowrap !important; padding: 4px 8px !important; }"
    )
    ui.add_css(
        ".model-tooltip { white-space: pre-wrap !important; max-width: none !important; }"
    )
    # The chat input must stay pinned to the bottom of the central pane.
    # Quasar wraps each tab panel in a `.q-panel` container between
    # `.q-tab-panels` and `.q-tab-panel`; it must also flex-fill so the chat
    # panel's scroll area can grow and push the input row down.
    ui.add_css(
        ".center-tab-panels > .q-panel { flex: 1; min-height: 0; display: flex; flex-direction: column; }"
    )
    # Status pane styling  ---  uses disclosure triangles (same pattern as inspector)
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
        ".status-entry .nicegui-markdown { overflow: hidden !important; height: auto !important; overflow-wrap: break-word !important; word-break: break-word !important; }"
    )
    # Keep heading sizes in status pane small so they don't compete with
    # the section summary label.  Nodes can use # freely without worrying
    # about hierarchy.
    ui.add_css(
        ".status-entry .nicegui-markdown h1, .status-entry .nicegui-markdown h2, "
        ".status-entry .nicegui-markdown h3, .status-entry .nicegui-markdown h4, "
        ".status-entry .nicegui-markdown h5, .status-entry .nicegui-markdown h6 { "
        "font-size: 0.7rem !important; "
        "font-weight: 600; "
        "margin: 0.15rem 0; "
        "line-height: 1.2; }"
    )
    # Reduce default padding on lists in the status pane (40px is too wide
    # at text-xs scale).
    ui.add_css(
        ".status-entry .nicegui-markdown ul, "
        ".status-entry .nicegui-markdown ol { "
        "padding-inline-start: 1rem; }"
    )

    # --- Persistent dark mode ---
    dark = ui.dark_mode()
    _store = _user_storage_or_none()
    if _store is not None:
        if "dark_mode" not in _store:
            _store["dark_mode"] = False
        dark.bind_value(_store, "dark_mode")

    mini_state = True
    # Mutable containers so refreshable functions can pick up changes.
    _current_chat_id = [chat_id]
    toggle_icon_ref: list = [None]
    _is_streaming: list = [False]
    # Reference to the central tab panel widget, filled in once the layout
    # is built (handlers above it may run later, at click time).
    _center_panels: list = [None]

    # ``user_id`` (a setup_layout parameter) is captured by reference in
    # every handler below.  Handlers therefore see a rebind made elsewhere
    # in this scope -- _confirm_delete_all uses ``nonlocal user_id`` to
    # switch the whole page session to a fresh identity after a delete.
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
                        text=linkify_md(text),
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
         ---  by the time a ``setTimeout`` or ``requestAnimationFrame``
        callback fires the new DOM may not be laid out yet, so
        ``scrollTop = scrollHeight`` or ``scrollIntoView`` land at the
        wrong position.

        ``setScrollPosition`` is Quasar's own scroll API on
        ``QScrollArea``; it coordinates with its internal layout cycle
        so the scroll lands correctly after the content updates are
        painted.  The large pixel value is safe  ---  Quasar clamps it to
        the actual scrollable extent.
        """
        logger.debug("attempting scroll for chat=%s", _current_chat_id[0])
        _scroll_area.scroll_to(pixels=99999)

    async def _fetch_model_info() -> None:
        """Fetch active model config for the current chat and update the status pane."""
        chat_id = _current_chat_id[0]
        if not chat_id:
            logger.debug("fetch model info: no active chat (user=%s)", user_id)
            return
        logger.debug("fetch model info: fetching for chat=%s", chat_id)
        active = await fetch_active_models(server_url, user_id, chat_id)
        if not active:
            logger.debug(
                "fetch model info: server returned no roles for chat=%s", chat_id
            )
            return
        current = ensure_chat(user_id, chat_id)
        current["model_info"] = active
        logger.debug(
            "fetch model info: cached %d role(s) for chat=%s",
            len(active),
            chat_id,
        )
        _status_pane.refresh()

    async def _model_config_dialog() -> None:
        """Open a dialog to view and change per-role model overrides."""
        chat_id = _current_chat_id[0]
        if not chat_id:
            logger.debug("model config dialog: no active chat (user=%s)", user_id)
            return
        current_chat = ensure_chat(user_id, chat_id)
        current_info = current_chat.get("model_info", {})
        roles = list(current_info.keys())
        if not roles:
            # No model info yet (e.g. a chat created by typing the first
            # message). Fetch it on demand so the dialog has roles to render.
            logger.debug(
                "model config dialog: no roles cached, fetching model info for chat=%s",
                chat_id,
            )
            await _fetch_model_info()
            current_chat = ensure_chat(user_id, chat_id)
            current_info = current_chat.get("model_info", {})
            roles = list(current_info.keys())
            if not roles:
                logger.debug(
                    "model config dialog: still no roles after fetch for chat=%s",
                    chat_id,
                )
                return
        logger.debug("model config dialog: chat=%s roles=%s", chat_id, roles)

        dialog = ui.dialog()

        async def _save_role(role: str, model_inp, api_key_inp):
            payload = {"model": model_inp.value}
            if api_key_inp.value.strip():
                payload["api_key"] = api_key_inp.value.strip()
            logger.debug(
                "model config dialog: saving role=%s model=%s chat=%s",
                role,
                payload["model"],
                chat_id,
            )
            ok = await set_model_override(server_url, user_id, chat_id, role, payload)
            logger.debug(
                "model config dialog: save role=%s ok=%s chat=%s",
                role,
                ok,
                chat_id,
            )
            if ok:
                dialog.close()
                await _fetch_model_info()

        async def _clear_role(role: str):
            logger.debug("model config dialog: clearing role=%s chat=%s", role, chat_id)
            ok = await clear_model_override(server_url, user_id, chat_id, role)
            logger.debug(
                "model config dialog: clear role=%s ok=%s chat=%s",
                role,
                ok,
                chat_id,
            )
            if ok:
                dialog.close()
                await _fetch_model_info()

        with dialog, ui.card().classes("w-full p-4"):
            with ui.tabs().classes("w-full") as tabs:
                tab_map = {}
                for role in roles:
                    cfg = current_info.get(role, {})
                    modifiable = cfg.get("modifiable", True)
                    tab_icon = None
                    if cfg.get("overridden"):
                        tab_icon = "person"
                    elif not modifiable:
                        tab_icon = "lock"
                    tab_map[role] = ui.tab(
                        name=role.capitalize(),
                        label=role.capitalize(),
                        icon=tab_icon,
                    )
            with ui.tab_panels(tabs, value=roles[0].capitalize()).classes("w-full"):
                for role in roles:
                    cfg = current_info.get(role, {})
                    modifiable = cfg.get("modifiable", True)
                    with ui.tab_panel(tab_map[role]):
                        model_inp = ui.input(
                            "Model", value=cfg.get("model", "")
                        ).classes("w-full")
                        if not modifiable:
                            model_inp.disable()
                        with model_inp.add_slot("append"):
                            ui.icon("info").classes(
                                "text-sm cursor-pointer text-grey-5"
                            ).on(
                                "click",
                                lambda: ui.run_javascript(
                                    "window.open('https://neuroklea.org/install.html#choosing-models', '_blank')"
                                ),
                            )
                            ui.tooltip("See the docs for model selection options")
                        api_key_inp = ui.input(
                            "API key", password=True, password_toggle_button=True
                        ).classes("w-full")
                        if not modifiable:
                            api_key_inp.disable()
                        with api_key_inp:
                            ui.tooltip(
                                "Stored per-chat on the server.\n"
                                "Truncated in API responses.\n"
                                "Reset the override or delete the chat to remove."
                            ).classes("model-tooltip")
                        if not modifiable:
                            ui.label("Locked by administrator").classes(
                                "text-xs text-grey-5 italic"
                            )
                        else:
                            with ui.row().classes("w-full justify-end gap-2"):
                                ui.button(
                                    "Reset",
                                    on_click=lambda r=role: background_tasks.create(
                                        _clear_role(r)
                                    ),
                                ).props("flat")
                                ui.button(
                                    "Save",
                                    on_click=lambda r=role, m=model_inp, a=api_key_inp: (
                                        background_tasks.create(_save_role(r, m, a))
                                    ),
                                ).props("unelevated color=primary")

        dialog.open()

    def _switch_chat(chat_id: str) -> None:
        """Switch the active chat without a page reload."""
        _safe_set_user("chat_id", chat_id)
        _current_chat_id[0] = chat_id
        logger.debug("chat_id=%s user_id=%s", chat_id, user_id)
        _render_chat_area()
        _render_chat_list.refresh()
        _status_pane.refresh()
        _reset_center_tab()
        _render_inspector_panel.refresh()
        background_tasks.create(_fetch_model_info())

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
                _safe_set_user("chat_id", "")
                _render_chat_area()
                _render_chat_list.refresh()
                _status_pane.refresh()
                _reset_center_tab()
                _render_inspector_panel.refresh()
        else:
            _render_chat_list.refresh()

    def _toggle_pin(chat_id: str) -> None:
        """Flip the pinned flag for a chat and refresh the list."""
        current_chat = ensure_chat(user_id, chat_id)
        current_chat["pinned"] = not current_chat["pinned"]
        logger.debug(
            "toggled pin for chat=%s pinned=%s", chat_id, current_chat["pinned"]
        )
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
        logger.debug("toggling left drawer: mini_state=%s", mini_state)
        if mini_state:
            left_drawer.props("mini")
            toggle_icon_ref[0].name = "keyboard_double_arrow_right"
        else:
            left_drawer.props(remove="mini")
            toggle_icon_ref[0].name = "keyboard_double_arrow_left"

    @ui.refreshable
    def _render_inspector_panel() -> None:
        """Render the inspector entries for the active chat in the inspect tab."""
        with ui.column().classes("w-full px-2 gap-0"):
            current_chat = chats.get(f"{user_id}:{_current_chat_id[0]}")
            if not current_chat or not current_chat.get("inspector_entries"):
                ui.label("No inspection data yet").classes("text-sm text-grey-5 py-8")
                ui.label(
                    "Inspector entries will appear here after a query completes."
                ).classes("text-xs text-grey-5")
                return

            entries = list(current_chat["inspector_entries"])
            logger.debug(
                "Rendering inspector panel for chat %s (entries=%d)",
                _current_chat_id[0],
                len(entries),
            )
            for idx, entry in enumerate(entries):
                heading = entry.get("heading", "")
                timing = entry.get("timing_seconds", None)
                with (
                    ui.element("details")
                    .props("open")
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

    def _reset_center_tab() -> None:
        """Switch the central tab panel back to the chat tab."""
        panels = _center_panels[0]
        if panels is not None:
            panels.set_value("chat")
            logger.debug("reset center panel to chat tab")

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
        logger.debug("opening rename dialog for chat=%s", chat_id)
        dialog = ui.dialog()

        async def _save():
            chat["name"] = inp.value
            dialog.close()
            logger.debug("renaming chat=%s to %r", chat_id, inp.value)
            await rename_chat_on_server(server_url, user_id, chat_id, inp.value)
            _render_chat_list.refresh()
            _status_pane.refresh()

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

        # ``user_id`` is setup_layout's parameter; every handler (send,
        # _new_chat, _switch_chat, status pane, ...) closes over that same
        # name.  Declaring it nonlocal here lets us rebind it below so the
        # whole page session switches to the fresh identity immediately,
        # instead of only after a page reload.  Without this, a new chat
        # created after deleting the session would be stored under the OLD
        # user id and become orphaned from the frontend's view.
        nonlocal user_id
        old_id = user_id
        logger.debug("confirming delete of user session for user_id=%s", old_id)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.delete(f"{server_url}/chat/{old_id}")
                if resp.status_code == 200:
                    logger.debug("delete_user_session succeeded for user_id=%s", old_id)
                else:
                    logger.warning(
                        "delete_user_session failed: HTTP %s for user_id=%s",
                        resp.status_code,
                        old_id,
                    )
                    ui.notification(
                        f"Failed to delete session (HTTP {resp.status_code}). "
                        "No data was cleared.",
                        type="negative",
                        timeout=10000,
                        close_button=True,
                    )
                    dialog.close()
                    return
        except Exception as e:
            logger.warning("Failed to delete user session: %s", e)
            ui.notification(
                f"Failed to delete session: {e}",
                type="negative",
                timeout=10000,
                close_button=True,
            )
            dialog.close()
            return

        logger.debug("clearing in-memory chats for user_id=%s", old_id)
        chat_key_prefix = f"{old_id}:"
        for key in list(chats.keys()):
            if key.startswith(chat_key_prefix):
                chats.pop(key, None)

        _current_chat_id[0] = ""
        _safe_set_user("chat_id", "")
        new_id = str(uuid.uuid4())
        _safe_set_user("user_id", new_id)
        # Rebind the closure so every handler (new chat, send, status pane,
        # etc.) uses the fresh identity for the rest of this page session.
        user_id = new_id
        logger.debug(
            "reset local state: new user_id=%s (was %s)",
            user_id,
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
            logger.debug("toggled dark mode: value=%s", dark.value)

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

        # Session list header (no icon  ---  plain text signals a section heading)
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

    # ---- Right drawer (status pane, hidden by default) ----
    with (
        ui.right_drawer(value=True)
        .props("width=420 bordered")
        .classes("overflow-y-auto overflow-x-hidden")
    ):

        @ui.refreshable
        def _status_pane() -> None:
            """Render chat name, model info and state sections for the active chat in the right drawer."""
            current_chat = chats.get(f"{user_id}:{_current_chat_id[0]}")
            if not current_chat:
                if _current_chat_id[0]:
                    logger.debug(
                        "No chat found for %s, status pane empty", _current_chat_id[0]
                    )
                return
            with ui.column().classes("w-full gap-0"):
                with ui.row().classes("items-center w-full gap-0"):
                    with ui.label(current_chat.get("name", "")).classes(
                        "text-sm font-bold mb-0"
                    ):
                        created = current_chat.get("created", 0)
                        if created:
                            ui.tooltip(
                                "Created: "
                                + datetime.fromtimestamp(created)
                                .astimezone()
                                .strftime("%a %d %b %Y at %X")
                            )
                    ui.space()
                    with (
                        ui.button(
                            icon="settings",
                            on_click=_model_config_dialog,
                        )
                        .props("flat dense round color=grey-9")
                        .classes("text-sm")
                    ):
                        ui.tooltip("Choose models")
                model_info = current_chat.get("model_info", {})
                if model_info:
                    tooltip_parts: list[str] = []
                    display_parts: list[str] = []
                    for role, cfg in model_info.items():
                        raw = cfg.get("model", "")
                        provider = cfg.get("provider", "")
                        required = cfg.get("required", True)
                        if not raw:
                            # No model set for this role -- show a clear
                            # placeholder so the user knows it is missing.
                            display_short = "Not set"
                            required_mark = " (required)" if required else ""
                            tooltip_short = f"Not set{required_mark}"
                            if cfg.get("overridden"):
                                tooltip_short += " [User]"
                        else:
                            short = (
                                parse_model_name(raw).model_name
                                if parse_model_name(raw)
                                else raw
                            )
                            if provider:
                                tooltip_short = f"{short} ({provider})"
                            else:
                                tooltip_short = short
                            if cfg.get("overridden"):
                                tooltip_short += " [User]"
                            display_short = short
                        tooltip_parts.append(f"{role.capitalize()}: {tooltip_short}")
                        display_parts.append(display_short)
                    with ui.label(" | ".join(display_parts)).classes(
                        "text-xs text-grey-5"
                    ):
                        if tooltip_parts:
                            ui.tooltip("\n".join(tooltip_parts)).classes(
                                "model-tooltip"
                            )
            token_usage = current_chat.get("token_usage", {})
            has_token_usage = any(token_usage.values())
            if has_token_usage:
                usage_display = (
                    f"{token_usage.get('input_tokens', 0)} in / "
                    f"{token_usage.get('output_tokens', 0)} out"
                )
                ui.label(usage_display).classes("text-xs text-grey-5 mb-0")
            sections = current_chat.get("state_sections", {})
            has_content = bool(model_info) or bool(sections) or has_token_usage
            if not has_content:
                ui.label("State updates will appear here").classes(
                    "text-sm text-gray-500"
                )
                return
            if sections:
                ui.separator().classes("my-0.5")
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
                        ui.markdown(linkify_md(display)).classes("text-xs w-full")
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

    # ---- Center: chat messages + input (pinned to bottom) ----
    with (
        ui.column()
        .classes("w-full px-2")
        .style("flex: 1; min-height: 0; display: flex; flex-direction: column;")
    ):
        with ui.tabs().classes("w-full") as center_tabs:
            chat_tab = ui.tab(name="chat", label="chat")
            inspect_tab = ui.tab(name="inspect", label="inspect")
        with (
            ui.tab_panels(center_tabs, value="chat")
            .classes("w-full grow center-tab-panels")
            .style("flex: 1; min-height: 0; display: flex; flex-direction: column;")
        ) as center_panels:
            _center_panels[0] = center_panels
            with (
                ui.tab_panel(chat_tab)
                .classes("w-full")
                .style(
                    "flex: 1; min-height: 0; display: flex; flex-direction: column; padding: 0;"
                )
            ):
                with ui.scroll_area().classes(
                    "w-full grow chat-scroll-area"
                ) as _scroll_area:
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
                        logger.debug("Streaming query for chat %s", chat_id)
                        _is_streaming[0] = True
                        current_chat["state_sections"] = {}
                        _status_pane.refresh()
                        # Inspector entries are buffered locally and only
                        # committed on completion, so the inspect tab keeps
                        # showing the previous query's data until the
                        # current one finishes.
                        new_entries: list[dict] = []
                        with _stream_container:
                            pg_row = ui.row().classes("w-full items-center gap-2 p-2")
                            with pg_row:
                                ui.spinner(type="dots").classes("w-4 h-4")
                                pg_label = ui.label("").classes(
                                    "text-xs text-grey-5 italic"
                                )

                        full_response = ""
                        try:
                            async for event in stream_events(
                                query, chat_id, server_url, user_id=user_id
                            ):
                                t = event.get("type", "?")
                                logger.debug("chat=%s stream event type=%s", chat_id, t)
                                if t == "progress":
                                    pg_label.set_text(f"{event.get('node', '')}")
                                elif t == "debug":
                                    data = event.get("data", {})
                                    new_entries.append(
                                        {
                                            "type": t,
                                            "node": event.get("node", ""),
                                            "heading": data.get("heading", ""),
                                            "summary": data.get("summary", ""),
                                            "details": data.get("details", {}),
                                            "timing_seconds": data.get(
                                                "timing_seconds", None
                                            ),
                                        }
                                    )
                                elif t == "usage":
                                    data = event.get("data", {})
                                    details = data.get("details", {})
                                    token_usage = current_chat.setdefault(
                                        "token_usage",
                                        {
                                            "input_tokens": 0,
                                            "output_tokens": 0,
                                            "total_tokens": 0,
                                        },
                                    )
                                    for key in (
                                        "input_tokens",
                                        "output_tokens",
                                        "total_tokens",
                                    ):
                                        token_usage[key] += details.get(key, 0)
                                    _status_pane.refresh()
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
                                    full_response = event.get(
                                        "message_for_user", full_response
                                    )
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
                                    _is_streaming[0] = False
                                    _status_pane.refresh()
                                    current_chat["inspector_entries"] = new_entries
                                    current_chat["inspector_expanded"] = set()
                                    _render_inspector_panel.refresh()
                                    break
                                elif t == "error":
                                    pg_row.delete()
                                    error_msg = event.get("message", "Unknown error")
                                    logger.debug(
                                        "chat=%s stream error: %s",
                                        chat_id,
                                        error_msg,
                                    )
                                    with _stream_container:
                                        message = f"Error: {error_msg}"
                                        # Missing-model errors are actionable:
                                        # point the user at the Choose models
                                        # dialog so they can set a model and
                                        # retry without leaving the page.
                                        if "No model configured" in error_msg:
                                            message += (
                                                " Use the settings (gear) icon "
                                                "to choose a model for this "
                                                "chat, then retry."
                                            )
                                        ui.notification(
                                            message,
                                            type="negative",
                                            timeout=10000,
                                            close_button=True,
                                        )
                                    _is_streaming[0] = False
                                    _status_pane.refresh()
                                    break
                        except httpx.RequestError as e:
                            pg_row.delete()
                            logger.debug("chat=%s request error: %s", chat_id, e)
                            with _stream_container:
                                ui.notification(
                                    f"Connection error: {e}",
                                    type="negative",
                                    timeout=10000,
                                    close_button=True,
                                )
                            _is_streaming[0] = False
                            _status_pane.refresh()

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
                            _safe_set_user("chat_id", current)
                            ensure_chat(user_id, current)
                            background_tasks.create(
                                create_chat_on_server(server_url, user_id, current)
                            )
                            # Populate model_info for the newly created chat so
                            # the "Choose models" dialog has roles to render.
                            # Without this the status pane stays empty and the
                            # dialog silently no-ops for chats started by typing
                            # the first message.
                            background_tasks.create(_fetch_model_info())
                            _render_chat_list.refresh()

                        ensure_chat(user_id, current)["messages"].append(
                            (query, stamp, True)
                        )
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
                    ui.label(disclaimer).classes(
                        "text-xs text-grey-5 pb-2 w-full text-center"
                    )
            with (
                ui.tab_panel(inspect_tab)
                .classes("w-full")
                .style(
                    "flex: 1; min-height: 0; display: flex; flex-direction: column; padding: 0;"
                )
            ):
                _render_inspector_panel()

    # ---- Footer ----
    with ui.footer().classes("bg-grey-3 dark:bg-grey-9 text-xs py-1"):
        ui.html(footer_text).classes("w-full text-center text-grey-6")


def run_nicegui_app(
    title: str,
    server_url: str,
    subtitle: str = "",
    disclaimer: str = "",
    footer_text: str = 'Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
    reload: bool = False,
    nicegui_url: str = "0.0.0.0:7860",
    storage_secret: str = "klea-nicegui-secret-change-me",
    app_name: str = "klea-web",
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
    :param reload: When ``True``, enable NiceGUI's file-watch hot
        reload (``reload=True``).  Set to ``False`` in production.
    :param nicegui_url: ``host:port`` to bind the NiceGUI web server to
        (default: ``"0.0.0.0:7860"``).
    :param storage_secret: Secret used by NiceGUI for browser session
        persistence (default: ``"klea-nicegui-secret-change-me"``).
    :param app_name: Log identity for this frontend process, used as the
        log file name so each app keeps its own logs (e.g.
        ``"klea-rag-web"``).
    """
    # Configure process-wide logging for this client process.  Lazy:
    # platformdirs / plogging imports are cheap and this is a CLI entry.
    from platformdirs import PlatformDirs

    from klea_utils.plogging import resolve_log_level, setup_root_logger

    setup_root_logger(
        app_name,
        stderr_level=resolve_log_level(),
        log_dir=PlatformDirs(app_name).user_data_dir,
    )

    host, port_str = nicegui_url.rsplit(":", 1)
    port = int(port_str)

    @ui.page("/", response_timeout=30)
    async def main_page():
        """Build the main page after ensuring the WebSocket is connected.

        ``ui.run_javascript`` (used inside ``_render_chat_area``) only
        works after the client has connected, so we await
        ``ui.context.client.connected()`` first.
        """
        # Establish the per-browser user identity at the very top of the
        # page builder, before any await. `app.storage.user` is only
        # guaranteed valid in the request context of the initial page
        # build; after awaiting client.connected() / check_api_is_ready()
        # the task may run outside that context and NiceGUI raises
        # "user storage for ... should be created before accessing it".
        # If ``.nicegui/storage-user-*.json`` was lost (e.g. HF rebuild)
        # the browser still sends the old session cookie → AssertionError.
        # Warn and recreate the backing storage so the page continues.
        store = await _ensure_user_storage()
        if "user_id" not in store:
            store["user_id"] = str(uuid.uuid4())
            logger.debug("NEW user_id=%s", store["user_id"])
        else:
            logger.debug("EXISTING user_id=%s", store["user_id"])

        user_id = store["user_id"]

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

        chat_id = ""

        logger.debug("user_id=%s chat_id=%s", user_id, chat_id)
        logger.debug("before hydrate: chats keys=%s", list(chats.keys()))
        await hydrate_chats(server_url, user_id)
        logger.debug("after hydrate: chats keys=%s", list(chats.keys()))

        setup_layout(
            chat_id=chat_id,
            user_id=user_id,
            server_url=server_url,
            title=title,
            subtitle=subtitle,
            disclaimer=disclaimer,
            footer_text=footer_text,
        )

    ui.run(
        port=port,
        host=host,
        title=title,
        show=False,
        reload=reload,
        storage_secret=storage_secret,
    )
