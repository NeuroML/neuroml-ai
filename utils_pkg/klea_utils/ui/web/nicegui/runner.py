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

from nicegui import app, ui


def setup_layout(session_id: str):
    """Set up the 3-column layout with left drawer, center content, and right drawer.

    Args:
        session_id: The session ID for this user (from app.storage.user)
    """
    # Left drawer for session management
    with ui.left_drawer(value=True, elevated=True, bordered=True).classes("w-64"):
        ui.label("Sessions").classes("text-lg font-bold mb-4")
        ui.button("New Session", on_click=lambda: print("New session clicked")).classes(
            "w-full mb-2"
        )
        ui.separator()
        ui.label(f"Session ID: {session_id[:8]}...").classes("text-sm text-gray-500")

    # Right drawer for inspector panel
    with ui.right_drawer(value=True, elevated=True, bordered=True).classes("w-80"):
        ui.label("Inspector").classes("text-lg font-bold mb-4")
        ui.separator()
        ui.label("Info and debug events will appear here").classes(
            "text-sm text-gray-500"
        )

    # Center content area
    with ui.column().classes("w-full h-screen flex flex-col"):
        # Header
        with ui.row().classes("w-full p-4 border-b"):
            ui.label("Chat Interface").classes("text-2xl font-bold")

        # Chat area with scroll
        with ui.scroll_area().classes("flex-1 w-full"):
            chat_container = ui.column().classes("w-full p-4")
            # Add initial welcome message
            with chat_container:
                ui.chat_message(
                    "Welcome! Type a message below to start chatting.",
                    name="System",
                    stamp="now",
                ).classes("w-full")

        # Input area
        with ui.row().classes("w-full p-4 border-t"):
            ui.input(placeholder="Type your message...").classes("flex-1")
            ui.button("Send", on_click=lambda: print("Send clicked")).classes("ml-2")


def run_nicegui_app(title: str, server_url: str, subtitle: str = ""):
    """Run the NiceGUI web interface with session management.

    Args:
        title: Application title
        server_url: Backend server URL
        subtitle: Optional subtitle/description
    """

    @ui.page("/")
    def main_page():
        """Main page with 3-column layout and session management."""
        # Get or create session_id for this user
        if "session_id" not in app.storage.user:
            app.storage.user["session_id"] = str(uuid.uuid4())

        session_id = app.storage.user["session_id"]

        # Set up the UI layout
        setup_layout(session_id=session_id)

    # Run NiceGUI server with storage support
    ui.run(
        port=7860,
        host="0.0.0.0",
        title=title,
        show=False,  # Don't auto-open browser
        reload=False,  # Disable auto-reload for production
        storage_secret="klea-nicegui-secret-change-me",  # Required for app.storage.user
    )
