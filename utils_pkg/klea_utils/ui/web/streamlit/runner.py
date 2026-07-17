#!/usr/bin/env python3
"""
Shared Streamlit chat runner.

File: klea_utils/ui/web/streamlit/runner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import uuid

import httpx
import streamlit as st

from klea_utils.api.sse import stream_events_sync
from klea_utils.api.utils import check_api_is_ready


def run_streamlit_app(title: str, url: str, subtitle: str = "") -> None:
    """Run the Streamlit chat interface.

    Waits for the backend to be ready, displays the chat history, and
    handles user queries via the SSE ``/query/stream`` endpoint.

    :param title: App title displayed at the top of the page
    :param url: Base URL of the API server (e.g. ``http://127.0.0.1:8005``)
    :param subtitle: Optional subtitle / disclaimer shown below the title
    """
    with st.spinner("Waiting for backend..."):
        asyncio.run(check_api_is_ready(f"{url}/health/ready"))

    st.title(title)
    if subtitle:
        st.info(subtitle)

    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if query := st.chat_input("Ask anything", key="user"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            progress = st.empty()
            full_response = ""
            last_node = ""

            def event_iter():
                nonlocal full_response, last_node, progress

                for event in stream_events_sync(
                    query, st.session_state.session_id, url
                ):
                    if event["type"] == "progress":
                        if event["node"] != last_node:
                            last_node = event["node"]
                            progress.caption(f"**{event['node']}**")
                    elif event["type"] == "complete":
                        progress.empty()
                        full_response = event.get("message_for_user", "")
                        yield full_response
                        return
                    elif event["type"] == "error":
                        msg = event.get("message", "Unknown server error")
                        progress.caption(f":red[Error: {msg}]")
                        full_response = f"Error: {msg}"
                        yield full_response
                        return

            try:
                st.write_stream(event_iter())
            except httpx.RequestError as e:
                st.error("An error occurred. Please try again.")
                st.error(f"```\n{e}\n```")

        st.session_state.history.append({"role": "assistant", "content": full_response})
