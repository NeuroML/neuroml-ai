#!/usr/bin/env python3
"""
Streamlit chat app interface

File: neuroml_ai/streamlit_ui.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import logging
import streamlit as st
from neuroml_ai.rag import NML_RAG


def runner():
    """Main runner for streamlit app """
    st.title("NeuroML AI chat")

    if "nml_ai" not in st.session_state:
        st.session_state.nml_ai = NML_RAG(logging_level=logging.INFO)
        st.session_state.nml_ai.setup()

    # get history and re-write it
    if "history" not in st.session_state:
        st.session_state.history = []

    for i, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask a question:", key="user"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            # stream = st.session_state.nml_ai.run_graph_stream(query)
            # response = st.write_stream(stream)
            response = st.session_state.nml_ai.run_graph_invoke(query)
            st.markdown(response)
        st.session_state.history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    runner()
