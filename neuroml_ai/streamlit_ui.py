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

    if query := st.chat_input("Ask a question:"):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            stream = st.session_state.nml_ai.run_graph_stream(query)
            st.write_stream(stream)


if __name__ == "__main__":
    runner()
