#!/usr/bin/env python3
"""
Cli for klea_rag.

File: klea_rag/ui/cli.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.ui.cli import make_client_app

rag_app = make_client_app(
    label="RAG",
    server_url_default="http://127.0.0.1:8005",
    app_module="klea_rag.api.main:app",
    tui_app_name="klea-rag-tui",
    web_app_name="klea-rag-web",
)
