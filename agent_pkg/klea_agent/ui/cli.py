#!/usr/bin/env python3
"""
Cli for klea_code.

File: klea_code/ui/cli.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.ui.cli import make_client_app

code_app = make_client_app(
    label="Code",
    server_url_default="http://127.0.0.1:8006",
    app_module="klea_code.api.main:app",
    tui_app_name="klea-code-tui",
    web_app_name="klea-code-web",
)
