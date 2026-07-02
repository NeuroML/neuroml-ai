#!/usr/bin/env python3
"""
Streamlit entry point for Klea chat interfaces.

This file is invoked by ``streamlit run`` (via ``subprocess`` from the
Typer CLI).  It reads title / subtitle / server URL from ``sys.argv``
and delegates to :func:`klea_utils.ui.web.streamlit.runner.run_streamlit_app`.

Usage::

    streamlit run app.py -- <title> <subtitle> <server_url>

File: klea_utils/ui/web/streamlit/app.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import sys

from klea_utils.ui.web.streamlit.runner import run_streamlit_app

if __name__ == "__main__":
    title = sys.argv[1]
    subtitle = sys.argv[2] if len(sys.argv) > 2 else ""
    url = sys.argv[3]
    run_streamlit_app(title, url, subtitle)
