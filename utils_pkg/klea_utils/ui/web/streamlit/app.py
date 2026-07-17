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

from klea_utils.cli.parser import make_parser
from klea_utils.ui.web.streamlit.runner import run_streamlit_app

if __name__ == "__main__":
    args = make_parser("Klea Streamlit web interface").parse_args()
    run_streamlit_app(args.title, args.url, args.disclaimer)
