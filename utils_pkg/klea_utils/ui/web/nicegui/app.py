#!/usr/bin/env python3
"""
NiceGUI entry point for Klea web interfaces.

This file is invoked directly to start the NiceGUI web interface.
It reads title / subtitle / server URL from ``sys.argv``
and delegates to :func:`klea_utils.ui.web.nicegui.runner.run_nicegui_app`.

Usage::

    python app.py <title> <subtitle> <server_url> [--debug]

File: klea_utils/ui/web/nicegui/app.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.cli.parser import make_parser
from klea_utils.ui.web.nicegui.runner import run_nicegui_app

# Use the multiprocessing-safe guard so that NiceGUI's file-watch reload
# (which spawns a subprocess where ``__name__`` is ``"__mp_main__"``)
# does not raise a RuntimeError.
if __name__ in {"__main__", "__mp_main__"}:
    args = make_parser("Klea NiceGUI web interface").parse_args()
    run_nicegui_app(
        args.title,
        args.url,
        subtitle=args.subtitle,
        disclaimer=args.disclaimer,
        footer_text=args.footer,
        debug=args.debug,
    )
