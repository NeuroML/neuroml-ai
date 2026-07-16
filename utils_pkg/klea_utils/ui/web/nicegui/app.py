#!/usr/bin/env python3
"""
NiceGUI entry point for Klea web interfaces.

This file is invoked directly to start the NiceGUI web interface.
It reads title / subtitle / server URL from ``sys.argv``
and delegates to :func:`klea_utils.ui.web.nicegui.runner.run_nicegui_app`.

Usage::

    python app.py <title> <subtitle> <server_url>

File: klea_utils/ui/web/nicegui/app.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import sys

from klea_utils.ui.web.nicegui.runner import run_nicegui_app

if __name__ == "__main__":
    title = sys.argv[1]
    subtitle = sys.argv[2] if len(sys.argv) > 2 else ""
    url = sys.argv[3]
    run_nicegui_app(title, url, subtitle)
