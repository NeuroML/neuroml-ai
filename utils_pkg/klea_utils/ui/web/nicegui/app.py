#!/usr/bin/env python3
"""
NiceGUI entry point for Klea web interfaces.

This file is invoked directly to start the NiceGUI web interface.
It reads title / subtitle / server URL from ``sys.argv``
and delegates to :func:`klea_utils.ui.web.nicegui.runner.run_nicegui_app`.

Usage::

    python app.py <title> <subtitle> <server_url> [--reload]

File: klea_utils/ui/web/nicegui/app.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
import sys
from pathlib import Path

from platformdirs import PlatformDirs

# Default NiceGUI storage to per-app user_data_dir/nicegui when the
# deployer has not set NICEGUI_STORAGE_PATH.  nicegui/storage.py:84
# honours NICEGUI_STORAGE_PATH at import time.  For per-app isolation
# (klea-rag-web vs klea-web share the same frontend code) we need the
# app_name, which is parsed below as --app-name.  When NICEGUI_STORAGE_PATH
# is already set by the deployer (e.g. /data/nicegui on HF) we preserve it;
# otherwise we defer the per-app default to runner.py:run_nicegui_app which
# knows app_name.  This early fallback only covers the generic case when
# app.py is run directly without --app-name.
if "NICEGUI_STORAGE_PATH" not in os.environ:
    # Peek at --app-name without fully parsing (avoid consuming title/url)
    _app_name = "klea-web"
    if "--app-name" in sys.argv:
        try:
            _app_name = sys.argv[sys.argv.index("--app-name") + 1]
        except (IndexError, ValueError):
            pass
    os.environ["NICEGUI_STORAGE_PATH"] = str(
        (Path(PlatformDirs(_app_name).user_data_dir) / "nicegui").resolve()
    )

from klea_utils.ui.web.nicegui.parser import make_parser
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
        reload=args.reload,
        nicegui_url=args.nicegui_url,
        storage_secret=args.storage_secret,
        app_name=args.app_name,
    )
