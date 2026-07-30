#!/usr/bin/env python3
"""
Shared CLI argument parser for Klea frontend entry points.

Provides a standard :mod:`argparse` parser so that all frontends
(nicegui, streamlit, textual, etc.) handle the same set of
positional arguments and optional flags in the same way.  Each
entry point simply calls :func:`make_parser().parse_args()` and
picks only the arguments it needs; unrecognised options are
silently ignored (they are never passed by the Typer CLI commands).

Usage::

    from klea_utils.cli.parser import make_parser

    args = make_parser("My frontend").parse_args()
    run_app(
        args.title, args.url,
        subtitle=args.subtitle,
        disclaimer=args.disclaimer,
        debug=args.debug,
    )

File: klea_utils/cli/parser.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import argparse


def make_parser(
    description: str = "Klea web interface",
) -> argparse.ArgumentParser:
    """Return a preconfigured :class:`argparse.ArgumentParser`.

    :param description: Description shown in ``--help``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("title", help="Application title (shown in header)")
    parser.add_argument(
        "subtitle", nargs="?", default="", help="Text shown next to title in header"
    )
    parser.add_argument("url", help="Backend server URL")
    parser.add_argument(
        "--disclaimer",
        default="",
        help="Disclaimer text shown below the chat area",
    )
    parser.add_argument(
        "--footer",
        default='Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
        help="Footer HTML content",
    )
    parser.add_argument(
        "--nicegui-url",
        default="0.0.0.0:7860",
        help="Host:port to bind the NiceGUI web server to (default: 0.0.0.0:7860)",
    )
    parser.add_argument(
        "--storage-secret",
        default="klea-nicegui-secret-change-me",
        help="NiceGUI storage secret for session persistence",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable auto-reload on file changes (supported by nicegui only)",
    )
    return parser
