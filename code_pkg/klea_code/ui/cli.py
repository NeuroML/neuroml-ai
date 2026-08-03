#!/usr/bin/env python3
"""
Cli for klea_code.

File: klea_code/ui/cli.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import importlib.util
import shlex
import subprocess
from contextlib import chdir
from pathlib import Path

import typer
from klea_utils.api.utils import validate_url

code_app = typer.Typer(help="Simple KLEA Code user client (WIP)")


def _validate_url(value: str) -> str:
    try:
        return validate_url(value)
    except ValueError as e:
        raise typer.BadParameter(str(e))


@code_app.command()
def cli(
    server_url: str = typer.Option(
        "http://127.0.0.1:8005",
        "--server",
        "-s",
        help="KLEA Code server (URL:port)",
        callback=_validate_url,
    ),
    single_query: str = typer.Option(
        None, "--single-query", "-q", help="Single query mode: answer a query and exit"
    ),
    title: str = typer.Option(
        "KLEA Code", "--title", "-t", help="Title for application"
    ),
):
    """Klea Code cli client"""
    from klea_utils.ui.tui.repl import run_repl

    try:
        asyncio.run(
            run_repl(
                url=server_url,
                title=title,
                single_query=single_query or "",
                app_prefix="klea",
                app_name="klea-code-tui",
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")


@code_app.command()
def web(
    title: str = typer.Option(
        "KLEA Code", "--title", "-t", help="Application title (shown in header)"
    ),
    subtitle: str = typer.Option(
        "",
        "--subtitle",
        "-b",
        help="Subtitle shown next to title in header",
    ),
    disclaimer: str = typer.Option(
        "Answers use LLM technology and may be incorrect. Please re-confirm.",
        "--disclaimer",
        "-c",
        help="Disclaimer text shown below the chat area",
    ),
    footer_text: str = typer.Option(
        'Powered by <a href="https://github.com/neuroml/klea">Klea</a>',
        "--footer",
        help="Footer HTML content",
    ),
    server_url: str = typer.Option(
        "http://127.0.0.1:8005",
        "--server",
        "-s",
        help="KLEA Code server URL:port",
        callback=_validate_url,
    ),
    nicegui_url: str = typer.Option(
        "0.0.0.0:7860",
        "--nicegui-url",
        help="Host:port to bind the NiceGUI web server to",
    ),
    storage_secret: str = typer.Option(
        "klea-nicegui-secret-change-me",
        "--storage-secret",
        help="NiceGUI storage secret for session persistence",
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Enable auto-reload on file changes"
    ),
):
    """Klea Code web client (NiceGUI)"""
    spec = importlib.util.find_spec("klea_utils.ui.web.nicegui.app")
    assert spec and spec.origin, "Could not locate nicegui app entry point"
    cwd = Path(spec.origin).parent
    with chdir(cwd):
        subprocess.run(
            shlex.split(
                f"python app.py '{title}' '{subtitle}' '{server_url}'"
                + f" --disclaimer '{disclaimer}'"
                + f" --footer '{footer_text}'"
                + f" --nicegui-url '{nicegui_url}'"
                + f" --storage-secret '{storage_secret}'"
                + " --app-name 'klea-code-web'"
                + (" --debug" if debug else "")
            )
        )
