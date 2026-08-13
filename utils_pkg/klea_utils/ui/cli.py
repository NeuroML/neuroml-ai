#!/usr/bin/env python3
"""
Shared Typer client CLI factory for Klea packages.

File: klea_utils/ui/cli.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import importlib.util
import shlex
import subprocess
from contextlib import chdir, nullcontext
from pathlib import Path

import typer

from klea_utils.api.utils import validate_url


def _validate_url(value: str) -> str:
    try:
        return validate_url(value)
    except ValueError as e:
        raise typer.BadParameter(str(e))


def _maybe_spawn_server(server_url: str, app_module: str):
    """Context manager spawning a local server when none is running.

    Shared by the cli and web clients.  Auto-starting only makes sense for
    a server on the local machine, so this is a no-op when *server_url*
    points at a remote host.  When the local server is already running it
    is reused and left running (no-op), otherwise one is spawned and
    stopped when the ``with`` block exits.

    :param server_url: Base URL of the API server (e.g. ``http://127.0.0.1:8005``)
    :param app_module: Uvicorn module string for the server (e.g.
        ``"klea_rag.api.main:app"``)
    """
    # Lazy: spawn_server pulls in the klea_utils.api machinery.
    from klea_utils.api.server import is_loopback_host, spawn_server, split_server_url

    host, port = split_server_url(server_url)
    if not is_loopback_host(host):
        return nullcontext()
    return spawn_server(app_module, host=host, port=port)


def _run_cli(
    server_url: str,
    title: str,
    single_query: str,
    tui_app_name: str,
    app_module: str,
) -> None:
    """Run the interactive terminal (cli) client."""
    with _maybe_spawn_server(server_url, app_module):
        from klea_utils.ui.tui.repl import run_repl

        try:
            asyncio.run(
                run_repl(
                    url=server_url,
                    title=title,
                    single_query=single_query,
                    app_prefix="klea",
                    app_name=tui_app_name,
                )
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")


def _run_web(
    server_url: str,
    title: str,
    subtitle: str,
    disclaimer: str,
    footer_text: str,
    nicegui_url: str,
    storage_secret: str,
    debug: bool,
    web_app_name: str,
    app_module: str,
) -> None:
    """Run the NiceGUI web client."""
    with _maybe_spawn_server(server_url, app_module):
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
                    + f" --app-name '{web_app_name}'"
                    + (" --debug" if debug else "")
                ),
                check=False,
            )


def make_client_app(
    *,
    label: str,
    server_url_default: str,
    app_module: str,
    tui_app_name: str,
    web_app_name: str,
) -> typer.Typer:
    """Create a Typer app for a Klea user client (cli + web).

    The app exposes ``cli`` / ``web`` subcommands for the terminal and
    NiceGUI clients respectively, each carrying its own options so
    ``--help`` on a subcommand lists everything it accepts.  Invoked with
    no subcommand the app prints its usage help.  All package-specific
    values are passed as parameters, so the rag and code entry points
    stay thin wrappers.

    :param label: Short package name used in help text (e.g. ``"RAG"``)
    :param server_url_default: Default server URL (e.g.
        ``"http://127.0.0.1:8005"``)
    :param app_module: Uvicorn module string for the server this client
        talks to (e.g. ``"klea_rag.api.main:app"``)
    :param tui_app_name: Log identity for the terminal client (e.g.
        ``"klea-rag-tui"``)
    :param web_app_name: Log identity for the web client (e.g.
        ``"klea-rag-web"``)
    :returns: A :class:`typer.Typer` app for use as a CLI entry point
    """
    app = typer.Typer(help=f"Simple KLEA {label} user client")

    cli_help = f"Klea {label} cli client"
    web_help = f"Klea {label} web client (NiceGUI)"

    # Shared options are defined once and attached to both subcommands so
    # each command's --help lists everything it accepts.
    server_option = typer.Option(
        server_url_default,
        "--server",
        "-s",
        help=f"KLEA {label} server (URL:port)",
        callback=_validate_url,
    )
    title_option = typer.Option(
        f"KLEA {label}", "--title", "-t", help="Title for application"
    )

    @app.callback(invoke_without_command=True)
    def main(ctx: typer.Context):
        """Print usage help when no subcommand is given."""
        if ctx.invoked_subcommand is None:
            print("Please specify a subcommand (cli | web).")
            print(ctx.get_help())
            ctx.exit()

    @app.command()
    def cli(
        server_url: str = server_option,
        title: str = title_option,
        single_query: str = typer.Option(
            None,
            "--single-query",
            "-q",
            help="Single query mode: answer a query and exit",
        ),
    ):
        _run_cli(
            server_url=server_url,
            title=title,
            single_query=single_query or "",
            tui_app_name=tui_app_name,
            app_module=app_module,
        )

    cli.__doc__ = cli_help

    @app.command()
    def web(
        server_url: str = server_option,
        title: str = title_option,
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
        _run_web(
            server_url=server_url,
            title=title,
            subtitle=subtitle,
            disclaimer=disclaimer,
            footer_text=footer_text,
            nicegui_url=nicegui_url,
            storage_secret=storage_secret,
            debug=debug,
            web_app_name=web_app_name,
            app_module=app_module,
        )

    web.__doc__ = web_help

    return app
