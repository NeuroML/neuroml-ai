#!/usr/bin/env python3
"""
Shared server launcher factory for Klea packages.

File: klea_utils/api/server.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import typer


def make_serve_app(app_module: str, default_port: int = 8005) -> typer.Typer:
    """Create a Typer app that runs uvicorn on the given *app_module*.

    The module string should be the importable path to a FastAPI ``app``
    instance, e.g. ``"klea_rag.api.main:app"``.

    :param app_module: Uvicorn module string
    :param default_port: Default port number
    :returns: A :class:`typer.Typer` app for use as a CLI entry point
    """
    serve_app = typer.Typer()

    @serve_app.command()
    def serve(
        host: str = "127.0.0.1",
        port: int = default_port,
        dev: bool = typer.Option(
            False, "--dev", help="Enable auto-reload (like fastapi dev)"
        ),
    ):
        """Run the API server."""
        # Lazy: uvicorn pulls in starlette/httptools/websockets etc.
        import uvicorn

        uvicorn.run(
            app_module,
            host=host,
            port=port,
            reload=dev,
        )

    return serve_app
