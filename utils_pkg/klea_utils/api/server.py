#!/usr/bin/env python3
"""
Shared server launcher factory for Klea packages.

File: klea_utils/api/server.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from subprocess import Popen
from urllib.parse import urlsplit

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


def split_server_url(url: str, default_port: int = 8005) -> tuple[str, int]:
    """Return ``(host, port)`` parsed from *url*.

    Falls back to ``127.0.0.1`` and *default_port* when the URL does not
    carry a hostname or port.

    :param url: Server URL (e.g. ``http://127.0.0.1:8005``)
    :param default_port: Port to use when the URL omits one
    :returns: ``(host, port)`` suitable for binding a local server
    """
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    try:
        port = parts.port or default_port
    except ValueError:
        port = default_port
    return host, port


def is_loopback_host(host: str) -> bool:
    """Return ``True`` when *host* refers to the local machine.

    :param host: Hostname from a server URL (e.g. ``"127.0.0.1"``)
    :returns: ``True`` for loopback addresses, ``False`` otherwise
    """
    return host.lower() in {"127.0.0.1", "localhost", "::1"}


@contextmanager
def spawn_server(
    app_module: str,
    host: str = "127.0.0.1",
    port: int = 8005,
    timeout: float = 180.0,
) -> Iterator[Popen | None]:
    """Context manager that runs an API server in a subprocess.

    If a healthy server is already listening at ``host:port`` (probed via
    ``/health/ready``), nothing is spawned and ``None`` is yielded, so the
    caller does not own the server's lifecycle.  Otherwise a uvicorn
    subprocess is spawned (stdout and stderr inherited so startup errors and
    app output are visible; access logs are disabled to keep the shared
    terminal clean, with full logging preserved in the server's rotating log
    file), readiness is waited on, and the subprocess is terminated when the
    ``with`` block exits.

    :param app_module: Uvicorn module string (e.g. ``"klea_rag.api.main:app"``)
    :param host: Host to bind
    :param port: Port to bind
    :param timeout: Total seconds to wait for readiness after spawning
    :returns: The spawned :class:`subprocess.Popen` (or ``None`` if an
        existing server was reused)
    """
    # Lazy: asyncio/httpx and the api utils pull in heavy deps; keep --help fast.
    import asyncio

    import httpx

    from klea_utils.api.utils import check_api_is_ready

    health_url = f"http://{host}:{port}/health/ready"

    def _probe_once() -> bool:
        try:
            asyncio.run(check_api_is_ready(health_url, attempts=1))
            return True
        except (httpx.HTTPError, OSError):
            return False

    if _probe_once():
        yield None
        return

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            app_module,
            "--host",
            host,
            "--port",
            str(port),
            "--no-access-log",
        ],
    )

    try:
        # A short fast-fail window catches an instantly-crashed server
        # (bad module path, port already in use) so the error surfaces
        # immediately instead of after the full retry timeout.
        for _ in range(5):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server process exited immediately (code {proc.returncode}). "
                    "Check the server's log file, or run the server directly "
                    "with 'klea-rag-serve serve' / 'klea-serve serve' "
                    "to see the error."
                )
            if _probe_once():
                break
            time.sleep(1.0)
        else:
            try:
                asyncio.run(check_api_is_ready(health_url, timeout=timeout))
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"Server at {health_url} did not become ready within {timeout:g}s. "
                    "Check the server's log file, or run the server directly "
                    "with 'klea-rag-serve serve' / 'klea-serve serve' "
                    "to see the error."
                ) from exc

        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
