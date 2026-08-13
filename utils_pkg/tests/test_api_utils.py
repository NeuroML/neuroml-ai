#!/usr/bin/env python3
"""
Tests for the shared API utilities.

File: tests/test_api_utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from klea_utils.api.utils import check_api_is_ready

logger = logging.getLogger(__name__)


class _ReadyHandler(BaseHTTPRequestHandler):
    """Always answers 200 with a ready payload, as a live server would."""

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status": "ready"}')

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _Always503(BaseHTTPRequestHandler):
    """Always answers 503, like a server that is up but not yet ready."""

    count = 0

    def do_GET(self):
        type(self).count += 1
        self.send_response(503)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class TestCheckApiIsReady:
    """Live checks against a throwaway server on an ephemeral port.

    The retry behaviour is validated by counting requests: an ``attempts``
    bound must stop after exactly that many probes, and the time-based
    ``timeout`` bound must give up quickly even while the server keeps
    answering.
    """

    @staticmethod
    def _serve(handler_cls):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    async def test_ready_returns_json(self):
        server = self._serve(_ReadyHandler)
        try:
            port = server.server_address[1]
            result = await check_api_is_ready(
                f"http://127.0.0.1:{port}/health/ready", attempts=3
            )
            assert result == {"status": "ready"}
        finally:
            server.shutdown()

    async def test_attempts_bound_is_honoured(self):
        _Always503.count = 0
        server = self._serve(_Always503)
        try:
            port = server.server_address[1]
            with pytest.raises(httpx.HTTPStatusError):
                await check_api_is_ready(
                    f"http://127.0.0.1:{port}/health/ready", attempts=2
                )
            # Two probes in, the attempt bound must have given up.
            assert _Always503.count == 2
            logger.debug("attempt bound stopped after %s requests", _Always503.count)
        finally:
            server.shutdown()

    async def test_timeout_bound_is_honoured(self):
        _Always503.count = 0
        server = self._serve(_Always503)
        try:
            port = server.server_address[1]
            start = time.monotonic()
            with pytest.raises(httpx.HTTPStatusError):
                await check_api_is_ready(
                    f"http://127.0.0.1:{port}/health/ready", timeout=0.1
                )
            # The time-based stop bound gives up quickly even though the
            # server keeps returning 503.
            elapsed = time.monotonic() - start
            assert elapsed < 5
            logger.debug("timeout bound gave up after %.2fs", elapsed)
        finally:
            server.shutdown()
