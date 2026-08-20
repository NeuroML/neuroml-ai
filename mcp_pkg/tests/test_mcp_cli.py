#!/usr/bin/env python3
"""
Tests for the nml-mcp CLI logging (--debug flag).

File: tests/test_mcp_cli.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from neuroml_mcp.server.main import mcp_app
from typer.testing import CliRunner

runner = CliRunner()


def test_mcp_cli_exposes_debug_option():
    """The nml-mcp command's --help lists --debug."""
    result = runner.invoke(mcp_app, ["--help"])
    assert result.exit_code == 0
    assert "--debug" in result.output


def test_mcp_cli_without_debug_uses_info_level(monkeypatch):
    """Without --debug the logger is configured at INFO."""
    seen = {}

    def fake_setup(app_name, stderr_level=logging.INFO, **kwargs):
        seen["level"] = stderr_level

    monkeypatch.setattr("klea_utils.plogging.setup_root_logger", fake_setup)
    monkeypatch.delenv("KLEA_LOG_LEVEL", raising=False)

    # Stub out the heavy server creation so only the logging wiring runs.
    class FakeMcp:
        def run(self, *a, **k):
            return None

    monkeypatch.setattr("neuroml_mcp.server.main.create_server", lambda *a, **k: None)
    monkeypatch.setattr(
        "neuroml_mcp.server.main.asyncio.run", lambda *a, **k: FakeMcp()
    )
    result = runner.invoke(mcp_app, [])
    assert result.exit_code == 0
    assert seen.get("level") == logging.INFO


def test_mcp_cli_with_debug_uses_debug_level(monkeypatch):
    """With --debug the logger is configured at DEBUG."""
    seen = {}

    def fake_setup(app_name, stderr_level=logging.INFO, **kwargs):
        seen["level"] = stderr_level

    monkeypatch.setattr("klea_utils.plogging.setup_root_logger", fake_setup)
    monkeypatch.delenv("KLEA_LOG_LEVEL", raising=False)

    class FakeMcp:
        def run(self, *a, **k):
            return None

    monkeypatch.setattr("neuroml_mcp.server.main.create_server", lambda *a, **k: None)
    monkeypatch.setattr(
        "neuroml_mcp.server.main.asyncio.run", lambda *a, **k: FakeMcp()
    )
    result = runner.invoke(mcp_app, ["--debug"])
    assert result.exit_code == 0
    assert seen.get("level") == logging.DEBUG
