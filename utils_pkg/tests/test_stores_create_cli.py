#!/usr/bin/env python3
"""
Tests for the klea-stores-create CLI logging (--debug flag).

File: tests/test_stores_create_cli.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging

from klea_utils.ui.stores_create import app
from typer.testing import CliRunner

runner = CliRunner()

#: Every klea-stores-create subcommand should accept --debug.
ALL_COMMANDS = ["pre-check", "chunk", "map-lint", "store", "store-lint", "build"]


def _make_map_lint_dir(tmp_path):
    """Create a source dir with a lintable metadata map."""
    (tmp_path / "doc.md").write_text("# Doc\n")
    cache = tmp_path / ".klea-cache"
    cache.mkdir()
    (cache / "metadata-map.template.json").write_text(
        json.dumps({"doc.md": {"DEFAULT": {}}})
    )
    return tmp_path


def test_all_commands_expose_debug_option():
    """Every subcommand's --help lists --debug."""
    for command in ALL_COMMANDS:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0, command
        assert "--debug" in result.output, command


def test_map_lint_without_debug_uses_info_level(monkeypatch, tmp_path):
    """Without --debug the logger is configured at INFO."""
    seen = {}

    def fake_setup(app_name, stderr_level=logging.INFO):
        seen["level"] = stderr_level

    monkeypatch.setattr("klea_utils.ui.stores_create.setup_root_logger", fake_setup)
    monkeypatch.delenv("KLEA_LOG_LEVEL", raising=False)

    _make_map_lint_dir(tmp_path)
    result = runner.invoke(app, ["map-lint", str(tmp_path)])
    assert result.exit_code == 0
    assert seen.get("level") == logging.INFO


def test_map_lint_with_debug_uses_debug_level(monkeypatch, tmp_path):
    """With --debug the logger is configured at DEBUG."""
    seen = {}

    def fake_setup(app_name, stderr_level=logging.INFO):
        seen["level"] = stderr_level

    monkeypatch.setattr("klea_utils.ui.stores_create.setup_root_logger", fake_setup)
    monkeypatch.delenv("KLEA_LOG_LEVEL", raising=False)

    _make_map_lint_dir(tmp_path)
    result = runner.invoke(app, ["map-lint", str(tmp_path), "--debug"])
    assert result.exit_code == 0
    assert seen.get("level") == logging.DEBUG


def test_chunk_worker_options_produce_cache_and_template(tmp_path, monkeypatch):
    """chunk runs cache-only via the worker path and writes the template.

    The conversion dispatcher is replaced with an in-process run so no
    subprocess is spawned; this verifies the CLI wires the worker options
    through and that the cache + metadata-map template are written.
    """
    from klea_utils.stores.ingestion import StoresBuilder

    src = tmp_path / "src"
    src.mkdir()
    (src / "test.md").write_text(
        "# Test\n\nEnough content here to produce at least one chunk.\n"
    )

    def fake_dispatch(self, pending, config, batch_size):
        from klea_utils.stores.chunk_worker import convert_batch_worker

        return convert_batch_worker(config, pending)

    monkeypatch.setattr(StoresBuilder, "_dispatch_conversion_batches", fake_dispatch)

    result = runner.invoke(
        app,
        [
            "chunk",
            str(src),
            "--no-ocr",
            "--worker-mem-limit",
            "64",
            "--worker-batch-size",
            "50",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (src / ".klea-cache" / "metadata-map.template.json").is_file()
    assert (src / ".klea-cache").is_dir()
