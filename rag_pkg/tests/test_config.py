#!/usr/bin/env python3
"""
Tests for the RAG config module (profile templates).

File: tests/test_config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
from pathlib import Path

import pytest
from klea_rag.config import AppConfig, write_config_template
from klea_rag.ui.cli import rag_app
from typer.testing import CliRunner


class TestWriteConfigTemplate:
    """Unit tests for :func:`klea_rag.config.write_config_template`."""

    def test_roundtrips_through_app_config(self, tmp_path):
        target = write_config_template(tmp_path)
        data = json.loads(target.read_text())
        config = AppConfig.model_validate(data)
        assert config.general.default_k == 5
        assert "ExampleDomain" in config.domains
        domain = config.domains["ExampleDomain"]
        assert domain.description
        assert domain.vector_stores[0].path.startswith("chroma:")
        assert domain.bm25_stores[0].name == "my-docs-bm25"

    def test_refuses_overwrite(self, tmp_path):
        write_config_template(tmp_path)
        with pytest.raises(FileExistsError):
            write_config_template(tmp_path)


class TestProfileTemplateCli:
    """``--profile template`` scaffolds a config through the CLI."""

    def test_cli_template_writes_config(self, tmp_path, monkeypatch):
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(rag_app, ["cli", "--profile", "template"])
        assert result.exit_code == 0
        assert (Path.cwd() / "klea_rag.json").exists()
        assert "Template config written" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
