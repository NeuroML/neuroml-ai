#!/usr/bin/env python3
"""
Tests wiring the bundled tools server into the RAG's resources config.

File: rag_pkg/tests/test_bundled_config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from fastmcp.mcp_config import StdioMCPServer, TransformingStdioMCPServer
from klea_rag.config import AppConfig, GeneralConfig, PerDomainConfig
from klea_rag.rag import RAG
from klea_utils.mcp.server.config import BundledToolsConfig


def _make_rag(monkeypatch, config: AppConfig) -> RAG:
    monkeypatch.setattr("klea_utils.plogging.setup_root_logger", lambda *a, **k: None)
    rag = RAG(logging_level=logging.WARNING)
    rag.app_config = config
    rag._configure_resources()
    return rag


def _app_config(bundled: BundledToolsConfig | None = None) -> AppConfig:
    return AppConfig(
        general=GeneralConfig(
            bundled_tools=bundled or BundledToolsConfig(enabled=False)
        ),
        domains={
            "Alpha": PerDomainConfig(description="first domain"),
            "Beta": PerDomainConfig(description="second domain"),
        },
    )


def _mcp(rag: RAG):
    mcp_config = rag.mcp_config
    assert mcp_config is not None
    return mcp_config


def test_bundled_absent_by_default(monkeypatch):
    rag = _make_rag(monkeypatch, _app_config())

    assert "bundled" not in _mcp(rag).mcpServers
    for domain_config in rag.domain_mcp_configs.values():
        assert "bundled" not in domain_config.mcpServers


def test_bundled_spread_across_all_domains_when_enabled(monkeypatch):
    rag = _make_rag(monkeypatch, _app_config(bundled=BundledToolsConfig(enabled=True)))

    entry = _mcp(rag).mcpServers["bundled"]
    assert isinstance(entry, StdioMCPServer)
    assert entry.args == ["-m", "klea_utils.mcp.server.bundled"]

    assert set(rag.domain_mcp_configs) == {"Alpha", "Beta"}
    for domain_config in rag.domain_mcp_configs.values():
        assert set(domain_config.mcpServers) == {"bundled"}


def test_bundled_joins_existing_domain_servers(monkeypatch):
    config = AppConfig(
        general=GeneralConfig(bundled_tools=BundledToolsConfig(enabled=True)),
        domains={
            "Alpha": PerDomainConfig(
                description="first domain",
                mcp_servers={"NeuroML": {"url": "http://127.0.0.1:8542/mcp"}},
            ),
        },
    )
    rag = _make_rag(monkeypatch, config)

    assert set(_mcp(rag).mcpServers) == {"NeuroML", "bundled"}
    alpha = rag.domain_mcp_configs["Alpha"]
    assert set(alpha.mcpServers) == {"NeuroML", "bundled"}


def test_bundled_filters_produce_transforming_server(monkeypatch):
    rag = _make_rag(
        monkeypatch,
        _app_config(bundled=BundledToolsConfig(include_tags={"remote"})),
    )

    entry = _mcp(rag).mcpServers["bundled"]
    assert isinstance(entry, TransformingStdioMCPServer)
    assert entry.include_tags == {"remote"}
