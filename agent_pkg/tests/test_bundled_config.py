#!/usr/bin/env python3
"""
Tests wiring the bundled tools server into the agent's resources config.

File: agent_pkg/tests/test_bundled_config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from fastmcp.mcp_config import StdioMCPServer, TransformingStdioMCPServer
from klea_agent.config import AppConfig, GeneralConfig
from klea_agent.klea_agent import KleaAgent
from klea_utils.mcp.server.config import BundledToolsConfig


def _make_agent(monkeypatch, config: AppConfig) -> KleaAgent:
    monkeypatch.setattr("klea_utils.plogging.setup_root_logger", lambda *a, **k: None)
    agent = KleaAgent(logging_level=logging.WARNING)
    agent.app_config = config
    agent._configure_resources()
    return agent


def _mcp(agent: KleaAgent):
    mcp_config = agent.mcp_config
    assert mcp_config is not None
    return mcp_config


def test_bundled_merged_by_default(monkeypatch):
    agent = _make_agent(monkeypatch, AppConfig())

    entry = _mcp(agent).mcpServers["bundled"]
    assert isinstance(entry, StdioMCPServer)
    assert entry.args == ["-m", "klea_utils.mcp.server.bundled"]

    code_config = agent.domain_mcp_configs["code"]
    assert "bundled" in code_config.mcpServers


def test_bundled_omitted_when_disabled(monkeypatch):
    config = AppConfig(
        general=GeneralConfig(bundled_tools=BundledToolsConfig(enabled=False))
    )
    agent = _make_agent(monkeypatch, config)

    assert "bundled" not in _mcp(agent).mcpServers
    assert "bundled" not in agent.domain_mcp_configs["code"].mcpServers


def test_bundled_filters_produce_transforming_server(monkeypatch):
    config = AppConfig(
        general=GeneralConfig(bundled_tools=BundledToolsConfig(include_tags={"local"}))
    )
    agent = _make_agent(monkeypatch, config)

    entry = _mcp(agent).mcpServers["bundled"]
    assert isinstance(entry, TransformingStdioMCPServer)
    assert entry.include_tags == {"local"}


def test_external_servers_still_merged_alongside_bundled(monkeypatch):
    config = AppConfig(mcp_servers={"NeuroML": {"url": "http://127.0.0.1:8542/mcp"}})
    agent = _make_agent(monkeypatch, config)

    assert set(_mcp(agent).mcpServers) == {"NeuroML", "bundled"}
    assert set(agent.domain_mcp_configs["code"].mcpServers) == {"NeuroML", "bundled"}
