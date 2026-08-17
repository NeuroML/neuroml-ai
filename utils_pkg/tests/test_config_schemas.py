#!/usr/bin/env python3
"""
Tests for the shared app-config schema additions.

File: utils_pkg/tests/test_config_schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.mcp.server.config import BundledToolsConfig


def test_defaults():
    cfg = BundledToolsConfig()
    assert cfg.enabled is True
    assert cfg.include_tags == set()
    assert cfg.exclude_tags == set()


def test_disabled_roundtrip():
    cfg = BundledToolsConfig(enabled=False)
    assert cfg.model_dump()["enabled"] is False
    assert BundledToolsConfig.model_validate(cfg.model_dump()).enabled is False


def test_tag_filters_parse_from_json():
    raw = {
        "enabled": True,
        "include_tags": ["local", "files"],
        "exclude_tags": ["code"],
    }
    cfg = BundledToolsConfig.model_validate(raw)
    assert cfg.include_tags == {"local", "files"}
    assert cfg.exclude_tags == {"code"}


def test_tag_filters_serialize_to_json_arrays():
    cfg = BundledToolsConfig(include_tags={"local"}, exclude_tags={"code"})
    data = cfg.model_dump(mode="json")
    assert data["include_tags"] == ["local"]
    assert data["exclude_tags"] == ["code"]


def test_empty_tag_lists_serialize():
    data = BundledToolsConfig().model_dump(mode="json")
    assert data["include_tags"] == []
    assert data["exclude_tags"] == []
