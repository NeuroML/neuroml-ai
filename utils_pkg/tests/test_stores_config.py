#!/usr/bin/env python3
"""
Tests for the store configuration models

File: utils_pkg/tests/test_stores_config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
from klea_utils.stores.config import FilterFieldInfo, PerDomainConfig
from pydantic import ValidationError


def test_per_domain_default_no_filter_fields():
    cfg = PerDomainConfig()
    assert cfg.filter_fields == []


def test_filter_field_roundtrip():
    field = FilterFieldInfo(
        name="repository_type",
        description="repository hosting type",
        value_type="string",
    )
    dumped = field.model_dump()
    assert dumped["name"] == "repository_type"
    assert dumped["description"] == "repository hosting type"
    assert dumped["value_type"] == "string"
    assert FilterFieldInfo.model_validate(dumped) == field


def test_filter_field_default_value_type_is_string():
    field = FilterFieldInfo(name="username", description="owner username")
    assert field.value_type == "string"


def test_filter_field_rejects_invalid_value_type():
    with pytest.raises(ValidationError):
        FilterFieldInfo.model_validate(
            {"name": "tags", "description": "tags", "value_type": "dict"}
        )


def test_filter_fields_parse_from_dict():
    cfg = PerDomainConfig.model_validate(
        {
            "filter_fields": [
                {
                    "name": "repository_type",
                    "description": "repository hosting type: github, biomodels, dandi, figshare",
                    "value_type": "string",
                },
                {
                    "name": "tags",
                    "description": "repository tags",
                    "value_type": "list",
                },
            ]
        }
    )
    assert [f.name for f in cfg.filter_fields] == ["repository_type", "tags"]
    assert cfg.filter_fields[1].value_type == "list"


def test_filter_fields_survive_model_dump_include():
    cfg = PerDomainConfig(
        filter_fields=[FilterFieldInfo(name="username", description="owner username")]
    )
    dumped = cfg.model_dump(include={"vector_stores", "bm25_stores", "filter_fields"})
    assert "filter_fields" in dumped
    assert dumped["filter_fields"] == [
        {"name": "username", "description": "owner username", "value_type": "string"}
    ]
