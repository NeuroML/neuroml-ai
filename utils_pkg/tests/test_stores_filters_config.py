#!/usr/bin/env python3
"""
Tests for configured-domain filter normalization.

File: utils_pkg/tests/test_stores_filters_config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
from klea_utils.stores.config import FilterFieldInfo
from klea_utils.stores.filters import normalize_config_filters, validate_metadata_filter


def _fields() -> list[FilterFieldInfo]:
    return [
        FilterFieldInfo(
            name="repository_type",
            description="repository hosting type",
            value_type="string",
        ),
        FilterFieldInfo(name="year", description="publication year", value_type="int"),
        FilterFieldInfo(name="tags", description="repository tags", value_type="list"),
    ]


def test_empty_input_returns_empty_list():
    assert normalize_config_filters({}, _fields()) == []


def test_undeclared_field_dropped_with_warning(caplog):
    out = normalize_config_filters({"bogus": "x"}, _fields())
    assert out == []
    assert "bogus" in caplog.text


def test_scalar_bare_value_is_eq():
    assert normalize_config_filters({"repository_type": "github"}, _fields()) == [
        {"repository_type": {"$eq": "github"}}
    ]


def test_scalar_list_is_in():
    assert normalize_config_filters(
        {"repository_type": ["github", "dandi"]}, _fields()
    ) == [{"repository_type": {"$in": ["github", "dandi"]}}]


def test_list_field_bare_value_is_contains():
    assert normalize_config_filters({"tags": "moose"}, _fields()) == [
        {"tags": {"$contains": "moose"}}
    ]


def test_list_field_single_element_list_collapses():
    assert normalize_config_filters({"tags": ["moose"]}, _fields()) == [
        {"tags": {"$contains": "moose"}}
    ]


def test_list_field_multi_value_and_of_contains():
    assert normalize_config_filters({"tags": ["moose", "ca1"]}, _fields()) == [
        {"$and": [{"tags": {"$contains": "moose"}}, {"tags": {"$contains": "ca1"}}]}
    ]


def test_operator_expression_validated_and_normalized():
    out = normalize_config_filters({"year": {"$gte": 2020, "$lte": 2025}}, _fields())
    assert out == [{"$and": [{"year": {"$gte": 2020}}, {"year": {"$lte": 2025}}]}]


def test_operator_expression_unsupported_operator_raises():
    with pytest.raises(ValueError):
        normalize_config_filters({"repository_type": {"$like": "x%"}}, _fields())


def test_empty_operand_list_ignored_with_warning(caplog):
    out = normalize_config_filters({"repository_type": [], "tags": "moose"}, _fields())
    assert out == [{"tags": {"$contains": "moose"}}]
    assert "empty filter list" in caplog.text


def test_multiple_fields_preserve_order():
    out = normalize_config_filters(
        {"tags": "moose", "repository_type": "github", "year": 2020}, _fields()
    )
    assert out == [
        {"tags": {"$contains": "moose"}},
        {"repository_type": {"$eq": "github"}},
        {"year": {"$eq": 2020}},
    ]


def test_clauses_roundtrip_through_validate_metadata_filter():
    clauses = normalize_config_filters(
        {"repository_type": ["github", "dandi"], "tags": ["moose", "ca1"]},
        _fields(),
    )
    for clause in clauses:
        # The canonical form validates cleanly on its own.
        validate_metadata_filter(clause)


def test_list_operand_for_scalar_field_with_single_value_is_in():
    out = normalize_config_filters({"repository_type": ["github"]}, _fields())
    assert out == [{"repository_type": {"$in": ["github"]}}]
