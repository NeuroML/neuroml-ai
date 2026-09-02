#!/usr/bin/env python3
"""
Tests for the retrieval-query output schema

File: rag_pkg/tests/test_schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
from klea_rag.schemas import RetrievalQueryOutput
from klea_utils.stores.filters import translate_metadata_filter


def test_no_constraints_returns_none():
    assert RetrievalQueryOutput().to_metadata_filter() is None


def test_single_config_clause_returned_directly():
    rq = RetrievalQueryOutput(
        search_query="repos", config_filters=[{"repository_type": {"$eq": "github"}}]
    )
    assert rq.to_metadata_filter() == {"repository_type": {"$eq": "github"}}


def test_multiple_config_clauses_wrapped_in_and():
    rq = RetrievalQueryOutput(
        search_query="repos",
        config_filters=[
            {"repository_type": {"$eq": "github"}},
            {"content_types": {"$eq": "modeling"}},
            {"year": {"$gte": 2020}},
        ],
    )
    assert rq.to_metadata_filter() == {
        "$and": [
            {"repository_type": {"$eq": "github"}},
            {"content_types": {"$eq": "modeling"}},
            {"year": {"$gte": 2020}},
        ]
    }


def test_config_multi_contains_and_merges():
    """A nested ``$and`` clause from normalize_config_filters merges as-is."""
    rq = RetrievalQueryOutput(
        search_query="papers",
        config_filters=[
            {
                "$and": [
                    {"tags": {"$contains": "moose"}},
                    {"tags": {"$contains": "ca1"}},
                ]
            },
            {"journal": {"$eq": "nature"}},
        ],
    )
    assert rq.to_metadata_filter() == {
        "$and": [
            {
                "$and": [
                    {"tags": {"$contains": "moose"}},
                    {"tags": {"$contains": "ca1"}},
                ]
            },
            {"journal": {"$eq": "nature"}},
        ]
    }


def test_raw_filters_do_not_affect_metadata_filter():
    """to_metadata_filter() runs only on the normalized config_filters."""
    rq = RetrievalQueryOutput(
        search_query="repos",
        filters={"repository_type": "github"},
        config_filters=[{"repository_type": {"$eq": "github"}}],
    )
    assert rq.to_metadata_filter() == {"repository_type": {"$eq": "github"}}

    rq2 = RetrievalQueryOutput(search_query="repos", filters={"tags": "moose"})
    assert rq2.to_metadata_filter() is None


def test_raw_and_normalized_filters_survive_roundtrip():
    rq = RetrievalQueryOutput(
        search_query="repos",
        filters={"repository_type": ["github", "dandi"]},
        config_filters=[{"repository_type": {"$in": ["github", "dandi"]}}],
    )
    restored = RetrievalQueryOutput.model_validate(rq.model_dump())
    assert restored.filters == {"repository_type": ["github", "dandi"]}
    assert restored.config_filters == [
        {"repository_type": {"$in": ["github", "dandi"]}}
    ]


def test_filter_accepts_backend_translation():
    rq = RetrievalQueryOutput(
        search_query="repos",
        config_filters=[
            {"repository_type": {"$in": ["github", "dandi"]}},
            {"tags": {"$contains": "moose"}},
        ],
    )
    out = rq.to_metadata_filter()
    assert out is not None
    translated = translate_metadata_filter("chroma:/data/store", out)
    assert translated["$and"][0] == {"repository_type": {"$in": ["github", "dandi"]}}
    assert translated["$and"][1] == {"tags": {"$contains": "moose"}}


def test_malformed_config_clauses_rejected_on_translation():
    """Malformed operator expressions surface as ValidationError (ValueError)."""
    rq = RetrievalQueryOutput(
        search_query="repos", config_filters=[{"tags": {"$nonexistent": 1}}]
    )
    out = rq.to_metadata_filter()
    assert out is not None
    with pytest.raises(ValueError):
        translate_metadata_filter("chroma:/data/store", out)
