#!/usr/bin/env python3
"""
Test store metadata filter translation.

File: utils_pkg/tests/test_stores_filters.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import pytest
from klea_utils.stores.filters import (
    filter_docs_by_metadata,
    to_chroma_filter,
    to_pgvector_filter,
    to_qdrant_filter,
    validate_metadata_filter,
)
from klea_utils.stores.utils import expand_person_names
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class _FakeEmbeddings(Embeddings):
    """Minimal embedding object for Chroma; never called during init."""

    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 8


def _docs():
    return [
        Document(
            page_content="a",
            metadata={
                "journal": "Nature",
                "authors": ["Magee", "Smith"],
                "year": 2020,
                "keywords": ["cortex"],
            },
        ),
        Document(
            page_content="b",
            metadata={
                "journal": "eLife",
                "authors": ["Jones"],
                "year": 2024,
                "keywords": ["hippocampus"],
            },
        ),
        Document(
            page_content="c",
            metadata={
                "journal": "Nature",
                "authors": ["Magee"],
                "year": 2022,
                "keywords": ["cortex", "network"],
            },
        ),
    ]


# ----------------------------------------------------------------------
# validate_metadata_filter: normalization
# ----------------------------------------------------------------------


def test_validate_wraps_bare_value_as_eq():
    """A bare scalar value becomes an explicit $eq clause."""
    assert validate_metadata_filter({"journal": "Nature"}) == {
        "journal": {"$eq": "Nature"}
    }


def test_validate_splits_multi_operator_field():
    """A multi-operator field clause splits into an $and of single ops."""
    assert validate_metadata_filter({"year": {"$gte": 2020, "$lte": 2025}}) == {
        "$and": [
            {"year": {"$gte": 2020}},
            {"year": {"$lte": 2025}},
        ]
    }


def test_validate_combines_multiple_top_level_fields():
    """Multiple top-level fields combine into an $and."""
    result = validate_metadata_filter({"journal": "Nature", "year": {"$gte": 2020}})
    assert result == {
        "$and": [
            {"journal": {"$eq": "Nature"}},
            {"year": {"$gte": 2020}},
        ]
    }


def test_validate_collapses_single_element_and():
    """An $and with a single element collapses to that clause."""
    assert validate_metadata_filter({"$and": [{"journal": "Nature"}]}) == {
        "journal": {"$eq": "Nature"}
    }


def test_validate_rejects_empty_filter():
    with pytest.raises(ValueError):
        validate_metadata_filter({})


def test_validate_rejects_unknown_operator():
    with pytest.raises(ValueError):
        validate_metadata_filter({"journal": {"$frobnicate": "Nature"}})


def test_validate_rejects_bad_in_operand():
    with pytest.raises(ValueError):
        validate_metadata_filter({"year": {"$in": []}})


# ----------------------------------------------------------------------
# Backend translators
#
# TODO: the Qdrant and pgvector tests below assert the translated filter
# *structure* only, not behaviour against a live store (no Qdrant/pgvector
# stores exist in this repo).  Add toy Qdrant + pgvector vector stores to
# the test fixtures so these filters are exercised end to end, mirroring
# test_to_chroma_filter_filters_real_store.
# ----------------------------------------------------------------------


def test_to_chroma_filter_returns_normalized_form():
    """The canonical form is Chroma's native where syntax."""
    assert to_chroma_filter({"journal": "Nature"}) == {"journal": {"$eq": "Nature"}}
    assert to_chroma_filter({"year": {"$gte": 2020, "$lte": 2025}}) == {
        "$and": [
            {"year": {"$gte": 2020}},
            {"year": {"$lte": 2025}},
        ]
    }


def test_to_chroma_filter_filters_real_store(tmp_path):
    """A filter produced by to_chroma_filter filters a real Chroma store."""
    pytest.importorskip("langchain_chroma")
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name="test",
        embedding_function=_FakeEmbeddings(),
        persist_directory=str(tmp_path),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    store.add_documents(_docs())

    # The multi-op year range must be split before chroma accepts it.
    results = store.similarity_search(
        "x", k=10, filter=to_chroma_filter({"year": {"$gte": 2021, "$lte": 2023}})
    )
    assert sorted(d.page_content for d in results) == ["c"]

    # $contains is Chroma's array-membership operator.
    results = store.similarity_search(
        "x", k=10, filter=to_chroma_filter({"authors": {"$contains": "Magee"}})
    )
    assert sorted(d.page_content for d in results) == ["a", "c"]

    results = store.similarity_search(
        "x",
        k=10,
        filter=to_chroma_filter(
            {"$and": [{"journal": "Nature"}, {"year": {"$gte": 2021}}]}
        ),
    )
    assert sorted(d.page_content for d in results) == ["c"]


def test_to_qdrant_filter_builds_native_objects():
    """The Qdrant translator builds a Filter tree of FieldConditions."""
    pytest.importorskip("qdrant_client")
    from qdrant_client import models

    f = to_qdrant_filter({"year": {"$gte": 2021, "$lte": 2023}, "journal": "Nature"})
    logger.info(f"qdrant filter: {f}")
    assert isinstance(f, models.Filter)

    def _collect(cond):
        """Flatten nested Filters into the leaf FieldConditions."""
        if isinstance(cond, models.FieldCondition):
            return [cond]
        out = []
        for sub in (cond.must or []) + (cond.should or []):
            out.extend(_collect(sub))
        return out

    conditions = _collect(f)
    assert len(conditions) == 3

    ranges = [c for c in conditions if c.range is not None]
    matches = [c for c in conditions if c.match is not None]
    assert {c.key for c in ranges} == {"year"}
    assert {c.key for c in matches} == {"journal"}
    assert {c.range.gte for c in ranges if c.range.gte is not None} == {2021.0}
    assert {c.range.lte for c in ranges if c.range.lte is not None} == {2023.0}


def test_to_qdrant_filter_contains_is_match_value():
    """List membership is a MatchValue (array element match)."""
    pytest.importorskip("qdrant_client")
    from qdrant_client import models

    f = to_qdrant_filter({"authors": {"$contains": "Magee"}})
    condition = f.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert isinstance(condition.match, models.MatchValue)
    assert condition.match.value == "Magee"


def test_to_pgvector_filter_contains_becomes_like():
    """pgvector has no array-contains op; approximated with a like."""
    assert to_pgvector_filter({"authors": {"$contains": "Magee"}}) == {
        "authors": {"$like": "%Magee%"}
    }


def test_to_pgvector_filter_passes_other_ops():
    """Range clauses pass through unchanged (single top-level key)."""
    assert to_pgvector_filter({"year": {"$gte": 2020, "$lte": 2025}}) == {
        "$and": [
            {"year": {"$gte": 2020}},
            {"year": {"$lte": 2025}},
        ]
    }


# ----------------------------------------------------------------------
# filter_docs_by_metadata (BM25 post-filter)
# ----------------------------------------------------------------------


def test_filter_docs_by_metadata_contains_list_field():
    got = filter_docs_by_metadata(_docs(), {"authors": {"$contains": "Magee"}})
    assert sorted(d.page_content for d in got) == ["a", "c"]


def test_filter_docs_by_metadata_range_and_or():
    docs = _docs()
    got = filter_docs_by_metadata(
        docs, {"$and": [{"journal": "Nature"}, {"year": {"$gte": 2021}}]}
    )
    assert sorted(d.page_content for d in got) == ["c"]

    got = filter_docs_by_metadata(
        docs, {"$or": [{"year": {"$gte": 2023}}, {"journal": "eLife"}]}
    )
    assert sorted(d.page_content for d in got) == ["b"]

    got = filter_docs_by_metadata(docs, {"year": {"$gte": 2021, "$lte": 2023}})
    assert sorted(d.page_content for d in got) == ["c"]


def test_filter_docs_by_metadata_missing_metadata_never_matches():
    got = filter_docs_by_metadata(
        [Document(page_content="x", metadata={"file_name": "f.md"})],
        {"journal": "Nature"},
    )
    assert got == []


def test_filter_docs_by_metadata_matches_expanded_person_name_variants():
    """A partial-name operand matches an author list expanded at store time."""
    docs = [
        Document(
            page_content="p",
            metadata={"authors": expand_person_names(["Ankur Sinha"])},
        )
    ]
    assert filter_docs_by_metadata(docs, {"authors": {"$contains": "Sinha"}})
    assert filter_docs_by_metadata(docs, {"authors": {"$contains": "sinha"}})
    assert filter_docs_by_metadata(docs, {"authors": {"$contains": "Ankur Sinha"}})
    assert filter_docs_by_metadata(docs, {"authors": {"$contains": "ankur sinha"}})
    assert not filter_docs_by_metadata(docs, {"authors": {"$contains": "Magee"}})
