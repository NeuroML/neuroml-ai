#!/usr/bin/env python3
"""
Test store retrieval utilities.

File: utils_pkg/tests/test_stores_utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from klea_utils.stores.utils import (
    format_source_scores,
    rrf_merge,
    serialize_vs_retrieval,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def _doc(content: str) -> Document:
    """Build a fresh Document to avoid metadata mutation leaking across tests."""
    return Document(page_content=content, metadata={"file_name": "test.md"})


def test_rrf_merge_orders_and_dedupes_by_content():
    """RRF orders by fused rank and dedupes documents seen by both sources."""
    d1 = _doc("NeuroML standard")
    d2 = _doc("Hodgkin-Huxley action potential")
    d3 = _doc("LTP synaptic strength")
    d4 = _doc("bouton synapse morphology")

    vector_results = [(d1, 0.9), (d2, 0.8), (d3, 0.7)]
    bm25_results = [(d3, 5.2), (d2, 4.1), (d4, 3.0)]
    logger.info(
        f"vector results: {[(d.page_content, s) for d, s in vector_results]}"
        f"\nbm25 results: {[(d.page_content, s) for d, s in bm25_results]}"
    )

    merged = rrf_merge(
        [("vector store", vector_results), ("BM25", bm25_results)],
        num_refs_max=10,
    )
    logger.info(f"merged order: {[d.page_content for d, _ in merged]}")

    # d3 is rank 1 in BM25 and rank 3 in vector: 1/61 + 1/63 > 1/62 + 1/62
    assert [doc.page_content for doc, _ in merged][:2] == [
        d3.page_content,
        d2.page_content,
    ]
    # d4 was only in BM25; d1 only in vector
    contents = {doc.page_content for doc, _ in merged}
    assert len(contents) == len(merged), "results should be deduplicated"


def test_rrf_merge_caps_at_num_refs_max():
    """rrf_merge caps the result list at num_refs_max."""
    docs = [_doc(f"content {i}") for i in range(5)]
    results = [(doc, float(5 - i)) for i, doc in enumerate(docs)]

    merged = rrf_merge([("vector store", results)], num_refs_max=2)
    logger.info(f"num_refs_max=2, returned {len(merged)} documents")

    assert len(merged) == 2


def test_rrf_merge_preserves_source_scores():
    """Docs matched by both sources carry both original scores."""
    d1 = _doc("Hodgkin-Huxley action potential")

    merged = rrf_merge(
        [("vector store", [(d1, 0.8)]), ("BM25", [(d1, 4.1)])],
        num_refs_max=10,
    )
    logger.info(f"merged source scores: {merged[0][0].metadata['_source_scores']}")

    assert len(merged) == 1
    assert merged[0][0].metadata["_source_scores"] == {
        "vector store": 0.8,
        "BM25": 4.1,
    }


def test_format_source_scores_joins_sources():
    """format_source_scores joins per-source scores with the given precision."""
    doc = _doc("Hodgkin-Huxley action potential")
    doc.metadata["_source_scores"] = {"vector store": 0.87234, "BM25": 3.21001}

    joined = format_source_scores(doc, precision=2)
    logger.info(f"{joined = }")

    assert joined == "vector store 0.87, BM25 3.21"
    assert format_source_scores(doc, precision=4) == "vector store 0.8723, BM25 3.2100"


def test_format_source_scores_none_when_absent():
    """format_source_scores returns None when a doc has no per-source scores."""
    doc = _doc("plain content")

    result = format_source_scores(doc, precision=2)
    logger.info(f"{result = }")

    assert result is None


def test_serialize_vs_retrieval_shows_per_source_scores():
    """serialize_vs_retrieval labels per-source scores for the LLM."""
    d1 = _doc("Hodgkin-Huxley action potential")
    d1.metadata["_source_scores"] = {"vector store": 0.8723, "BM25": 3.2100}

    text = serialize_vs_retrieval({"NeuroML": [(d1, 0.0323)]})
    logger.info(f"serialized text:\n{text}")

    assert "relevance: vector store 0.8723, BM25 3.2100" in text


def test_serialize_vs_retrieval_falls_back_to_relevance_score():
    """Untagged docs fall back to the plain relevance score."""
    d1 = _doc("plain content")

    text = serialize_vs_retrieval({"NeuroML": [(d1, 0.42)]})
    logger.info(f"serialized text:\n{text}")

    assert "relevance score: 0.4200" in text
    assert "_source_scores" not in text
