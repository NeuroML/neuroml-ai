#!/usr/bin/env python3
"""
Tests for the RAG retrieve information node.

File: rag_pkg/tests/test_retrieve_info.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any

from klea_rag.nodes.retrieve_info import RetrieveInfoNode
from klea_rag.schemas import EvaluateAnswerSchema, RAGState
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class FakeRetriever:
    """Minimal retriever returning canned results and recording k calls."""

    def __init__(self, results, name="fake"):
        self.results = results
        self.source_label = name
        self.inc_count = 0
        self.reset_count = 0
        self.queries: list[str] = []

    def retrieve(self, domain_name, query):
        self.queries.append(query)
        return self.results

    def inc_k(self):
        self.inc_count += 1
        return True

    def reset_k(self):
        self.reset_count += 1


def _make_node(retrievers) -> RetrieveInfoNode:
    node = object.__new__(RetrieveInfoNode)
    node.logger = logging.getLogger("test_retrieve_info")
    node.label = "Retrieving information"
    node.retrievers = retrievers
    node.num_refs_max = 10
    node.write_custom_stream = lambda event: None
    logger.info(
        f"configured retrievers: "
        f"{[(r.source_label, len(r.results)) for r in retrievers]}"
    )
    return node


def _doc(content: str) -> Document:
    return Document(page_content=content, metadata={"file_name": "test.md"})


async def test_execute_merges_retrievers_with_rrf():
    """execute() fuses results from all retrievers with RRF."""
    d1 = _doc("NeuroML standard")
    d2 = _doc("Hodgkin-Huxley action potential")
    r1 = FakeRetriever([(d1, 0.9), (d2, 0.8)], name="vector store")
    r2 = FakeRetriever([(d2, 4.1)], name="BM25")
    node = _make_node([r1, r2])

    state = RAGState(
        query="action potential",
        query_domains=["NeuroML"],
        retrieval_query="action potential",
    )
    result = await node.execute(state)

    refs = result["reference_material"]["NeuroML"]
    logger.info(
        f"merged order: {[d.page_content for d, _ in refs]}"
        f"\nsource scores: {[d.metadata.get('_source_scores') for d, _ in refs]}"
    )

    assert [doc.page_content for doc, _ in refs][:2] == [
        d2.page_content,
        d1.page_content,
    ]
    # d2 was seen by both sources, so it carries both original scores
    assert refs[0][0].metadata["_source_scores"] == {
        "vector store": 0.8,
        "BM25": 4.1,
    }


async def test_execute_inc_k_on_all_retrievers_for_more_info():
    """inc_k() is called on every retriever when more info is requested."""
    r1 = FakeRetriever([], name="vector store")
    r2 = FakeRetriever([], name="BM25")
    node = _make_node([r1, r2])

    state = RAGState(
        query="q",
        query_domains=["NeuroML"],
        retrieval_query="q",
        text_response_eval=EvaluateAnswerSchema(next_step="retrieve_more_info"),
    )
    await node.execute(state)
    logger.info(f"inc counts: r1={r1.inc_count}, r2={r2.inc_count}")

    assert r1.inc_count == 1
    assert r2.inc_count == 1


async def test_execute_normalizes_retrieval_query():
    """Artifact-laden retrieval queries reach every retriever normalized.

    Queries are LLM-generated so artifacts are rare, but they must share
    the same plain-text form as indexed chunks (soft hyphens, no-break
    spaces, etc. stripped at indexing time).
    """
    r1 = FakeRetriever([(_doc("content"), 1.0)], name="vector store")
    r2 = FakeRetriever([(_doc("content"), 1.0)], name="BM25")
    node = _make_node([r1, r2])

    state = RAGState(
        query="q",
        query_domains=["NeuroML"],
        retrieval_query="multi-\u00adscale model\u00ading in\u00a0neuroscience",
    )
    await node.execute(state)

    expected = "multi-scale modeling in neuroscience"
    assert r1.queries == [expected]
    assert r2.queries == [expected]


async def test_execute_labels_url_keys_in_display():
    """url_<label> keys display as 'label: <url>'; bare/numbered stay plain."""
    events: list[dict] = []

    doc = _doc("NeuroML content")
    doc.metadata.update(
        {
            "url": "https://example.org/main",
            "url_1": "https://example.org/extra",
            "url_orcid": "https://orcid.org/0000-0000-0000-0000",
        }
    )
    node: Any = _make_node([FakeRetriever([(doc, 0.9)], name="vector store")])
    node.write_custom_stream = lambda event: events.append(event)

    state = RAGState(
        query="q",
        query_domains=["NeuroML"],
        retrieval_query="q",
    )
    await node.execute(state)

    display = next(e["data"]["display"] for e in events if e.get("type") == "state")
    logger.info(f"reference display markdown:\n{display}")

    assert "- orcid: https://orcid.org/0000-0000-0000-0000" in display
    assert "- https://example.org/main" in display
    assert "- https://example.org/extra" in display


async def test_execute_no_retrievers_skips():
    """execute() returns no updates when no retrievers are configured."""
    node = _make_node([])

    result = await node.execute(RAGState(query="q", query_domains=["NeuroML"]))
    logger.info(f"result with no retrievers: {result}")

    assert result == {}


async def test_execute_skips_undefined_domain():
    """The 'undefined' domain is skipped and yields no references."""
    node = _make_node([FakeRetriever([(_doc("content"), 1.0)])])

    state = RAGState(query="q", query_domains=["undefined"], retrieval_query="q")
    result = await node.execute(state)
    logger.info(f"result with undefined domain: {result}")

    assert "reference_material" in result
    assert result["reference_material"] == {}
