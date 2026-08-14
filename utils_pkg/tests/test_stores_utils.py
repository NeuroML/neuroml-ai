#!/usr/bin/env python3
"""
Test store retrieval utilities.

File: utils_pkg/tests/test_stores_utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import pytest
from klea_utils.stores.utils import (
    REF_DOC_OVERHEAD,
    format_source_scores,
    instantiate_vector_store,
    rrf_merge,
    serialize_reference_material,
    truncate_reference_material,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class _FakeEmbeddings:
    """Minimal embedding object for Chroma; never called during init."""

    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 8


def test_instantiate_chroma_rejects_file_path(tmp_path):
    """A chroma location pointing at an existing file is rejected."""
    pytest.importorskip("langchain_chroma")
    store_file = tmp_path / "chroma.sqlite3"
    store_file.touch()

    with pytest.raises(FileNotFoundError):
        instantiate_vector_store(
            f"chroma:{store_file}",
            "test_collection",
            _FakeEmbeddings(),
            logger,
            create=True,
        )


def test_instantiate_chroma_creates_missing_folder(tmp_path):
    """create=True creates a missing store folder and initialises it."""
    pytest.importorskip("langchain_chroma")
    store_dir = tmp_path / "new-store"

    store = instantiate_vector_store(
        f"chroma:{store_dir}",
        "test_collection",
        _FakeEmbeddings(),
        logger,
        create=True,
    )
    logger.info(f"store: {store}")
    assert store is not None
    assert store_dir.is_dir()
    assert (store_dir / "chroma.sqlite3").is_file()


def test_instantiate_chroma_folder_must_exist_when_not_creating(tmp_path):
    """create=False requires an existing store folder."""
    pytest.importorskip("langchain_chroma")
    with pytest.raises(FileNotFoundError):
        instantiate_vector_store(
            f"chroma:{tmp_path / 'missing-store'}",
            "test_collection",
            _FakeEmbeddings(),
            logger,
        )


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


def test_rrf_merge_no_cap_returns_all():
    """With num_refs_max=None, rrf_merge keeps every fused document."""
    docs = [_doc(f"content {i}") for i in range(5)]
    results = [(doc, float(5 - i)) for i, doc in enumerate(docs)]

    merged = rrf_merge([("vector store", results)], num_refs_max=None)
    logger.info(f"num_refs_max=None, returned {len(merged)} documents")

    assert len(merged) == 5


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


def test_serialize_reference_material_shows_per_source_scores():
    """serialize_reference_material labels per-source scores for the LLM."""
    d1 = _doc("Hodgkin-Huxley action potential")
    d1.metadata["_source_scores"] = {"vector store": 0.8723, "BM25": 3.2100}

    text = serialize_reference_material({"NeuroML": [(d1, 0.0323)]})
    logger.info(f"serialized text:\n{text}")

    assert "relevance: vector store 0.8723, BM25 3.2100" in text


def test_serialize_reference_material_falls_back_to_relevance_score():
    """Untagged docs fall back to the plain relevance score."""
    d1 = _doc("plain content")

    text = serialize_reference_material({"NeuroML": [(d1, 0.42)]})
    logger.info(f"serialized text:\n{text}")

    assert "relevance score: 0.4200" in text
    assert "_source_scores" not in text


def test_serialize_reference_material_groups_chunks_by_file():
    """Multiple chunks of one file emit document metadata once."""
    d1 = Document(
        page_content="chunk one content",
        metadata={
            "file_name": "paper.pdf",
            "headings": ["Intro"],
            "authors": ["A. Sinha"],
            "year": 2020,
        },
    )
    d2 = Document(
        page_content="chunk two content",
        metadata={
            "file_name": "paper.pdf",
            "headings": ["Intro", "Methods"],
            "authors": ["A. Sinha"],
            "year": 2020,
        },
    )

    text = serialize_reference_material({"NeuroML": [(d1, 0.9), (d2, 0.7)]})
    logger.info(f"serialized text:\n{text}")

    # Document-level metadata appears once, on the source header.
    assert text.count("Metadata: authors=['A. Sinha'] | year=2020") == 1
    assert "Source document 1/1: [paper.pdf]" in text
    # Both chunks are present, numbered within the single source document.
    assert "chunk one content" in text
    assert "chunk two content" in text
    assert text.count("Chunk 1:") == 1
    assert text.count("Chunk 2:") == 1


def test_serialize_reference_material_chunk_numbering_resets_per_file():
    """Chunk numbering restarts for each source document."""
    d1 = Document(
        page_content="one",
        metadata={"file_name": "a.pdf", "headings": ["Intro"]},
    )
    d2 = Document(
        page_content="two",
        metadata={"file_name": "a.pdf", "headings": ["Intro", "Methods"]},
    )
    d3 = Document(
        page_content="three",
        metadata={"file_name": "b.pdf", "headings": ["Intro"]},
    )

    text = serialize_reference_material({"NeuroML": [(d1, 0.9), (d2, 0.7), (d3, 0.8)]})
    logger.info(f"serialized text:\n{text}")

    # Each source document numbers its own chunks from 1.
    assert "Chunk 1: Intro" in text
    assert "Chunk 2: Intro > Methods" in text
    # b.pdf only has one chunk, so it gets Chunk 1, not Chunk 3.
    assert text.count("Chunk 1:") == 2
    assert text.count("Chunk 2:") == 1


def test_serialize_reference_material_chunk_metadata_differs_inline():
    """Chunk-level metadata that differs from the file-level is inline."""
    d1 = Document(
        page_content="intro content",
        metadata={
            "file_name": "docs/index.md",
            "headings": ["Home"],
            "url": "https://example.com/",
        },
    )
    d2 = Document(
        page_content="guide content",
        metadata={
            "file_name": "docs/index.md",
            "headings": ["Home", "Guide"],
            "url": "https://example.com/",
            "url_section": "https://example.com/guide",
        },
    )

    text = serialize_reference_material({"Docs": [(d1, 0.8), (d2, 0.6)]})
    logger.info(f"serialized text:\n{text}")

    # Shared url only once (file-level); the section url inline.
    assert text.count("url=https://example.com/") == 1
    assert "url_section=https://example.com/guide" in text


def test_serialize_reference_material_different_urls_per_chunk():
    """Chunks with different urls keep their own url inline, not the file's."""
    d1 = Document(
        page_content="intro content",
        metadata={
            "file_name": "docs/index.md",
            "headings": ["Home"],
            "url": "https://example.com/home",
        },
    )
    d2 = Document(
        page_content="guide content",
        metadata={
            "file_name": "docs/index.md",
            "headings": ["Home", "Guide"],
            "url": "https://example.com/guide",
        },
    )

    text = serialize_reference_material({"Docs": [(d1, 0.8), (d2, 0.6)]})
    logger.info(f"serialized text:\n{text}")

    # Both urls present; each chunk carries its own.
    assert "url=https://example.com/home" in text
    assert "url=https://example.com/guide" in text
    assert text.count("url=https://example.com/home") == 1
    assert text.count("url=https://example.com/guide") == 1
    # Neither url was hoisted to the file level (they differ).
    assert not text.startswith("Metadata: url=")


def test_serialize_reference_material_shared_url_hoisted_once():
    """An identical url across all chunks is emitted once on the file level."""
    d1 = Document(
        page_content="one",
        metadata={
            "file_name": "paper.pdf",
            "headings": ["Intro"],
            "url": "https://doi.org/10.1/x",
        },
    )
    d2 = Document(
        page_content="two",
        metadata={
            "file_name": "paper.pdf",
            "headings": ["Intro", "Methods"],
            "url": "https://doi.org/10.1/x",
        },
    )

    text = serialize_reference_material({"NeuroML": [(d1, 0.9), (d2, 0.7)]})
    logger.info(f"serialized text:\n{text}")

    assert text.count("url=https://doi.org/10.1/x") == 1
    assert "Metadata: url=https://doi.org/10.1/x" in text


def test_serialize_reference_material_shared_fields_once_per_file():
    """Shared bibliographic fields appear once per file, never per chunk."""
    d1 = Document(
        page_content="one",
        metadata={
            "file_name": "paper.pdf",
            "headings": ["Intro"],
            "authors": ["A. Sinha"],
            "year": 2020,
        },
    )
    d2 = Document(
        page_content="two",
        metadata={
            "file_name": "paper.pdf",
            "headings": ["Intro", "Methods"],
            "authors": ["A. Sinha"],
            "year": 2020,
        },
    )

    text = serialize_reference_material({"NeuroML": [(d1, 0.9), (d2, 0.7)]})
    logger.info(f"serialized text:\n{text}")

    assert text.count("authors=['A. Sinha']") == 1
    assert text.count("year=2020") == 1
    assert "Metadata: authors=['A. Sinha'] | year=2020" in text


def test_serialize_reference_material_orders_files_by_best_score():
    """Files are ordered by their best chunk's score."""
    d1 = Document(
        page_content="a",
        metadata={"file_name": "low.pdf", "year": 2020},
    )
    d2 = Document(
        page_content="b",
        metadata={"file_name": "high.pdf", "year": 2021},
    )
    d3 = Document(
        page_content="c",
        metadata={"file_name": "high.pdf", "year": 2021},
    )

    text = serialize_reference_material({"NeuroML": [(d1, 0.3), (d2, 0.9), (d3, 0.8)]})
    logger.info(f"serialized text:\n{text}")

    assert text.index("high.pdf") < text.index("low.pdf")


def test_serialize_reference_material_no_file_name_falls_back_to_no_file():
    """Docs without a file_name still serialize under a '(no file)' group."""
    d1 = Document(page_content="lonely chunk", metadata={"year": 2020})

    text = serialize_reference_material({"NeuroML": [(d1, 0.5)]})
    logger.info(f"serialized text:\n{text}")

    assert "(no file)" in text
    assert "lonely chunk" in text


def _budgeted_refs(n_docs, content_len):
    """Build ``{domain: [(doc, score)]}`` in RRF order with known sizes."""
    return {
        "NeuroML": [(_doc("a" * content_len), float(n_docs - i)) for i in range(n_docs)]
    }


def test_truncate_reference_material_keeps_top_ranked_within_budget():
    """A tight budget keeps the top-ranked docs and stops at the budget."""
    refs = _budgeted_refs(5, 50)
    doc_size = 50 + REF_DOC_OVERHEAD  # 250

    # budget fits doc1 (250); doc2 crosses (500 > 450) so it is the crossing
    # doc, kept; doc3 and beyond are dropped
    budgeted = truncate_reference_material(refs, max_chars=450)
    logger.info(f"budget=450 -> {len(budgeted['NeuroML'])} docs")

    kept = budgeted["NeuroML"]
    assert [doc.page_content for doc, _ in kept] == ["a" * 50] * 2
    assert len(kept) < 5
    assert doc_size <= 450 < 2 * doc_size


def test_truncate_reference_material_large_budget_admits_lower_ranked():
    """A generous budget admits the lower-ranked docs too (k-increase case)."""
    refs = _budgeted_refs(5, 50)
    doc_size = 50 + REF_DOC_OVERHEAD

    budgeted = truncate_reference_material(refs, max_chars=5 * doc_size)
    logger.info(f"budget=5*doc_size -> {len(budgeted['NeuroML'])} docs")

    assert len(budgeted["NeuroML"]) == 5


def test_truncate_reference_material_keeps_oversized_single_doc():
    """A single doc larger than the whole budget is still kept."""
    refs = {"NeuroML": [(_doc("b" * 5000), 1.0)]}

    budgeted = truncate_reference_material(refs, max_chars=2000)
    logger.info(f"oversized doc kept: {len(budgeted['NeuroML'])} docs")

    assert len(budgeted["NeuroML"]) == 1


def test_truncate_reference_material_global_across_domains():
    """The budget is shared across domains: the first to hit it empties the rest."""
    refs = {
        "A": [(_doc("c" * 50), 0.9), (_doc("c" * 50), 0.8)],
        "B": [(_doc("c" * 50), 0.7)],
    }
    # doc A1 fits (250); A2 crosses the 450 budget and is kept; B has no budget
    # left and is dropped entirely
    budgeted = truncate_reference_material(refs, max_chars=450)
    logger.info(f"domain counts: A={len(budgeted['A'])}, B={len(budgeted['B'])}")

    assert len(budgeted["A"]) == 2
    assert len(budgeted["B"]) == 0
