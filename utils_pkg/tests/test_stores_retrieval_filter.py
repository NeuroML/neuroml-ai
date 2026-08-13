#!/usr/bin/env python3
"""
Test metadata filtering threaded through the retrievers.

File: utils_pkg/tests/test_stores_retrieval_filter.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import pickle

import pytest
from klea_utils.stores.filters import translate_metadata_filter
from klea_utils.stores.retrieval.bm25 import BM25RetrieverManager
from klea_utils.stores.retrieval.vs import VSRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# translate_metadata_filter dispatch
# ----------------------------------------------------------------------


def test_translate_chroma_filter():
    assert translate_metadata_filter("chroma:/data/store", {"journal": "Nature"}) == {
        "journal": {"$eq": "Nature"}
    }


def test_translate_qdrant_filter():
    pytest.importorskip("qdrant_client")
    from qdrant_client import models

    f = translate_metadata_filter(
        "qdrant:http://localhost:6333", {"year": {"$gte": 2020}}
    )
    assert isinstance(f, models.Filter)


def test_translate_pgvector_contains():
    assert translate_metadata_filter(
        "pgvector:postgresql://localhost/db", {"authors": {"$contains": "Magee"}}
    ) == {"authors": {"$like": "%Magee%"}}


def test_translate_unknown_scheme():
    with pytest.raises(ValueError):
        translate_metadata_filter(
            "elasticsearch:http://localhost:9200", {"year": {"$gte": 2020}}
        )


def test_translate_missing_scheme():
    with pytest.raises(ValueError):
        translate_metadata_filter("no-scheme-path", {"year": {"$gte": 2020}})


# ----------------------------------------------------------------------
# VSRetriever: filter passed into the native similarity search
# ----------------------------------------------------------------------


class _RecordingLoadedObject:
    """Records the args passed to the similarity search."""

    def __init__(self, store):
        self._store = store

    def similarity_search_with_relevance_scores(self, *args, **kwargs):
        self._store.calls.append({"args": args, "kwargs": kwargs})
        return self._store.results


class _FakeStore:
    """Minimal store object: name, URI path, and a recording loaded object."""

    def __init__(self, name, path, results=None):
        self.name = name
        self.path = path
        self.results = results or []
        self.calls = []
        self.loaded_object = _RecordingLoadedObject(self)


def _make_vs_retriever():
    vs = object.__new__(VSRetriever)
    vs.logger = logging.getLogger("test_retrieval_filter")
    vs.sim_thresh = 0.15
    return vs


def test_vs_passes_native_filter_for_chroma():
    store = _FakeStore("test", "chroma:/data/store")
    vs = _make_vs_retriever()
    vs._retrieve_from_store(store, "query", 5, {"authors": {"$contains": "Magee"}})
    kwargs = store.calls[0]["kwargs"]
    assert kwargs["filter"] == {"authors": {"$contains": "Magee"}}
    assert kwargs["k"] == 5


def test_vs_passes_native_filter_for_qdrant():
    pytest.importorskip("qdrant_client")
    from qdrant_client import models

    store = _FakeStore("test", "qdrant:http://localhost:6333")
    vs = _make_vs_retriever()
    vs._retrieve_from_store(store, "query", 5, {"year": {"$gte": 2020}})
    assert isinstance(store.calls[0]["kwargs"]["filter"], models.Filter)


def test_vs_passes_native_filter_for_pgvector():
    store = _FakeStore("test", "pgvector:postgresql://localhost/db")
    vs = _make_vs_retriever()
    vs._retrieve_from_store(store, "query", 5, {"authors": {"$contains": "Magee"}})
    assert store.calls[0]["kwargs"]["filter"] == {"authors": {"$like": "%Magee%"}}


def test_vs_no_filter_no_filter_kwarg():
    store = _FakeStore("test", "chroma:/data/store")
    vs = _make_vs_retriever()
    vs._retrieve_from_store(store, "query", 5)
    assert "filter" not in store.calls[0]["kwargs"]


# ----------------------------------------------------------------------
# BM25RetrieverManager: post-filtering with a margin
# ----------------------------------------------------------------------


def _write_corpus(tmp_path):
    docs = [
        Document(
            page_content="motor cortex model of motor learning",
            metadata={"authors": ["Magee"], "journal": "Nature", "year": 2020},
        ),
        Document(
            page_content="motor cortex stimulation for rehabilitation",
            metadata={"authors": ["Jones"], "journal": "eLife", "year": 2024},
        ),
        Document(
            page_content="hippocampal place cells and spatial navigation",
            metadata={"authors": ["Smith"], "journal": "Neuron", "year": 2022},
        ),
    ]
    path = tmp_path / "corpus.pkl"
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    return str(path)


def _make_bm25_manager():
    mgr = object.__new__(BM25RetrieverManager)
    mgr.logger = logging.getLogger("test_retrieval_filter")
    return mgr


class _FakeBM25Store:
    """Mimics the config store object whose ``loaded_object`` is set on load."""

    def __init__(self, loaded_object):
        self.loaded_object = loaded_object


def _load_bm25_store(tmp_path):
    mgr = _make_bm25_manager()
    retriever = mgr._instantiate_store(_write_corpus(tmp_path), "test")
    assert retriever is not None
    return mgr, _FakeBM25Store(retriever)


def test_bm25_post_filters_by_metadata(tmp_path):
    mgr, store = _load_bm25_store(tmp_path)
    results = mgr._retrieve_from_store(
        store, "motor cortex", k=5, metadata_filter={"authors": {"$contains": "Magee"}}
    )
    logger.info(f"filtered authors: {[d.metadata['authors'] for d, _ in results]}")
    assert [d.metadata["authors"] for d, _ in results] == [["Magee"]]


def test_bm25_filter_caps_at_k(tmp_path):
    mgr, store = _load_bm25_store(tmp_path)
    results = mgr._retrieve_from_store(
        store, "motor cortex", k=1, metadata_filter={"authors": {"$contains": "Magee"}}
    )
    assert len(results) == 1


def test_bm25_no_filter_returns_all(tmp_path):
    mgr, store = _load_bm25_store(tmp_path)
    results = mgr._retrieve_from_store(store, "motor cortex", k=5)
    assert len(results) == 2
