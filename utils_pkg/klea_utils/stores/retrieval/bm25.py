#!/usr/bin/env python3
"""
BM25 keyword retriever manager

File: klea_utils/stores/retrieval/bm25.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import pickle
from pathlib import Path
from typing import Any, override

from langchain_core.documents import Document

from klea_utils.stores.retrieval.base import BaseKleaRetriever

from ..config import PerDomainConfig, RetrieverConfig


class BM25RetrieverManager(BaseKleaRetriever):
    """Manages domain-specific BM25 keyword stores.

    Each BM25 store is a pickled corpus of chunked documents written by
    :class:`~klea_utils.stores.ingestion.StoresBuilder.write_bm25_store`.
    Stores are loaded lazily per domain: the corpus is unpickled and used
    to build a :class:`langchain_community.retrievers.BM25Retriever`, which
    is queried with BM25 keyword scoring.

    A store whose corpus file is missing is skipped with a warning, so a
    misconfigured domain degrades gracefully instead of failing retrieval.

    Scalability note: this is a pure-Python in-memory index
    (``rank_bm25``).  Building and querying stay fast to well over
    ~100k chunks, but memory grows roughly with the total number of
    unique terms (one Python dict entry per term per chunk, ~100-150
    bytes each), so a single collection becomes heavy in the ~50-100k
    chunk range.  That is far beyond current corpora, but if large
    platformed deployments are ever planned, consider a proper keyword
    backend (e.g. Elasticsearch, Qdrant sparse vectors, or Postgres
    full-text search) instead.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        logger: logging.Logger,
        default_k: int = 5,
        k_max: int = 10,
        k_inc: int = 1,
    ):
        """Initialise the BM25 retriever manager.

        :param config: Retriever configuration for all domains
        :param logger: Logger instance (injected from orchestrator)
        :param default_k: Fallback number of documents to retrieve
        :param k_max: Fallback maximum number of documents to retrieve
        :param k_inc: Fallback amount to increase ``k`` by per ``inc_k``
        """
        super().__init__(
            config=config,
            logger=logger,
            default_k=default_k,
            k_max=k_max,
            k_inc=k_inc,
        )

    @override
    def _stores_of(self, domain: PerDomainConfig) -> list[Any]:
        """Return the BM25 stores configured for *domain*."""
        return domain.bm25_stores

    @override
    def _instantiate_store(self, path: str, name: str):
        """Load a BM25 corpus pickle and build a BM25Retriever from it.

        :param path: Path to the pickled document corpus
        :param name: Store name from the configuration
        :returns: A :class:`langchain_community.retrievers.BM25Retriever`,
            or ``None`` if the corpus file is missing
        """
        # Lazy: importing langchain_community pulls in the whole integration
        # package, which is heavy.  Only needed when a BM25 store is actually
        # configured, so defer the import to first load.
        from langchain_community.retrievers import BM25Retriever

        corpus_path = Path(path)
        if not corpus_path.is_file():
            self.logger.warning(
                f"BM25 corpus not found, skipping store '{name}': {corpus_path}"
            )
            return None

        with open(corpus_path, "rb") as f:
            docs = pickle.load(f)
        self.logger.debug(f"Loaded {len(docs)} chunks for BM25 store '{name}'")
        return BM25Retriever.from_documents(docs)

    @override
    def _retrieve_from_store(
        self, store: Any, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Run BM25 keyword search on a single store.

        :param store: Loaded BM25 store to query
        :param query: User query string
        :param k: Number of documents to retrieve from this store
        :returns: List of (document, relevance_score) tuples.  Documents
            with a non-positive score (no term overlap with the query)
            are dropped.
        """
        retriever = store.loaded_object
        processed = retriever.preprocess_func(query)
        top_docs = retriever.vectorizer.get_top_n(processed, retriever.docs, n=k)
        scores = retriever.vectorizer.get_scores(processed)
        # get_top_n returns the same Document objects, so map scores by id.
        score_by_id = {id(doc): score for doc, score in zip(retriever.docs, scores)}

        result = [(doc, score_by_id[id(doc)]) for doc in top_docs]
        return [(doc, score) for doc, score in result if score > 0]
