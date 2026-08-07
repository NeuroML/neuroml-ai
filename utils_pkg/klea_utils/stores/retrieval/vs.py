#!/usr/bin/env python3
"""
Vector store retriever manager

File: klea_utils/stores/retrieval/vs.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, override

from langchain_core.documents import Document

from klea_utils.llm import setup_embedding
from klea_utils.stores.retrieval.base import BaseKleaRetriever

from ..config import PerDomainConfig, RetrieverConfig
from ..utils import instantiate_vector_store


class VSRetriever(BaseKleaRetriever):
    """Manages domain-specific vector stores.

    Loads vector stores on demand per domain and provides similarity search
    retrieval across multiple stores within a domain.

    Store paths use a URI-style scheme prefix to identify the backend:

    - ``chroma:/path/to/dir``  ---  ChromaDB (persistent, local disk)
    - ``qdrant:http://host:port``  ---  Qdrant (remote HTTP)
    - ``pgvector:postgresql://host/db``  ---  PGVector (PostgreSQL)
    """

    def __init__(
        self,
        config: RetrieverConfig,
        logger: logging.Logger,
        embedding_model: str,
        default_k: int = 5,
        k_max: int = 10,
        k_inc: int = 1,
    ):
        """Initialise vector stores manager.

        ``default_k``, ``k_max``, and ``k_inc`` are the graph-wide fallback
        values used by stores that do not define their own per-store
        settings in the config.

        :param config: Retriever configuration for all domains
        :param logger: Logger instance (injected from orchestrator)
        :param embedding_model: Embedding model identifier for retrieval
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
        self.sim_thresh = 0.15
        self.embeddings = None
        self.embedding_model = embedding_model

    @override
    def setup(self) -> None:
        """Initialise embedding model."""
        assert self.embedding_model

        self.embeddings = setup_embedding(self.embedding_model, self.logger)
        assert self.embeddings

    @override
    def _assert_ready(self) -> None:
        """Stores can only be loaded once the embedding model is ready."""
        assert self.embeddings

    @override
    def _stores_of(self, domain: PerDomainConfig) -> list[Any]:
        """Return the vector stores configured for *domain*."""
        return domain.vector_stores

    @override
    def _instantiate_store(self, path: str, name: str):
        """Instantiate a vector store based on the URI scheme in path.

        :param path: URI-style string with scheme prefix
        :param name: Collection name for the vector store
        :returns: Instantiated LangChain VectorStore
        """
        return instantiate_vector_store(path, name, self.embeddings, self.logger)

    @override
    def _retrieve_from_store(
        self, store: Any, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Run similarity search on a single vector store.

        :param store: Loaded vector store to query
        :param query: User query string
        :param k: Number of documents to retrieve from this store
        :returns: List of (document, relevance_score) tuples
        """
        data = store.loaded_object.similarity_search_with_relevance_scores(
            query,
            k=k,
            score_threshold=self.sim_thresh,
        )
        self.logger.debug(f"{data =}")
        if len(data) == 0:
            self.logger.warning(
                f"No data retrieved. Check VS is correctly populated and that "
                f"the collection name is correct ({store.name})"
            )
        return data
