#!/usr/bin/env python3
"""
Vector stores retrieval manager

File: klea_utils/stores/retrieval.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from langchain_core.documents import Document

from ..llm import setup_embedding
from .config import VectorStoreInfo, VectorStoresConfig
from .utils import instantiate_vector_store


class VSRetriever:
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
        vs_config: VectorStoresConfig,
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

        :param logger: Logger instance (injected from orchestrator)
        :param embedding_model: Embedding model identifier for retrieval
        :param default_k: Fallback number of documents to retrieve
        :param k_max: Fallback maximum number of documents to retrieve
        :param k_inc: Fallback amount to increase ``k`` by per ``inc_k``
        """
        self.default_k = default_k
        self.k_max = k_max
        self.k_inc = k_inc
        # Current retrieval depth (k) per store, keyed by (domain, store name).
        # Seeded lazily from each store's default_k on first use; mutated by
        # inc_k()/reset_k().  Keeping k per store lets each store be tuned
        # independently even though retrieval is driven by a single graph-wide
        # routing decision.
        self._k: dict[tuple[str, str], int] = {}
        self.sim_thresh = 0.15
        self.embeddings = None
        self.embedding_model = embedding_model
        self.vs_config: VectorStoresConfig = vs_config
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")

    def setup(self) -> None:
        """Initialise embedding model."""
        assert self.embedding_model

        self.embeddings = setup_embedding(self.embedding_model, self.logger)
        assert self.embeddings

    def _default_k_for(self, store: VectorStoreInfo) -> int:
        """Return the default k for a store, falling back to the global value."""
        return store.default_k if store.default_k is not None else self.default_k

    def _k_max_for(self, store: VectorStoreInfo) -> int:
        """Return the k cap for a store, falling back to the global value."""
        return store.k_max if store.k_max is not None else self.k_max

    def _k_inc_for(self, store: VectorStoreInfo) -> int:
        """Return the k increment for a store, falling back to the global value."""
        return store.k_inc if store.k_inc is not None else self.k_inc

    def _current_k(self, domain_name: str, store: VectorStoreInfo) -> int:
        """Return the current k for a store, seeding it from its default on first use."""
        key = (domain_name, store.name)
        if key not in self._k:
            self._k[key] = self._default_k_for(store)
        return self._k[key]

    def _loaded_stores(self) -> list[tuple[str, VectorStoreInfo]]:
        """Return ``(domain_name, store)`` pairs for stores that are loaded."""
        loaded = []
        for domain_name, domain in self.vs_config.domains.items():
            for store in domain.vector_stores:
                if store.loaded_object is not None:
                    loaded.append((domain_name, store))
        return loaded

    def inc_k(self) -> bool:
        """Increase k for all loaded stores by their per-store increment.

        Each store's k is capped by its own ``k_max``, so stores with a
        smaller cap stop being incremented sooner.  Stores that are not yet
        loaded keep their default k until they are loaded.

        :returns: True if at least one store's k was increased
        """
        incremented = False
        for domain_name, store in self._loaded_stores():
            current = self._current_k(domain_name, store)
            new_k = current + self._k_inc_for(store)
            if new_k <= self._k_max_for(store):
                self._k[(domain_name, store.name)] = new_k
                self.logger.debug(
                    f"{store.name = }\n{self._k_inc_for(store) = }\n{new_k = }"
                )
                incremented = True
        if not incremented:
            self.logger.debug("k not increased for any store")
        return incremented

    def reset_k(self) -> None:
        """Reset k for all loaded stores to their per-store default value."""
        for domain_name, store in self._loaded_stores():
            self._k[(domain_name, store.name)] = self._default_k_for(store)
            self.logger.debug(
                f"k reset to {self._default_k_for(store) = } for {store.name = }"
            )

    def load_all_stores(self) -> None:
        """Load all vector stores for all domains."""
        for domain_name in self.domains:
            self.load(domain_name)

    @property
    def domains(self) -> list[str]:
        """Get a list of all configured domains."""
        return list(self.vs_config.domains.keys())

    def load(self, domain_name: str) -> None:
        """Load vector stores for a domain (lazy loading).

        :param domain_name: Name of the domain to load stores for
        """
        assert self.embeddings

        domain = self.vs_config.domains.get(domain_name, None)
        assert domain

        self.logger.debug(f"Got domain information: {domain}")

        stores = domain.vector_stores

        for store in stores:
            if store.loaded_object is not None:
                self.logger.debug(f"Store '{store.name}' already loaded, skipping")
                continue

            store_name = store.name
            self.logger.debug(
                f"Got store for domain {domain_name}: {store_name} ({store.path})"
            )

            store.loaded_object = self._instantiate_store(store.path, store_name)

            self.logger.debug(
                f"Finished loading vector store '{store_name}' from {store.path}"
            )

    def _instantiate_store(self, path: str, name: str):
        """Instantiate a vector store based on the URI scheme in path.

        :param path: URI-style string with scheme prefix
        :param name: Collection name for the vector store
        :returns: Instantiated LangChain VectorStore
        """
        return instantiate_vector_store(path, name, self.embeddings, self.logger)

    def retrieve(self, domain_name: str, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents from vector stores for a query.

        :param domain_name: Name of the domain to search in
        :param query: User query string
        :returns: List of (document, relevance_score) tuples
        """
        self.load(domain_name)

        domain = self.vs_config.domains.get(domain_name, None)
        assert domain
        stores = domain.vector_stores

        res = []

        for store in stores:
            assert store.loaded_object
            data = store.loaded_object.similarity_search_with_relevance_scores(
                query,
                k=self._current_k(domain_name, store),
                score_threshold=self.sim_thresh,
            )
            self.logger.debug(f"{data =}")
            if len(data) == 0:
                self.logger.warning(
                    f"No data retrieved. Check VS is correctly populated and that "
                    f"the collection name is correct ({store.name})"
                )
            res.extend(data)

        return res
