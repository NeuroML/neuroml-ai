#!/usr/bin/env python3
"""
Base class for retriever managers

File: klea_utils/stores/retrieval/base.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from ..config import PerDomainConfig, RetrieverConfig


class BaseKleaRetriever(ABC):
    """Base class for domain-aware retriever managers.

    Holds the machinery common to all retrievers: lazy per-domain store
    loading, per-store retrieval depth (``k``) tracking with graph-wide
    fallbacks, and the retrieval contract
    ``retrieve(domain, query) -> list[tuple[Document, float]]``.

    Subclasses must implement:

    - :meth:`_stores_of`: the list of stores configured for a domain
    - :meth:`_instantiate_store`: build the underlying retriever object
      for a store
    - :meth:`_retrieve_from_store`: run a single store against a query

    Subclasses should set :attr:`source_label` to a human-readable name
    for the retriever type (e.g. ``"vector store"``, ``"BM25"``), used to
    label the original per-source scores preserved during fusion.
    """

    #: Human-readable name for this retriever type, used to label scores.
    source_label: str = "retriever"

    def __init__(
        self,
        config: RetrieverConfig,
        logger: logging.Logger,
        default_k: int = 5,
        k_max: int = 10,
        k_inc: int = 1,
    ):
        """Initialise the retriever manager.

        ``default_k``, ``k_max``, and ``k_inc`` are the manager-wide fallback
        values used by stores that do not define their own per-store
        settings in the config.

        :param config: Store configuration for all domains
        :param logger: Logger instance (injected from orchestrator)
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
        self.config: RetrieverConfig = config
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")

    def setup(self) -> None:
        """Hook for subclasses to initialise shared resources.

        Called once by the orchestrator before retrieval.  Subclasses that
        need no shared setup leave this as a no-op.
        """
        pass

    def _assert_ready(self) -> None:
        """Hook called at the start of :meth:`load`.

        Subclasses can assert that their required resources (e.g. an
        embedding model) are available before stores are loaded.
        """
        pass

    def _default_k_for(self, store: Any) -> int:
        """Return the default k for a store, falling back to the global value."""
        return store.default_k if store.default_k is not None else self.default_k

    def _k_max_for(self, store: Any) -> int:
        """Return the k cap for a store, falling back to the global value."""
        return store.k_max if store.k_max is not None else self.k_max

    def _k_inc_for(self, store: Any) -> int:
        """Return the k increment for a store, falling back to the global value."""
        return store.k_inc if store.k_inc is not None else self.k_inc

    def _current_k(self, domain_name: str, store: Any) -> int:
        """Return the current k for a store, seeding it from its default on first use."""
        key = (domain_name, store.name)
        if key not in self._k:
            self._k[key] = self._default_k_for(store)
        return self._k[key]

    def _loaded_stores(self) -> list[tuple[str, Any]]:
        """Return ``(domain_name, store)`` pairs for stores that are loaded."""
        loaded = []
        for domain_name, domain in self.config.domains.items():
            for store in self._stores_of(domain):
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
        """Load all stores for all domains."""
        for domain_name in self.domains:
            self.load(domain_name)

    @property
    def domains(self) -> list[str]:
        """Get a list of all configured domains."""
        return list(self.config.domains.keys())

    def load(self, domain_name: str) -> None:
        """Load stores for a domain (lazy loading).

        :param domain_name: Name of the domain to load stores for
        """
        self._assert_ready()

        domain = self.config.domains.get(domain_name, None)
        assert domain

        self.logger.debug(f"Got domain information: {domain}")

        for store in self._stores_of(domain):
            if store.loaded_object is not None:
                self.logger.debug(f"Store '{store.name}' already loaded, skipping")
                continue

            store_name = store.name
            self.logger.debug(
                f"Got store for domain {domain_name}: {store_name} ({store.path})"
            )

            store.loaded_object = self._instantiate_store(store.path, store_name)

            self.logger.debug(
                f"Finished loading store '{store_name}' from {store.path}"
            )

    def retrieve(self, domain_name: str, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents from all stores for a domain.

        :param domain_name: Name of the domain to search in
        :param query: User query string
        :returns: List of (document, relevance_score) tuples
        """
        self.load(domain_name)

        domain = self.config.domains.get(domain_name, None)
        assert domain

        res = []
        for store in self._stores_of(domain):
            if store.loaded_object is None:
                continue
            data = self._retrieve_from_store(
                store, query, self._current_k(domain_name, store)
            )
            res.extend(data)

        return res

    # ------------------------------------------------------------------
    # Abstract hooks -- subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _stores_of(self, domain: PerDomainConfig) -> list[Any]:
        """Return the list of stores configured for *domain*."""
        ...

    @abstractmethod
    def _instantiate_store(self, path: str, name: str) -> Any:
        """Instantiate the underlying retriever object for a store.

        :param path: Store path from the configuration
        :param name: Store name from the configuration
        :returns: Instantiated retriever object
        """
        ...

    @abstractmethod
    def _retrieve_from_store(
        self, store: Any, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Retrieve from a single store, returning (document, score) tuples.

        :param store: Loaded store object to query
        :param query: User query string
        :param k: Number of documents to retrieve from this store
        :returns: List of (document, relevance_score) tuples
        """
        ...
