#!/usr/bin/env python3
"""
Retriever store configuration models

File: klea_utils/stores/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any

from pydantic import BaseModel


class StoreInfo(BaseModel):
    """Information about a single store used by a retriever manager.

    ``default_k``, ``k_max``, and ``k_inc`` configure retrieval depth per
    store.  When left ``None`` they fall back to the global values set on
    the retriever manager, so stores that do not need tuning inherit the
    graph-wide defaults.

    ``loaded_object`` holds the lazily-instantiated retriever object for
    the store (e.g. a LangChain VectorStore or BM25Retriever).
    """

    name: str
    path: str
    default_k: int | None = None
    k_max: int | None = None
    k_inc: int | None = None
    loaded_object: Any | None = None


class VectorStoreInfo(StoreInfo):
    """Information about a single vector store."""


class BM25StoreInfo(StoreInfo):
    """Information about a single BM25 store.

    ``path`` points to the pickled document corpus that the
    ``BM25RetrieverManager`` loads to build its keyword index.
    """


class PerDomainConfig(BaseModel):
    """Configuration for a single domain."""

    vector_stores: list[VectorStoreInfo] = []
    bm25_stores: list[BM25StoreInfo] = []


class RetrieverConfig(BaseModel):
    """Top-level retriever configuration.

    Holds the per-domain store configuration for all retriever managers
    (vector stores and BM25 stores).
    """

    domains: dict[str, PerDomainConfig]
