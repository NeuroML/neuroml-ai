#!/usr/bin/env python3
"""
Vector store configuration models

File: klea_utils/stores/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any

from pydantic import BaseModel


class VectorStoreInfo(BaseModel):
    """Information about a single vector store.

    ``default_k``, ``k_max``, and ``k_inc`` configure retrieval depth per
    store.  When left ``None`` they fall back to the global values set on
    the ``VSRetriever``, so stores that do not need tuning inherit the
    graph-wide defaults.
    """

    name: str
    path: str
    default_k: int | None = None
    k_max: int | None = None
    k_inc: int | None = None
    loaded_object: Any | None = None


class PerDomainConfig(BaseModel):
    """Configuration for a single domain."""

    vector_stores: list[VectorStoreInfo]


class VectorStoresConfig(BaseModel):
    """Top-level vector stores configuration."""

    domains: dict[str, PerDomainConfig]
