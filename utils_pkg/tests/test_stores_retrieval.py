#!/usr/bin/env python3
"""
Test vector store related code.

File: tests/test_stores_retrieval.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

# TODO: Add tests for Qdrant and PGVector backends.
# Currently only Chroma is exercised here.

import json
import logging
import os
import unittest

import pytest
from klea_utils.stores.config import (
    PerDomainConfig,
    RetrieverConfig,
    VectorStoreInfo,
)
from klea_utils.stores.retrieval.vs import VSRetriever
from ollama import ResponseError


class FakeStore:
    """Fake vector store that records the k used in similarity searches."""

    def __init__(self):
        self.calls = []

    def similarity_search_with_relevance_scores(self, query, k, score_threshold):
        self.calls.append({"query": query, "k": k, "score_threshold": score_threshold})
        return []


class TestStores(unittest.TestCase):
    """Docstring for TestStores."""

    def _make_retriever(self) -> VSRetriever:
        """Build a retriever with a per-store configured and a fallback store."""
        config = RetrieverConfig(
            domains={
                "NeuroML": PerDomainConfig(
                    vector_stores=[
                        VectorStoreInfo(
                            name="big",
                            path="chroma:/fake/big",
                            default_k=2,
                            k_max=4,
                            k_inc=2,
                        ),
                        VectorStoreInfo(name="small", path="chroma:/fake/small"),
                    ]
                )
            }
        )
        retriever = VSRetriever(
            config=config,
            logger=logging.getLogger("test_stores"),
            embedding_model="dummy",
            default_k=5,
            k_max=10,
            k_inc=1,
        )
        retriever.embeddings = object()
        return retriever

    def test_per_store_resolution(self):
        """Per-store k settings resolve with a fall back to the global values."""
        retriever = self._make_retriever()
        big, small = retriever.config.domains["NeuroML"].vector_stores

        # "big" has its own per-store settings
        self.assertEqual(retriever._default_k_for(big), 2)
        self.assertEqual(retriever._k_max_for(big), 4)
        self.assertEqual(retriever._k_inc_for(big), 2)

        # "small" falls back to the global values
        self.assertEqual(retriever._default_k_for(small), 5)
        self.assertEqual(retriever._k_max_for(small), 10)
        self.assertEqual(retriever._k_inc_for(small), 1)

    def test_retrieve_uses_per_store_k(self):
        """retrieve() passes each store its own current k."""
        retriever = self._make_retriever()
        big, small = retriever.config.domains["NeuroML"].vector_stores
        big_fake = FakeStore()
        small_fake = FakeStore()
        big.loaded_object = big_fake
        small.loaded_object = small_fake

        retriever.retrieve("NeuroML", "some query")

        self.assertEqual(big_fake.calls[0]["k"], 2)
        self.assertEqual(small_fake.calls[0]["k"], 5)
        self.assertEqual(big_fake.calls[0]["score_threshold"], 0.15)

    def test_inc_k_loaded_only_and_capped(self):
        """inc_k() only touches loaded stores, capped per-store at k_max."""
        retriever = self._make_retriever()
        big, small = retriever.config.domains["NeuroML"].vector_stores

        # only load "big"; "small" stays unloaded
        big.loaded_object = FakeStore()

        # big: 2 -> 4 (k_inc=2), returns True
        self.assertTrue(retriever.inc_k())
        self.assertEqual(retriever._current_k("NeuroML", big), 4)
        # small is not loaded, so it is not incremented
        self.assertNotIn(("NeuroML", "small"), retriever._k)

        # big is at k_max=4 and small is still unloaded, so nothing increments
        self.assertFalse(retriever.inc_k())
        self.assertEqual(retriever._current_k("NeuroML", big), 4)

        # load small; inc_k() now increments it 5 -> 6, capped by its own k_max
        small.loaded_object = FakeStore()
        self.assertTrue(retriever.inc_k())
        self.assertEqual(retriever._current_k("NeuroML", small), 6)
        self.assertEqual(retriever._current_k("NeuroML", big), 4)

    def test_reset_k(self):
        """reset_k() restores loaded stores to their per-store defaults."""
        retriever = self._make_retriever()
        big, small = retriever.config.domains["NeuroML"].vector_stores
        big.loaded_object = FakeStore()
        small.loaded_object = FakeStore()

        # increment both: big 2 -> 4, small 5 -> 6
        retriever.inc_k()
        self.assertEqual(retriever._current_k("NeuroML", big), 4)
        self.assertEqual(retriever._current_k("NeuroML", small), 6)

        retriever.reset_k()
        self.assertEqual(retriever._current_k("NeuroML", big), 2)
        self.assertEqual(retriever._current_k("NeuroML", small), 5)

    def test_retrieval(self):
        """Test retrieval"""
        try:
            vs_config_file = os.environ.get("VS_TEST_CONFIG", None)
            assert vs_config_file
            with open(vs_config_file, "r") as f:
                config = json.load(f)
            print(config)
            retriever_config = RetrieverConfig(**config)

            logger = logging.getLogger("test_stores")
            stores = VSRetriever(
                config=retriever_config,
                logger=logger,
                embedding_model="ollama:bge-m3:latest",
            )
            stores.setup()
            stores.retrieve("NeuroML", "NeuroML community")
        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))


if __name__ == "__main__":
    unittest.main()
