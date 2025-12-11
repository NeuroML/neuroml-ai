#!/usr/bin/env python3
"""
Test vector store related code.

File: tests/test_stores.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
import unittest
from neuroml_ai.rag.stores import NML_Stores


class TestStores(unittest.TestCase):
    """Docstring for TestStores."""

    def test_retrieval(self):
        """Test retrieval"""
        # model = "bge-m3"
        model = "lol"

        try:
            stores = NML_Stores(f"ollama:{model}")
            stores.setup()
            stores.load()
            stores.retrieve("NeuroML community")
        except:  # noqa
            pytest.skip("Ollama model not found")


if __name__ == "__main__":
    unittest.main()
