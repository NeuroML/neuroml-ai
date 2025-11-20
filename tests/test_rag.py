#!/usr/bin/env python3
"""
Smoke tests

File: test_rag.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest
from neuroml_ai.rag import NML_RAG


class TestRAG(unittest.TestCase):
    """Smoke tests for RAG"""

    def setUp(self):
        """Set up for tests
        :returns: TODO

        """
        self.nml_ai = NML_RAG(chat_model="ollama:qwen3:0.6b", embedding_model="ollama:bge-m3")
        self.nml_ai.setup()

    def test_retrieval(self):
        """Test retrieval"""
        self.nml_ai._load_vector_stores()
        self.nml_ai._retrieve_docs("NeuroML community")


if __name__ == "__main__":
    unittest.main()
