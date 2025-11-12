#!/usr/bin/env python3
"""
Smoke tests

File: test_rag.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest
from langchain_core.messages import AIMessage
from neuroml_ai.rag import NML_RAG
from neuroml_ai.schemas import AgentState


class TestRAG(unittest.TestCase):
    """Smoke tests for RAG"""

    def setUp(self):
        """Set up for tests
        :returns: TODO

        """
        self.nml_ai = NML_RAG()
        self.nml_ai.setup()
        self.state = AgentState()

    def test_retrieval(self):
        """Test retrieval"""
        self.nml_ai._load_vector_stores()
        self.state.query = "NeuroML community"
        # uses previous message for tool call id
        self.state.messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "_retrieve_docs",
                        "args": {"query": "NeuroML community"},
                        "id": "tool_call_id",
                        "type": "tool_call",
                    }
                ],
            )
        )
        self.nml_ai._retrieve_docs(self.state)


if __name__ == "__main__":
    unittest.main()
