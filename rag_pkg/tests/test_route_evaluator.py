#!/usr/bin/env python3
"""
Tests for the RAG route evaluator node.

File: rag_pkg/tests/test_route_evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from klea_rag.nodes.route_evaluator import RouteEvaluator
from klea_rag.schemas import EvaluateAnswerSchema, RAGState

logger = logging.getLogger(__name__)


class FakeRetriever:
    """Minimal retriever recording k calls."""

    def __init__(self, name="fake"):
        self.name = name
        self.inc_count = 0
        self.reset_count = 0

    def inc_k(self):
        self.inc_count += 1
        return True

    def reset_k(self):
        self.reset_count += 1


def _make_router(retrievers) -> RouteEvaluator:
    router = object.__new__(RouteEvaluator)
    router.logger = logging.getLogger("test_route_evaluator")
    router.label = "Routing evaluation"
    router.retrievers = retrievers
    router.max_retrieval_attempts = 2
    router.max_rewrite_attempts = 1
    router.fallback_to_training_data = False
    router.write_custom_stream = lambda event: None
    logger.info(f"configured retrievers: {[r.name for r in retrievers]}")
    return router


def _continue_state() -> RAGState:
    """A state whose evaluator is fully satisfied."""
    return RAGState(
        query="q",
        text_response_eval=EvaluateAnswerSchema(
            confidence=0.9,
            coverage=0.9,
            relevance=0.9,
            groundedness=0.9,
            coherence=0.9,
            conciseness=0.9,
            next_step="continue",
        ),
    )


def test_continue_resets_k_on_all_retrievers():
    """Routing 'continue' resets k on every retriever."""
    r1 = FakeRetriever("vector")
    r2 = FakeRetriever("bm25")
    router = _make_router([r1, r2])

    route = router.execute(_continue_state())
    logger.info(f"route: {route} | resets: r1={r1.reset_count}, r2={r2.reset_count}")

    assert route == "continue"
    assert r1.reset_count == 1
    assert r2.reset_count == 1


def test_retrieve_more_info_increments_k_on_all_retrievers():
    """Routing 'retrieve_more_info' increments k on every retriever."""
    r1 = FakeRetriever("vector")
    r2 = FakeRetriever("bm25")
    router = _make_router([r1, r2])

    # coverage >= 0.3 keeps the modify_query branch from short-circuiting
    state = RAGState(
        query="q",
        text_response_eval=EvaluateAnswerSchema(
            coverage=0.5, confidence=0.4, next_step="retrieve_more_info"
        ),
    )
    route = router.execute(state)
    logger.info(f"route: {route} | incs: r1={r1.inc_count}, r2={r2.inc_count}")

    assert route == "retrieve_more_info"
    assert r1.inc_count == 1
    assert r2.inc_count == 1


def test_retrieve_more_info_without_retrievers_continues():
    """Without retrievers, 'retrieve_more_info' routes to 'continue'."""
    router = _make_router([])

    state = RAGState(
        query="q",
        text_response_eval=EvaluateAnswerSchema(
            coverage=0.5, confidence=0.4, next_step="retrieve_more_info"
        ),
    )
    route = router.execute(state)
    logger.info(f"route with no retrievers: {route}")

    assert route == "continue"
