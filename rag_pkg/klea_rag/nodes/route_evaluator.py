#!/usr/bin/env python3
"""
Route evaluator node

File: rag_pkg/klea_rag/nodes/route_evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from klea_utils.nodes.abstract import (
    AbstractRouterNode,
    NodeStreamData,
    NodeStreamEvent,
)
from klea_utils.stores.retrieval.base import BaseKleaRetriever

from klea_rag.schemas import RAGState


class RouteEvaluator(AbstractRouterNode):
    """Route based on Evaluator node results"""

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        retrievers: list[BaseKleaRetriever] | None = None,
        max_retrieval_attempts: int = 2,
        max_rewrite_attempts: int = 1,
        fallback_to_training_data: bool = False,
    ):
        """Initialise the evaluator node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param retrievers: Retrievers whose k is incremented/reset when
            routing between retrieval attempts
        :param max_retrieval_attempts: Max retrieval query modifications
        :param max_rewrite_attempts: Max answer rewrites
        :param fallback_to_training_data: Whether to fall back to LLM training data
        """
        super().__init__(logger, label)
        self.retrievers = retrievers or []
        self.max_retrieval_attempts = max_retrieval_attempts
        self.max_rewrite_attempts = max_rewrite_attempts
        self.fallback_to_training_data = fallback_to_training_data

    def execute(self, state: RAGState):
        """Route based on state, set by evaluator node."""
        self.write_custom_stream({"type": "progress", "node": self.label})
        self.logger.debug(f"{state =}")
        resp = state.text_response_eval
        next_step = resp.next_step

        # Determine route
        route = None
        if next_step == "continue" and (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and resp.relevance >= 0.5
            and resp.groundedness >= 0.5
            and resp.coherence >= 0.5
            and resp.conciseness >= 0.5
        ):
            if self.retrievers:
                for retriever in self.retrievers:
                    retriever.reset_k()
            self.logger.debug("returning: continue")
            route = "continue"
        elif state.retrieval_attempts < self.max_retrieval_attempts and (
            next_step == "modify_query" or resp.coverage < 0.3
        ):
            self.logger.debug("returning: modify_query")
            route = "modify_query"
        elif next_step == "retrieve_more_info" or (
            resp.coverage >= 0.5 and resp.confidence < 0.5
        ):
            # there are no retrievers, and no more information to retrieve
            if not self.retrievers:
                route = "continue"
            # limit what max k we can have, otherwise, we end up pulling the
            # whole store..
            elif any(retriever.inc_k() for retriever in self.retrievers):
                self.logger.debug("returning: retrieve_more_info")
                route = "retrieve_more_info"
            else:
                # we are already at max context, so we need to modify the query
                # to get a better result if possible
                if state.retrieval_attempts < self.max_retrieval_attempts:
                    self.logger.debug("returning: modify_query")
                    route = "modify_query"
                # if we've already modified query, fallback to training data if
                # possible, otherwise ask for clarification
                else:
                    if self.fallback_to_training_data:
                        self.logger.debug("returning: fallback")
                        route = "fallback"
                    else:
                        self.logger.debug("returning: undefined")
                        route = "undefined"
        elif state.rewrite_attempts < self.max_rewrite_attempts and (
            next_step == "rewrite_answer"
            or (
                resp.coverage >= 0.5
                and resp.confidence >= 0.5
                and (
                    resp.relevance < 0.5
                    and resp.groundedness < 0.5
                    and resp.coherence < 0.5
                    and resp.conciseness < 0.5
                )
            )
        ):
            self.logger.debug("returning: rewrite_answer")
            route = "rewrite_answer"
        # all other cases: fallback to training data if enabled, otherwise ask for clarification
        else:
            if self.fallback_to_training_data:
                self.logger.debug("returning: fallback")
                route = "fallback"
            else:
                self.logger.debug("returning: undefined")
                route = "undefined"

        # Emit info event with routing decision
        info_data = NodeStreamData(
            heading="Route Evaluation",
            summary=f"Routing decision: {route}",
            details={
                "route": route,
                "next_step": next_step,
                "retrieval_attempts": state.retrieval_attempts,
                "rewrite_attempts": state.rewrite_attempts,
            },
        )
        info_event = NodeStreamEvent(type="info", node=self.label, data=info_data)
        self.write_custom_stream(info_event.model_dump())

        # Emit debug event with routing context
        debug_details = info_data.details.copy()
        debug_details["thresholds"] = {
            "max_retrieval_attempts": self.max_retrieval_attempts,
            "max_rewrite_attempts": self.max_rewrite_attempts,
            "fallback_to_training_data": self.fallback_to_training_data,
        }
        debug_data = NodeStreamData(
            heading=info_data.heading, summary=info_data.summary, details=debug_details
        )
        debug_event = NodeStreamEvent(type="debug", node=self.label, data=debug_data)
        self.write_custom_stream(debug_event.model_dump())

        return route
