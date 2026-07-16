#!/usr/bin/env python3
"""
Retrieve information node

File: rag_pkg/klea_rag/nodes/retrieve_info.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.nodes.abstract import (
    AbstractLangGraphNode,
    NodeStreamData,
    NodeStreamEvent,
)
from klea_utils.stores.retrieval import VSRetriever

from klea_rag.schemas import RAGState


class RetrieveInfoNode(AbstractLangGraphNode[RAGState, Dict[str, Any]]):
    """Retrieve reference material from vector stores.

    Queries the vector stores for all domains in the query_domains list using
    the same retrieval query, ranks results by relevance score, and keeps the
    top N references for each domain. Optionally increments k when asked to
    retrieve more info.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        stores: VSRetriever | None,
        num_refs_max: int = 10,
    ):
        """Initialise the retrieval node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param stores: VSRetriever instance for retrieval (None skips retrieval)
        :param num_refs_max: Maximum number of references to keep per domain
        """
        super().__init__(logger, label)
        self.stores = stores
        self.num_refs_max = num_refs_max

    @override
    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve and rank reference material."""
        if self.stores is None:
            self.logger.debug("No vector stores configured, skipping retrieval")
            return {}

        self.write_custom_stream({"type": "progress", "node": self.label})

        reference_material = state.reference_material
        cleaned_query = state.retrieval_query

        self.logger.debug(f"retrieval query: {cleaned_query}")

        # Check if evaluator requested more info
        if state.text_response_eval.next_step == "retrieve_more_info":
            self.stores.inc_k()

        # Retrieve from vector stores for all domains
        for domain_name in state.query_domains:
            # Skip undefined domain
            if domain_name == "undefined":
                continue

            res = self.stores.retrieve(domain_name=domain_name, query=cleaned_query)

            # Rank by relevance score, keep top N
            sorted_res = sorted(res, key=lambda tup: tup[1], reverse=True)
            new_ref = {domain_name: sorted_res[: self.num_refs_max]}

            reference_material.update(new_ref)

        self.logger.debug(f"{reference_material =}")

        # Emit info event
        per_domain_counts = {
            domain: len(docs) for domain, docs in reference_material.items()
        }
        total_docs = sum(per_domain_counts.values())
        info_data = NodeStreamData(
            summary=f"Retrieved {total_docs} documents from {len(per_domain_counts)} domains",
            details={"per_domain_counts": per_domain_counts},
        )
        info_event = NodeStreamEvent(type="info", node=self.label, data=info_data)
        self.write_custom_stream(info_event.model_dump())

        # Emit debug event
        debug_details = info_data.details.copy()
        debug_details["reference_material"] = {
            domain: [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in docs
            ]
            for domain, docs in reference_material.items()
        }
        debug_data = NodeStreamData(summary=info_data.summary, details=debug_details)
        debug_event = NodeStreamEvent(type="debug", node=self.label, data=debug_data)
        self.write_custom_stream(debug_event.model_dump())

        return {"reference_material": reference_material}
