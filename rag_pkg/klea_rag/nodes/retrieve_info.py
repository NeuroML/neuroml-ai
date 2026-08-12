#!/usr/bin/env python3
"""
Retrieve information node

File: rag_pkg/klea_rag/nodes/retrieve_info.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from collections.abc import Hashable
from typing import Any, override

from klea_utils.nodes.abstract import (
    AbstractLangGraphNode,
    NodeStreamData,
    NodeStreamEvent,
)
from klea_utils.stores.retrieval.base import BaseKleaRetriever
from klea_utils.stores.utils import format_source_scores, normalize_text, rrf_merge

from klea_rag.schemas import RAGState


def _format_scores(doc: Any, score: float, precision: int = 2) -> str:
    """Format a document's scores for the reference material display.

    Shows the original per-source scores when present (from the RRF merge),
    otherwise falls back to the single relevance score.

    :param doc: Document to format
    :param score: Relevance score for *doc*
    :param precision: Number of decimal places
    :returns: Display string, e.g. ``[vector store 0.87, BM25 3.21]``
    """
    source_scores = format_source_scores(doc, precision)
    if source_scores:
        return f"[{source_scores}]"
    return f"[{score:.{precision}f}]"


class RetrieveInfoNode(AbstractLangGraphNode[RAGState, dict[str, Any]]):
    """Retrieve reference material from the configured retrievers.

    Queries all retrievers (vector stores, BM25 stores) for the domains in
    the ``query_domains`` list using the same retrieval query, fuses the
    results with Reciprocal Rank Fusion, and keeps the top N references for
    each domain.  Optionally increments k when asked to retrieve more info.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        retrievers: list[BaseKleaRetriever] | None = None,
        num_refs_max: int = 10,
    ):
        """Initialise the retrieval node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param retrievers: Retrievers to query (empty list skips retrieval)
        :param num_refs_max: Maximum number of references to keep per domain
        """
        super().__init__(logger, label)
        self.retrievers = retrievers or []
        self.num_refs_max = num_refs_max

    @override
    async def execute(self, state: RAGState) -> dict[str, Any]:
        """Retrieve and rank reference material."""
        if not self.retrievers:
            self.logger.debug("No retrievers configured, skipping retrieval")
            return {}

        self.write_custom_stream({"type": "progress", "node": self.label})

        reference_material = state.reference_material
        # Apply the same normalization used at indexing time so query and
        # stored chunks share an identical plain-text form (see
        # klea_utils.stores.utils.normalize_text).  The query is LLM-generated
        # so artifacts are rare, but this makes the invariant explicit.
        raw_query = state.retrieval_query
        cleaned_query = normalize_text(raw_query)
        self.logger.debug(f"{raw_query = }\n{cleaned_query = }")

        # Check if evaluator requested more info
        if state.text_response_eval.next_step == "retrieve_more_info":
            for retriever in self.retrievers:
                retriever.inc_k()

        # Retrieve from all retrievers for all domains
        for domain_name in state.query_domains:
            # Skip undefined domain
            if domain_name == "undefined":
                continue

            result_sets = [
                (
                    retriever.source_label,
                    retriever.retrieve(domain_name=domain_name, query=cleaned_query),
                )
                for retriever in self.retrievers
            ]
            merged = rrf_merge(result_sets, num_refs_max=self.num_refs_max)
            reference_material[domain_name] = merged

        self.logger.debug(f"{reference_material =}")

        # Emit info event
        per_domain_counts = {
            domain: len(docs) for domain, docs in reference_material.items()
        }
        total_docs = sum(per_domain_counts.values())
        info_data = NodeStreamData(
            heading="Document Retrieval",
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
        debug_data = NodeStreamData(
            heading=info_data.heading, summary=info_data.summary, details=debug_details
        )
        debug_event = NodeStreamEvent(type="debug", node=self.label, data=debug_data)
        self.write_custom_stream(debug_event.model_dump())

        # Emit state event
        # URLs, file name, score, not full content
        md_lines = []
        for domain, docs in reference_material.items():
            if not docs:
                continue

            md_lines.append(f"### {domain}\n")

            seen: dict[Hashable, tuple[Any, float, list[str]]] = {}
            for doc, score in sorted(docs, key=lambda x: x[1], reverse=True):
                url_values = [v for k, v in doc.metadata.items() if k.startswith("url")]
                file_name = doc.metadata.get("file_name", "") or ""
                if url_values:
                    key: Hashable = tuple(sorted(url_values))
                    display_values = url_values
                elif file_name:
                    key = file_name
                    display_values = [file_name]
                else:
                    self.logger.warning(f"No metadata to show for {doc}")
                    continue
                if key not in seen or score > seen[key][1]:
                    seen[key] = (doc, score, display_values)
            ref_lines = [
                f"1.  {_format_scores(doc, score)}\n"
                + "\n".join(f"    - {v}" for v in values)
                for doc, score, values in sorted(
                    seen.values(), key=lambda x: x[1], reverse=True
                )
            ]
            md_lines += "\n".join(ref_lines)
            md_lines += "\n\n"

        display_md = "".join(md_lines) if md_lines else "No documents retrieved"
        status_data = NodeStreamData(
            heading="Reference Material",
            summary=f"{total_docs} docs from {len(per_domain_counts)} domains",
            display=display_md,
        )
        status_event = NodeStreamEvent(type="state", node=self.label, data=status_data)
        self.logger.debug(f"{status_event.model_dump() = }")
        self.write_custom_stream(status_event.model_dump())

        return {"reference_material": reference_material}
