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
from klea_utils.stores.utils import (
    normalize_text,
    rerank_by_recency,
    rrf_merge,
    truncate_reference_material,
)

from klea_rag.schemas import RAGState


def _format_scores(doc: Any, score: float, precision: int = 2) -> str:
    """Format a document's blended score for the reference material display.

    The score carried in the tuple is the final recency-blended score (see
    ``klea_utils.stores.utils.rerank_by_recency``).

    :param doc: Document (retained for a stable signature)
    :param score: Blended relevance/recency score for *doc*
    :param precision: Number of decimal places
    :returns: Display string, e.g. ``[0.87]``
    """
    return f"[{score:.{precision}f}]"


class RetrieveInfoNode(AbstractLangGraphNode[RAGState, dict[str, Any]]):
    """Retrieve reference material from the configured retrievers.

    Queries all retrievers (vector stores, BM25 stores) for the domains in
    the ``query_domains`` list using the same retrieval query, fuses the
    results with Reciprocal Rank Fusion, and keeps the references ranked
    best by RRF up to a global character budget (``max_refs_size``).
    Optionally increments k when asked to retrieve more info.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        retrievers: list[BaseKleaRetriever] | None = None,
        max_refs_size: int = 20000,
    ):
        """Initialise the retrieval node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param retrievers: Retrievers to query (empty list skips retrieval)
        :param max_refs_size: Global character budget for the reference
            material fed to the answer LLM (see
            ``klea_utils.stores.utils.truncate_reference_material``)
        """
        super().__init__(logger, label)
        self.retrievers = retrievers or []
        self.max_refs_size = max_refs_size

    @override
    async def execute(self, state: RAGState) -> dict[str, Any]:
        """Retrieve and rank reference material."""
        # Every pass through this node is one retrieval attempt (the initial
        # query retrieval, retrieve_more_info k-increases, and modify_query
        # re-retrievals), so the evaluator loop budgets against a single
        # combined counter instead of per-kind counters.
        retrieval_attempts = state.retrieval_attempts + 1

        if not self.retrievers:
            self.logger.debug("No retrievers configured, skipping retrieval")
            return {"retrieval_attempts": retrieval_attempts}

        self.write_custom_stream({"type": "progress", "node": self.label})

        reference_material = state.reference_material
        # Apply the same normalization used at indexing time so query and
        # stored chunks share an identical plain-text form (see
        # klea_utils.stores.utils.normalize_text).  The query is LLM-generated
        # so artifacts are rare, but this makes the invariant explicit.
        raw_query = state.retrieval_query.search_query
        cleaned_query = normalize_text(raw_query)
        self.logger.debug(f"{raw_query = }\n{cleaned_query = }")

        # Check if evaluator requested more info
        if state.text_response_eval.next_step == "retrieve_more_info":
            for retriever in self.retrievers:
                retriever.inc_k()

        # Retrieve from all retrievers for all domains
        metadata_filter = state.retrieval_query.to_metadata_filter()
        self.logger.debug(f"{metadata_filter = }")
        for domain_name in state.query_domains:
            # Skip undefined domain
            if domain_name == "undefined":
                continue

            result_sets = [
                (
                    retriever.source_label,
                    retriever.retrieve(
                        domain_name=domain_name,
                        query=cleaned_query,
                        metadata_filter=metadata_filter,
                    ),
                )
                for retriever in self.retrievers
            ]
            merged = rrf_merge(result_sets)
            reference_material[domain_name] = rerank_by_recency(merged)

        reference_material = truncate_reference_material(
            reference_material, max_chars=self.max_refs_size
        )

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
                url_items = [
                    (k, v)
                    for k, v in doc.metadata.items()
                    if k.startswith("url") and isinstance(v, str)
                ]
                file_name = doc.metadata.get("file_name", "") or ""
                if url_items:
                    # A non-numeric key suffix becomes the display label
                    # (e.g. url_orcid -> "orcid: <url>"); bare "url" and
                    # numbered keys (url_1, url_2) stay plain.
                    display_values = []
                    for k, v in url_items:
                        label = k[len("url") :].lstrip("_")
                        if label and not label.isdigit():
                            display_values.append(f"{label}: {v}")
                        else:
                            display_values.append(v)
                    key: Hashable = tuple(sorted(v for _, v in url_items))
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

        return {
            "reference_material": reference_material,
            "retrieval_attempts": retrieval_attempts,
        }
