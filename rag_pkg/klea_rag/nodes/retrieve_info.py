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
from klea_utils.stores.retrieval import VSRetriever

from klea_rag.schemas import RAGState


class RetrieveInfoNode(AbstractLangGraphNode[RAGState, dict[str, Any]]):
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
    async def execute(self, state: RAGState) -> dict[str, Any]:
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

            seen: dict[Hashable, tuple[float, list[str]]] = {}
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
                if key not in seen or score > seen[key][0]:
                    seen[key] = (score, display_values)
            ref_lines = [
                f"1.  [{score:.2f}]\n" + "\n".join(f"    - {v}" for v in values)
                for score, values in sorted(
                    seen.values(), key=lambda x: x[0], reverse=True
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
