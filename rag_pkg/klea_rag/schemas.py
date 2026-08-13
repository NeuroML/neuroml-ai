#!/usr/bin/env python3
"""
Schemas used in the RAG

File: klea_rag/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Annotated, Any, Literal

from fastmcp.client.client import CallToolResult
from klea_utils.graph.reducers import add_token_usage
from klea_utils.graph.schemas import TokenUsage
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field


class EvaluateAnswerSchema(BaseModel):
    """Evaluation of LLM generated answer. Descriptions given in the main prompt"""

    confidence: float = 0.0
    coverage: float = 0.0
    relevance: float = 0.0
    groundedness: float = 0.0
    coherence: float = 0.0
    conciseness: float = 0.0
    next_step: Literal[
        "continue", "retrieve_more_info", "modify_query", "rewrite_answer", "undefined"
    ] = Field(default="undefined", validate_default=True)
    summary: str = ""


class ToolCallSchema(BaseModel):
    """Schema for tool call response."""

    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


# For Tool Picker
class ToolCallsSchema(BaseModel):
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)


class RetrievalQueryOutput(BaseModel):
    """Structured output of the retrieval-query generator.

    Holds the search query alongside any retrieval constraints derived
    from the user's question (publication year range, journal, authors,
    keywords).  The typed filter fields are the LLM-facing surface; the
    normalized backend-agnostic filter dict is produced by
    :meth:`to_metadata_filter` and consumed by the retrievers.
    """

    search_query: str = ""
    year_from: int | None = None
    year_to: int | None = None
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    def to_metadata_filter(self) -> dict[str, Any] | None:
        """Return the normalized metadata filter, or ``None``.

        Maps the typed filter fields to a backend-agnostic filter dict
        (``{field: {op: value}}``): ``year`` becomes a range clause,
        scalar fields (``journal``) use implicit equality, and list
        fields (``authors``/``keywords``) use ``$contains``
        element-membership, combined with ``$and`` when several are set.
        Year bounds below 1 are treated as unset (models sometimes echo
        the example placeholder ``0``).  Returns ``None`` when no
        constraint is set.
        """
        clauses: dict[str, Any] = {}

        if self.year_from is not None and self.year_from > 0:
            clauses.setdefault("year", {})["$gte"] = self.year_from
        if self.year_to is not None and self.year_to > 0:
            clauses.setdefault("year", {})["$lte"] = self.year_to

        if self.journal:
            clauses["journal"] = self.journal

        and_terms: list[dict[str, Any]] = []
        for field in ("authors", "keywords"):
            for value in getattr(self, field):
                and_terms.append({field: {"$contains": value}})

        if and_terms:
            if len(and_terms) == 1:
                clauses.update(and_terms[0])
            else:
                clauses["$and"] = and_terms

        return clauses if clauses else None


class RAGState(BaseModel):
    """The state of the graph"""

    query: str = ""
    # schema for this is computed at run time for the classifier node
    query_domains: list[str] = Field(default=["undefined"], validate_default=True)
    text_response_eval: EvaluateAnswerSchema = EvaluateAnswerSchema()
    guard_decision: str = "safe"
    messages: list[AnyMessage] = Field(default_factory=list)

    # summarised version of context so far
    context_summary: str = ""

    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""

    # tool calls
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)
    tool_results: list[CallToolResult] = Field(default_factory=list)

    # reference material from retrievals
    reference_material: dict[str, list[tuple]] = Field(default_factory=dict)

    # number of retrieval passes in the evaluator loop (the initial query
    # retrieval, retrieve_more_info k-increases, and modify_query
    # re-retrievals); incremented once per RetrieveInfoNode execution
    retrieval_attempts: int = 0

    # number of answer rewrite attempts in evaluator loop
    rewrite_attempts: int = 0

    # generated retrieval query (and any retrieval filters) for the
    # current round
    retrieval_query: RetrievalQueryOutput = RetrievalQueryOutput()

    # Token usage is reduced so parallel nodes can update it safely.
    usage_metrics: Annotated[TokenUsage, add_token_usage] = Field(
        default_factory=TokenUsage
    )
