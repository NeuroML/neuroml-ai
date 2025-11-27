#!/usr/bin/env python3
"""
Schemas used in the RAG

File: neuroml_ai/schemas.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing_extensions import List, Literal, Tuple, Dict


class QueryTypeSchema(BaseModel):
    """Schema for query type."""

    query_type: Literal[
        "undefined", "general_question", "neuroml_question", "neuroml_code_generation"
    ] = Field(
        default="undefined",
    )


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
    ] = Field(default="undefined")
    summary: str = ""


class AgentState(BaseModel):
    """The state of the graph"""

    query: str = ""
    query_type: QueryTypeSchema = QueryTypeSchema()
    text_response_eval: EvaluateAnswerSchema = EvaluateAnswerSchema()
    # TODO: code_response_eval: EvaluateAnswerSchema
    messages: List[AnyMessage] = Field(default_factory=list)
    # summarised version of context so far
    context_summary: str = ""
    # index till which summarised
    summarised_till: int = 0
    message_for_user: str = ""
    reference_material: Dict[str, List[Tuple]] = Field(default_factory=dict)


class ToolCallSchema(BaseModel):
    """Schema for tool call response."""

    action: Literal["tool_call", "text_reply", "undefined"] = Field(
        default="undefined",
    )
    tool: str = ""
    args: Dict[str, str] = Field(default_factory=dict)
    reason: str = ""
