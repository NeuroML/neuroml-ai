#!/usr/bin/env python3
"""
Schemas used in the RAG

File: neuroml_ai/schemas.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import List, Literal


class QueryTypeSchema(BaseModel):
    """Docstring for QueryTypeSchema."""

    query_type: Literal["undefined", "general question", "neuroml question",
                        "neuroml code generation"] = Field(
        default="undefined",
        description="'question' if user is asking for information, 'code_generation', if the user is asking for code, 'unknown' otherwise",
    )


class EvaluateAnswerSchema(BaseModel):
    """Evaluation of LLM generated answer"""

    relevance: float = 0.0
    groundedness: float = 0.0
    completeness: float = 0.0
    coherence: float = 0.0
    conciseness: float = 0.0
    confidence: float = 0.0
    # description given in the system prompt
    next_step: Literal[
        "continue", "retrieve_more_info", "modify_query", "ask_user", "undefined"
    ] = Field(default="undefined")
    summary: str = ""


class AgentState(BaseModel):
    """The state of the graph"""

    query: str = ""
    query_type: QueryTypeSchema = QueryTypeSchema()
    text_response_eval: EvaluateAnswerSchema = EvaluateAnswerSchema()
    # TODO: code_response_eval: EvaluateAnswerSchema
    messages: List[BaseMessage] = Field(default_factory=list)
    # summarised version of context so far
    context_summary: str = ""
    user_message: str = ""
