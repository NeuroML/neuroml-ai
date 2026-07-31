#!/usr/bin/env python3
"""
Reducers shared by LangGraph orchestrators.

File: klea_utils/graph/reducers.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.graph.schemas import TokenUsage


def add_token_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Add token usage updates from sequential or concurrent graph nodes."""
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )
