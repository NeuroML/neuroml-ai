#!/usr/bin/env python3
"""
Schemas shared by LangGraph orchestrators.

File: klea_utils/graph/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic import BaseModel


class TokenUsage(BaseModel):
    """Token usage accumulated across the nodes in a graph run."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
