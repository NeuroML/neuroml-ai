#!/usr/bin/env python3
"""
Schemas shared by MCP servers and clients.

File: klea_utils/mcp/schemas.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any

from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Metadata used to describe an MCP tool to clients and models."""

    # Detailed tool documentation for the LLM; falls back to the function docstring.
    description: str | None = None
    # Short human-facing label for UI and MCP clients.
    title: str | None = None
    # Categories used to group and filter tools.
    tags: set[str] | None = None
    # Argument names that are filesystem paths and must pass
    # check_path_access before the tool is invoked.  Read client-side by the
    # tool caller node to gate tool calls before they reach the MCP server.
    checkpaths: list[str] | None = None
    # Additional application-specific metadata.
    meta: dict[str, Any] | None = None
