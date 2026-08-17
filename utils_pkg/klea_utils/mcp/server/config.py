#!/usr/bin/env python3
"""
Config schema for the bundled tools server.

File: klea_utils/mcp/server/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic import BaseModel, Field


class BundledToolsConfig(BaseModel):
    """Configuration for the shared bundled tools server.

    Read by the app orchestrators (``BaseLangGraph``) to decide whether the
    bundled tools server is wired into the graph and, when it is, which of
    its tools are exposed.  Tools are filtered by *tag* (see
    ``klea_utils.mcp.server.bundled_tools`` for the bundled tag vocabulary);
    the tags are applied to the stdio config entry so fastmcp's
    ``TransformingStdioMCPServer`` enforces them.

    Pydantic-only on purpose: config loads must not pull in fastmcp.
    """

    enabled: bool = True
    #: Only tools carrying at least one of these tags are exposed.
    include_tags: set[str] = Field(default_factory=set)
    #: Tools carrying any of these tags are hidden.
    exclude_tags: set[str] = Field(default_factory=set)
