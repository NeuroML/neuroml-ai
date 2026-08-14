#!/usr/bin/env python3
"""
Custom error classes for the MCP tooling.

File: klea_utils/mcp/errors.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


class PermissionDeniedError(PermissionError):
    """Raised when a tool is denied access to a path."""


class DocumentConversionError(Exception):
    """Raised when a document file cannot be converted to text.

    Carries a user-facing message that tools report through their ``error``
    result field.
    """
