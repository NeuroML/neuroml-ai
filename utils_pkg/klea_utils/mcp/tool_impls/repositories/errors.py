#!/usr/bin/env python3
"""
Error classes for the repository source implementations.

File: klea_utils/mcp/tool_impls/repositories/errors.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


class RepositorySourceError(Exception):
    """Raised when a repository source cannot be queried.

    Carries a user-facing message that the public functions report through
    their ``error`` result field.
    """
