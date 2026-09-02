#!/usr/bin/env python3
"""
Helper to map Klea dict results with an ``error`` field to MCP ``isError``.

File: klea_utils/mcp/tool_result.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
from typing import Any

from fastmcp.tools import ToolResult
from mcp.types import TextContent

logger = logging.getLogger(__name__)


def to_result(result: dict[str, Any]) -> ToolResult:
    """Return a ``ToolResult`` with ``is_error`` derived from the ``error`` field.

    Klea tool implementations are framework-agnostic and signal failure via a
    non-empty ``error`` (or ``Error`` for legacy repository tools) string in
    their returned dict.  The MCP spec requires such failures to be reported
    as ``isError: true`` in the ``CallToolResult`` (see
    ``mcp/types.py:CallToolResult`` and the spec section
    ``server/tools#Error Handling``); FastMCP only sets this when the tool
    explicitly returns ``ToolResult(is_error=True)`` or raises ``ToolError``
    (see ``fastmcp/tools/base.py:convert_result`` and
    ``mcp/server/lowlevel/server.py:_make_error_result``).

    This helper bridges the two: it preserves the dict as ``structured_content``
    (so LLM remediation keeps full context) while marking ``is_error`` when
    the ``error``/``Error`` field is truthy.  It intentionally preserves the
    structured payload rather than raising ``ToolError``, which would strip it.

    :param result: Tool return dict containing at least an ``error`` key;
        legacy tools may use ``Error`` (capitalised).
    :returns: ``ToolResult`` with ``is_error`` reflecting the error field.
    """
    error = result.get("error")
    if error is None:
        error = result.get("Error", "")
    # Treat missing, empty, or whitespace-only as success.
    is_error = (
        bool(isinstance(error, str) and error.strip())
        if isinstance(error, str)
        else bool(error)
    )
    # Non-string truthy errors (unlikely) still count as error.
    if not isinstance(error, str) and error:
        is_error = True
    logger.debug(f"{is_error = }\n{error = }")
    # FastMCP serialises dicts via pydantic_core; ensure JSON text is safe.
    try:
        text = json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        text = json.dumps(result, ensure_ascii=False, default=str)
    return ToolResult(
        content=[TextContent(type="text", text=text)],
        structured_content=result,
        is_error=is_error,
    )
