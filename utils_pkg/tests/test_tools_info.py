#!/usr/bin/env python3
"""
Tests for the compact MCP tool description formatter.

File: tests/test_tools_info.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import unittest

from klea_utils.tools import (
    _format_tool_parameters,
    build_tool_description,
    clean_tool_meta,
)
from mcp.types import Tool


def _make_tool(
    name="test_tool",
    title="Test tool",
    description="Does useful things.",
    input_schema=None,
):
    input_schema = input_schema or {}
    return Tool(
        name=name,
        title=title,
        description=description,
        inputSchema=input_schema,
        outputSchema=None,
        annotations=None,
        execution=None,
        icons=[],
    )


class TestFormatToolParameters(unittest.TestCase):
    """Tests for _format_tool_parameters."""

    def test_none_and_empty_schema(self):
        self.assertEqual(_format_tool_parameters(None), "")
        self.assertEqual(_format_tool_parameters({}), "")
        self.assertEqual(_format_tool_parameters({"properties": {}}), "")

    def test_required_flag_and_type(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "A path."}},
            "required": ["path"],
        }
        result = _format_tool_parameters(schema)
        self.assertIn("- path (string, required): A path.", result)
        self.assertNotIn("default", result)

    def test_optional_param_without_required_flag(self):
        schema = {
            "type": "object",
            "properties": {"recursive": {"type": "boolean"}},
        }
        result = _format_tool_parameters(schema)
        self.assertIn("- recursive (boolean):", result)

    def test_anyof_collapses_null(self):
        schema = {
            "type": "object",
            "properties": {
                "max_depth": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "description": "Depth.",
                }
            },
        }
        result = _format_tool_parameters(schema)
        self.assertIn("- max_depth (integer): Depth.", result)
        self.assertNotIn("anyOf", result)

    def test_validators_and_defaults_dropped(self):
        schema = {
            "type": "object",
            "properties": {
                "k": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "A number.",
                }
            },
        }
        result = _format_tool_parameters(schema)
        self.assertIn("- k (integer): A number.", result)
        self.assertNotIn("default", result)
        self.assertNotIn("minimum", result)

    def test_description_whitespace_normalised(self):
        schema = {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "\n  a\n    b  c\n"}
            },
        }
        result = _format_tool_parameters(schema)
        self.assertIn("- pattern (string): a b c", result)


class TestBuildToolDescription(unittest.TestCase):
    """Tests for build_tool_description."""

    def test_heading_description_and_parameters(self):
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        }
        desc = build_tool_description(_make_tool(input_schema=schema))
        self.assertIn("## test_tool", desc)
        self.assertIn("Does useful things.", desc)
        self.assertIn("Parameters:", desc)
        self.assertIn("- path (string):", desc)

    def test_no_parameters_section_when_schema_missing(self):
        desc = build_tool_description(_make_tool(input_schema=None))
        self.assertIn("## test_tool", desc)
        self.assertIn("Does useful things.", desc)
        self.assertNotIn("Parameters:", desc)

    def test_no_description(self):
        desc = build_tool_description(_make_tool(description=""))
        self.assertEqual(desc, "## test_tool")


class TestCleanToolMeta(unittest.TestCase):
    """Tests for clean_tool_meta."""

    def test_strips_fastmcp_tags(self):
        self.assertEqual(clean_tool_meta({"fastmcp": {"tags": ["testing"]}}), None)
        self.assertEqual(
            clean_tool_meta({"fastmcp": {"tags": ["testing"]}, "other": 1}),
            {"other": 1},
        )

    def test_preserves_other_metadata(self):
        self.assertEqual(
            clean_tool_meta({"fastmcp": {"tags": ["testing"], "x": 2}}),
            {"fastmcp": {"x": 2}},
        )

    def test_none_and_empty(self):
        self.assertIsNone(clean_tool_meta(None))
        self.assertIsNone(clean_tool_meta({}))

    def test_input_not_mutated(self):
        meta = {"fastmcp": {"tags": ["testing"]}}
        clean_tool_meta(meta)
        self.assertEqual(meta, {"fastmcp": {"tags": ["testing"]}})


if __name__ == "__main__":
    unittest.main()
