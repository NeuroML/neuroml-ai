#!/usr/bin/env python3
"""
Tests for the structured-output schema prompt block.

File: tests/test_node_base_schema.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
from typing import Literal

from klea_utils.nodes.base import BaseLLMNode, _is_empty_result, _schema_to_example
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AnswerSchema(BaseModel):
    """A minimal structured-output schema for testing."""

    answer: str = ""
    references: list[str] = Field(default_factory=list)


class EvalSchema(BaseModel):
    """Schema exercising enum and numeric types."""

    confidence: float = 0.0
    next_step: Literal["continue", "rewrite_answer"] = "continue"
    summary: str = ""


class NestedSchema(BaseModel):
    """Schema exercising nested object/array types."""

    tool_calls: list[dict] = Field(default_factory=list)


class MemoryState(BaseModel):
    """Minimal state exposing the fields the memory hook reads."""

    messages: list = Field(default_factory=list)
    context_summary: str = ""


class DummyNode(BaseLLMNode):
    """Concrete BaseLLMNode for testing the prompt block."""

    model_type = "chat"

    def _get_prompt_variables(self, state: BaseModel) -> dict:
        return {}

    def _update_state(self, result, state: BaseModel) -> dict:
        return {}

    def _get_default_error_result(self):
        return AIMessage(content="")


def _node(schema) -> DummyNode:
    return DummyNode(
        logger=logging.getLogger("test_node_base_schema"),
        label="Dummy",
        llm_models={"chat": None},
        output_schema=schema,
    )


def _render(block: str) -> str:
    """Simulate ChatPromptTemplate un-escaping the braces in the block."""
    tpl = ChatPromptTemplate([("system", block), ("human", "{query}")])
    return tpl.invoke({"query": "q"}).to_string()


def test_schema_to_example_string():
    assert _schema_to_example({"type": "string"}) == "text"


def test_schema_to_example_number_and_boolean():
    assert _schema_to_example({"type": "integer"}) == 0
    assert _schema_to_example({"type": "number"}) == 0
    assert _schema_to_example({"type": "boolean"}) is True


def test_schema_to_example_enum_uses_first_value():
    schema = {"type": "string", "enum": ["continue", "rewrite_answer"]}
    assert _schema_to_example(schema) == "continue"


def test_schema_to_example_array_and_object():
    assert _schema_to_example({"type": "array", "items": {"type": "string"}}) == [
        "text"
    ]
    nested = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    assert _schema_to_example(nested) == {"name": "text", "count": 0}


def test_schema_to_example_unknown_type_is_none():
    assert _schema_to_example({"type": "unknown"}) is None


def test_prompt_block_strips_title_and_description():
    """The prompt block drops top-level title/description metadata."""
    rendered = _render(_node(AnswerSchema)._format_output_schema_prompt())
    assert '"title"' not in rendered
    assert '"description"' not in rendered


def test_prompt_block_contains_directive_and_example():
    """The prompt block tells the model not to echo the schema and shows an example."""
    rendered = _render(_node(AnswerSchema)._format_output_schema_prompt())
    assert "Do not output the schema definition itself" in rendered
    assert '"answer": "text"' in rendered
    assert '"references": ["text"]' in rendered


def test_prompt_block_example_matches_schema():
    """The rendered example parses as JSON and uses the schema's keys."""
    rendered = _render(_node(NestedSchema)._format_output_schema_prompt())
    example_line = next(
        line.strip()
        for line in rendered.splitlines()
        if line.strip().startswith("{") and "[{}]" in line
    )
    example = json.loads(example_line)
    assert example == {"tool_calls": [{}]}


def test_system_prompt_puts_schema_after_memory(tmp_path):
    """The output-schema block is appended after memory so it is the last
    system instruction before the human query."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "DummyNode_system.md").write_text("Base system prompt.")
    node = _node(AnswerSchema)
    node.prompt_registry_location = prompts
    node.memory = True
    state = MemoryState(context_summary="remember-the-context")

    system = node._get_system_prompt(state)

    # With memory enabled the system prompt is a list of ``("system", text)``
    # plus any recent history messages; the text lives in the first element.
    assert isinstance(system, list)
    system_text = system[0][1]
    assert "remember-the-context" in system_text
    assert system_text.index("## Output schema (strict)") > system_text.index(
        "remember-the-context"
    )


def test_is_empty_result_structured_default():
    """An all-default structured instance is flagged as empty."""
    assert _is_empty_result(AnswerSchema(), AnswerSchema) is True


def test_is_empty_result_structured_populated():
    """A populated structured instance is not empty."""
    assert _is_empty_result(AnswerSchema(answer="some answer"), AnswerSchema) is False


def test_is_empty_result_non_structured_blank():
    """A blank AIMessage (no output schema) is flagged as empty."""
    assert _is_empty_result(AIMessage(content="")) is True
    assert _is_empty_result(AIMessage(content="   ")) is True


def test_is_empty_result_non_structured_populated():
    """A non-blank AIMessage is not empty."""
    assert _is_empty_result(AIMessage(content="some text")) is False


def test_process_output_warns_on_empty_result(caplog):
    """An all-default structured parse triggers a warning log."""
    node = _node(AnswerSchema)
    output = {
        "parsed": AnswerSchema(),
        "parsing_error": None,
        "raw": AIMessage(content="{}"),
    }
    with caplog.at_level(logging.WARNING):
        result = node._process_output(output)
    assert result == AnswerSchema()
    assert "Empty LLM output from Dummy" in caplog.text


def test_process_output_no_warning_on_populated_result(caplog):
    """A populated structured parse does not warn."""
    node = _node(AnswerSchema)
    output = {
        "parsed": AnswerSchema(answer="a real answer"),
        "parsing_error": None,
        "raw": AIMessage(content='{"answer": "a real answer"}'),
    }
    with caplog.at_level(logging.WARNING):
        result = node._process_output(output)
    assert result == AnswerSchema(answer="a real answer")
    assert "Empty LLM output" not in caplog.text
