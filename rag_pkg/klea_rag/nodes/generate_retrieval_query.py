#!/usr/bin/env python3
"""
Generate retrieval query node

File: rag_pkg/klea_rag/nodes/generate_retrieval_query.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent
from typing import Any, override

from klea_utils.llm import (
    content_to_str,
    extract_llm_output_content,
    prompt_value_to_messages,
)
from klea_utils.nodes.abstract import NodeStreamData
from klea_utils.nodes.base import BaseLLMNode
from langchain_core.messages import AIMessage
from langchain_core.runnables.utils import Output

from klea_rag.schemas import RAGState


class GenerateRetrievalQuery(BaseLLMNode[RAGState]):
    """Node that generates a concise retrieval query from the user's question."""

    model_type = "chat"

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
        temperature: float = 0.3,
    ):
        """Initialise the node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            temperature=temperature,
            output_schema=None,
            memory=True,
        )

    @override
    def _get_system_prompt(self, state: RAGState) -> str:
        """Load system prompt, optionally appending evaluator feedback."""
        system_prompt = super()._get_system_prompt(state)

        if state.retrieval_attempts > 0:
            self.logger.info("Regenerating retrieval query, updating system prompt")
            sentence, newline, rest = system_prompt.partition("\n")
            new_sentence = dedent(
                """
                Generate a new concise retrieval query from the user's question. Think about the user's intent step by step.
                Take the evaluator's feedback into account.

                Previous query: {previous}

                Evaluator feedback:

                {feedback}

                """
            )
            system_prompt = new_sentence + rest
            self.logger.debug(f"New {system_prompt =}")

        return system_prompt

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with user query."""
        return {
            "query": state.query,
            "feedback": state.text_response_eval.summary,
            "previous": state.retrieval_query,
        }

    @override
    def _update_state(self, result: Output, state: RAGState) -> dict[str, Any]:
        """Update state with the generated retrieval query."""
        content = content_to_str(result.content)
        thought, answer = (
            content.split("</think>", 1) if "</think>" in content else ("", content)
        )
        answer = answer.strip()

        messages = state.messages
        output = AIMessage(content=answer)
        messages.append(output)

        return {
            "messages": messages,
            "retrieval_query": answer,
            "retrieval_attempts": state.retrieval_attempts + 1,
        }

    @override
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails."""
        return ""

    @override
    def _get_info(self) -> NodeStreamData:
        """Return retrieval query and attempt number."""
        assert self._last_state_updates is not None
        query = self._last_state_updates.get("retrieval_query", "")
        attempt = self._last_state_updates.get("retrieval_attempts", 1)
        action = "Regenerated" if attempt > 1 else "Generated"
        return NodeStreamData(
            heading="Retrieval Query Generation",
            summary=f"{action} retrieval query (attempt {attempt})",
            details={
                "retrieval_query": query,
                "retrieval_attempts": attempt,
            },
        )

    @override
    def _get_debug(self) -> NodeStreamData:
        """Return info + input prompt, raw output, processed output, and evaluator feedback."""
        assert self._last_state is not None
        assert self._last_prompt is not None
        assert self._last_output is not None
        assert self._last_result is not None
        info = self._get_info()
        details = info.details.copy()
        details.update(
            {
                "input_prompt": prompt_value_to_messages(self._last_prompt),
                "unprocessed_output": extract_llm_output_content(self._last_output),
                "processed_output": str(self._last_result),
            }
        )
        # Add evaluator feedback if this is a retry
        state: RAGState = self._last_state  # type: ignore[assignment]
        if state.retrieval_attempts > 0 and state.text_response_eval:
            details["evaluator_feedback"] = state.text_response_eval.summary
            details["previous_query"] = state.retrieval_query
        return NodeStreamData(
            heading=info.heading, summary=info.summary, details=details
        )
