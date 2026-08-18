#!/usr/bin/env python3
"""
Generate retrieval query node

File: rag_pkg/klea_rag/nodes/generate_retrieval_query.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent
from typing import Any, cast, override

from klea_utils.llm import (
    extract_llm_output_content,
    prompt_value_to_messages,
)
from klea_utils.nodes.abstract import NodeStreamData
from klea_utils.nodes.base import BaseLLMNode
from langchain_core.messages import AIMessage

from klea_rag.schemas import RAGState, RetrievalQueryOutput


class GenerateRetrievalQuery(BaseLLMNode[RetrievalQueryOutput]):
    """Node that generates a concise retrieval query from the user's question.

    Uses structured output (:class:`RetrievalQueryOutput`) so the search
    query and any retrieval constraints (year range, journal, authors,
    keywords) are produced together from the user's question.
    """

    model_type = "chat"
    model_defaults = {"temperature": 0.3, "max_output_tokens": 2048}

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
    ):
        """Initialise the node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        """
        super().__init__(
            logger=logger,
            label=label,
            llm_models=llm_models,
            output_schema=RetrievalQueryOutput,
            memory=True,
        )

    @override
    def _get_system_prompt(self, state: RAGState) -> str | list[Any]:
        """Load system prompt, optionally appending evaluator feedback."""
        system_prompt = super()._get_system_prompt(state)

        if state.retrieval_attempts > 0:
            self.logger.info("Regenerating retrieval query, updating system prompt")
            new_sentence = dedent(
                """
                Generate a new concise retrieval query from the user's question. Think about the user's intent step by step.
                Take the evaluator's feedback into account.

                Previous query: {previous}

                Evaluator feedback:

                {feedback}

                """
            )
            if isinstance(system_prompt, list):
                # Memory enabled: super() returns a ("system", text) list plus
                # recent history; replace just the text part.
                text = system_prompt[0][1]
                sentence, newline, rest = text.partition("\n")
                system_prompt = list(system_prompt)
                system_prompt[0] = ("system", new_sentence + rest)
            else:
                sentence, newline, rest = system_prompt.partition("\n")
                system_prompt = new_sentence + rest
            self.logger.debug(f"New {system_prompt =}")

        return system_prompt

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with user query."""
        return {
            "query": state.query,
            "feedback": state.text_response_eval.summary,
            "previous": state.retrieval_query.search_query,
        }

    @override
    def _update_state(
        self, result: RetrievalQueryOutput, state: RAGState
    ) -> dict[str, Any]:
        """Update state with the generated search query and filters.

        Always writes a fresh :class:`RetrievalQueryOutput` instance from
        the current LLM output, so nothing from a prior turn is carried
        over.
        """
        messages = state.messages
        output = AIMessage(content=result.search_query)
        messages.append(output)

        return {
            "messages": messages,
            "retrieval_query": result,
        }

    @override
    def _get_default_error_result(self) -> RetrievalQueryOutput:
        """Return default result when processing fails."""
        self.logger.error("Processing failed")
        return RetrievalQueryOutput()

    @override
    def _get_info(self) -> NodeStreamData:
        """Return search query, filters, and attempt number."""
        assert self._last_state_updates is not None
        assert self._last_state is not None
        rq = self._last_state_updates.get("retrieval_query") or RetrievalQueryOutput()
        search_query = rq.search_query
        # Display-only: this node does not bump the counter (the retrieval
        # node does). Here it holds the number of prior retrieval passes, so
        # the current query generation is labelled as the next attempt.
        state = cast(RAGState, self._last_state)
        attempt = state.retrieval_attempts + 1
        action = "Regenerated" if state.retrieval_attempts > 0 else "Generated"
        return NodeStreamData(
            heading="Retrieval Query Generation",
            summary=f"{action} retrieval query (attempt {attempt})",
            details={
                "search_query": search_query,
                "metadata_filter": rq.to_metadata_filter(),
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
            details["previous_query"] = state.retrieval_query.search_query
        return NodeStreamData(
            heading=info.heading, summary=info.summary, details=details
        )
