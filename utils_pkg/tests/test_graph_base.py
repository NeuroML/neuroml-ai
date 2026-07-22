#!/usr/bin/env python3
"""
Test BaseLangGraph execution methods with a toy graph.

File: tests/test_graph_base.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import List, Type, override

import pytest
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

from klea_utils.graph.base import BaseLangGraph, LLMModel
from klea_utils.llm import setup_llm
from klea_utils.nodes.answer_general import AnswerGeneral
from klea_utils.nodes.fixed_answer import FixedAnswer
from klea_utils.plogging import setup_logger


class ToyState(BaseModel):
    """Minimal state for the toy graph."""

    query: str = ""
    message_for_user: str = ""
    messages: List[AnyMessage] = Field(default_factory=list)
    context_summary: str = ""


class ToyGraph(BaseLangGraph):
    """Minimal graph: AnswerGeneral (LLM) -> FixedAnswer (non-LLM) -> END."""

    env_class: Type[BaseModel] = BaseModel
    config_class: Type[BaseModel] = BaseModel
    env_var: str = "TOY_ENV_FILE"
    env_file_default: str = "toy.env"
    graph_name: str = "ToyGraph"

    def __init__(self):
        super().__init__(logging_level=logging.WARNING, memory=False)
        from platformdirs import PlatformDirs

        self.paths = PlatformDirs(self.graph_name.lower())
        self.logger = setup_logger(self.graph_name, stderr_level=logging.INFO)
        self.logger.propagate = False

    @override
    def _load_env(self) -> None:
        """No-op: skip env file loading."""
        pass

    @override
    def _configure_resources(self) -> None:
        pass

    @override
    def _setup_models(self) -> None:
        self.llm_models = {
            "chat": LLMModel(
                instance=setup_llm("ollama:qwen3:0.6b", logger=self.logger)
            ),
        }

    @override
    async def _create_graph(self) -> None:
        from langgraph.graph import END, START, StateGraph

        workflow = StateGraph(ToyState)  # ty: ignore[invalid-assignment]

        self._answer_node = AnswerGeneral(
            logger=self.logger,
            label="Saying hello",
            llm_models=self.llm_models,
            temperature=0.3,
            memory=False,
        )
        workflow.add_node(self._answer_node.label, self._answer_node.execute)

        self._fixed_node = FixedAnswer(
            logger=self.logger,
            label="Fixed answer",
            state_attr="message_for_user",
            message="This is the fixed answer.",
        )
        workflow.add_node(self._fixed_node.label, self._fixed_node.execute)

        workflow.add_edge(START, self._answer_node.label)
        workflow.add_edge(self._answer_node.label, self._fixed_node.label)
        workflow.add_edge(self._fixed_node.label, END)

        self.graph = workflow.compile()


class TestGraphBase:
    """Test all BaseLangGraph execution methods."""

    def setup_method(self):
        self.logger = logging.getLogger("test_graph_base")

    async def _make_graph(self):
        """Build a ToyGraph, skipping if ollama is unavailable."""
        try:
            graph = ToyGraph()
            await graph.setup()
            self.logger.info("ToyGraph setup complete")
            return graph
        except (ConnectionError, AssertionError) as e:
            self.logger.warning(f"Ollama unavailable, skipping: {e}")
            pytest.skip(str(e))

    @pytest.mark.localonly
    async def test_run_graph_invoke(self):
        """Test basic invoke returns a string."""
        toy_graph = await self._make_graph()
        self.logger.info("Testing run_graph_invoke")
        query = "Say 'hi' for testing."
        self.logger.info(f"{query = }")

        result = await toy_graph.run_graph_invoke(query)
        self.logger.info(f"Got result: {result!r}")

        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "This is the fixed answer."
        self.logger.info("run_graph_invoke test passed")

    @pytest.mark.localonly
    async def test_run_graph_invoke_state(self):
        """Test invoke with full state dict."""
        toy_graph = await self._make_graph()
        self.logger.info("Testing run_graph_invoke_state")

        state = {"query": "Say 'hi' for testing."}
        self.logger.info(f"Initial {state = }")

        result = await toy_graph.run_graph_invoke_state(state)
        self.logger.info(f"Final state keys: {list(result.keys())}")
        self.logger.info(f"Final {result = }")

        assert isinstance(result, dict)
        assert "message_for_user" in result
        assert result["message_for_user"] == "This is the fixed answer."
        self.logger.info("run_graph_invoke_state test passed")

    @pytest.mark.localonly
    async def test_run_graph_stream(self):
        """Test astream yields message_for_user strings."""
        toy_graph = await self._make_graph()
        self.logger.info("Testing run_graph_stream")

        messages = []
        async for msg in toy_graph.run_graph_stream("Say 'hi' for testing."):
            self.logger.info(f"Streamed message: {msg!r}")
            messages.append(msg)

        self.logger.info(f"Got {len(messages)} streamed messages")

        assert len(messages) >= 1
        assert "This is the fixed answer." in messages
        self.logger.info("run_graph_stream test passed")

    @pytest.mark.localonly
    async def test_run_graph_astream_events(self):
        """Test astream_events v3 yields progress, token, and complete events.

        Event flow for the toy graph (AnswerGeneral -> FixedAnswer):

          1. AnswerGeneral (LLM node) calls ``write_custom_stream()`` at
             the top of ``execute()``, emitting a progress event via the
             ``custom`` channel.

          2. Token chunks from AnswerGeneral are yielded as ``type="token"``
             events (from the ``messages`` channel).

          3. FixedAnswer (non-LLM node) also calls ``write_custom_stream()``,
             emitting another progress event via the ``custom`` channel.

          4. After the graph completes, the final state is read from the
             ``values`` channel and yielded as ``type="complete"``.
             ``values`` channel and yielded as ``type="complete"``.
        """
        toy_graph = await self._make_graph()
        self.logger.info("Testing run_graph_astream_events")

        events = []
        async for event in toy_graph.run_graph_astream_events("Say 'hi' for testing."):
            self.logger.info(f"Event: {event}")
            events.append(event)

        self.logger.info(f"Got {len(events)} total events")

        progress_events = [e for e in events if e.get("type") == "progress"]
        token_events = [e for e in events if e.get("type") == "token"]
        complete_events = [e for e in events if e.get("type") == "complete"]

        self.logger.info(
            f"Event breakdown: "
            f"{len(progress_events)} progress, "
            f"{len(token_events)} token, "
            f"{len(complete_events)} complete"
        )

        self.logger.info(f"{progress_events = }")
        self.logger.info(f"{token_events = }")
        self.logger.info(f"{complete_events = }")

        assert len(events) >= 2

        progress_nodes = [e["node"] for e in progress_events]
        self.logger.info(f"Progress nodes: {progress_nodes}")

        # Both nodes appear via the custom channel (write_custom_stream).
        # "Saying hello" is emitted by AbstractLLMNode.execute() in its
        # template method; "Fixed answer" is emitted in its own execute().
        # Every node with a label now emits via the custom channel, so the
        # messages channel is only used for token content.
        assert "Saying hello" in progress_nodes
        assert "Fixed answer" in progress_nodes

        assert len(complete_events) == 1
        self.logger.info(f"Complete event: {complete_events[0]}")
        assert complete_events[0]["message_for_user"] == "This is the fixed answer."

        for te in token_events:
            assert "content" in te
            assert "node" in te
            assert isinstance(te["content"], str)

        self.logger.info("run_graph_astream_events test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
