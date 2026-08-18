#!/usr/bin/env python3
"""
Tests for adaptive LLM invoke retries (BaseLLMNode._invoke_with_retries).

File: tests/test_llm_invoke_retries.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from unittest import mock

import pytest
from klea_utils.llm import LLMModel
from klea_utils.models_catalog import ModelLimits
from klea_utils.nodes.base import (
    MAX_CONTEXT_OVERFLOW_RETRIES,
    MAX_OUTPUT_TOKENS_CEILING,
    MAX_TRUNCATION_RETRIES,
    BaseLLMNode,
)
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import StringPromptValue
from pydantic import BaseModel

logger = logging.getLogger("test")


class _OutputSchema(BaseModel):
    answer: str


class _MinimalLLMNode(BaseLLMNode[BaseModel]):
    """Concrete LLM node implementing the remaining abstract methods."""

    model_type = "chat"

    def _get_prompt_variables(self, state):
        return {}

    def _update_state(self, result, state):
        return {}

    def _get_default_error_result(self):
        return ""


def make_node(inst, output_schema=None):
    """Build a minimal node backed by a mocked model instance."""
    return _MinimalLLMNode(
        logger=logger,
        label="test",
        llm_models={"chat": LLMModel(instance=inst, model_name="openai:gpt-4o")},
        output_schema=output_schema,
    )


def make_config(max_tokens=4096, model="gpt-4o", provider="openai"):
    """Build a per-invoke config in the shape _build_invoke_config returns."""
    return {
        "configurable": {
            "model": model,
            "model_provider": provider,
            "max_tokens": max_tokens,
        }
    }


class TestBuildInvokeConfigNoModel:
    """Tests for the role-aware 'no model configured' guard."""

    def test_empty_model_raises_clear_error(self):
        """An empty resolved model raises an actionable error."""
        node = _MinimalLLMNode(
            logger=logger,
            label="test",
            llm_models={
                "chat": LLMModel(instance=mock.Mock(), model_name="", required=True)
            },
            output_schema=None,
        )

        with pytest.raises(RuntimeError, match="No model configured for role 'chat'"):
            node._build_invoke_config()

    def test_set_model_builds_config(self):
        """A resolved model proceeds past the guard."""
        node = _MinimalLLMNode(
            logger=logger,
            label="test",
            llm_models={
                "chat": LLMModel(
                    instance=mock.Mock(), model_name="openai:gpt-4o", required=True
                )
            },
            output_schema=None,
        )
        node._last_prompt = StringPromptValue(text="hi")

        config = node._build_invoke_config()
        assert config["configurable"]["model"] == "gpt-4o"


class TestInvokeWithRetries:
    """Tests for _invoke_with_retries via the plain (non-structured) path."""

    def setup_method(self):
        # No catalog data by default so tests are hermetic and offline.
        self._catalog_patcher = mock.patch(
            "klea_utils.llm.get_model_limits", return_value=None
        )
        self._catalog_patcher.start()

    def teardown_method(self):
        self._catalog_patcher.stop()

    async def _invoke(self, inst, config=None):
        node = make_node(inst)
        return await node._invoke_llm(
            inst, StringPromptValue(text="hi"), config or make_config()
        )

    async def test_context_overflow_retry_shrinks_window(self):
        """An overflow error retries once with a halved output window."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=[
                RuntimeError("Error code: 400 - context_length_exceeded"),
                AIMessage(content="ok", response_metadata={"finish_reason": "stop"}),
            ]
        )
        config = make_config(max_tokens=4096)
        out = await self._invoke(inst, config)

        assert inst.ainvoke.await_count == 2
        assert out.content == "ok"
        assert config["configurable"]["max_tokens"] == 2048

    async def test_context_overflow_exhausts_retries(self):
        """Persistent overflow raises after all shrink retries are used."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=RuntimeError("Error code: 400 - context_length_exceeded")
        )
        try:
            await self._invoke(inst)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

        assert inst.ainvoke.await_count == 1 + MAX_CONTEXT_OVERFLOW_RETRIES

    async def test_rate_limited_no_retry(self):
        """Rate-limit errors surface immediately, as designed."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=RuntimeError("Error code: 429 - Rate limit reached")
        )
        try:
            await self._invoke(inst)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

        assert inst.ainvoke.await_count == 1

    async def test_truncation_retry_grows_window(self):
        """A truncated output retries once with a doubled output window."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=[
                AIMessage(
                    content="partial", response_metadata={"finish_reason": "length"}
                ),
                AIMessage(content="full", response_metadata={"finish_reason": "stop"}),
            ]
        )
        config = make_config(max_tokens=4096)
        out = await self._invoke(inst, config)

        assert inst.ainvoke.await_count == 2
        assert out.content == "full"
        assert config["configurable"]["max_tokens"] == 8192

    async def test_truncation_exception_retry_grows_window(self):
        """A raised truncation error (OpenAI streaming path) retries with a doubled window."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=[
                RuntimeError(
                    "Could not parse response content as the length limit was reached"
                ),
                AIMessage(content="full", response_metadata={"finish_reason": "stop"}),
            ]
        )
        config = make_config(max_tokens=4096)
        out = await self._invoke(inst, config)

        assert inst.ainvoke.await_count == 2
        assert out.content == "full"
        assert config["configurable"]["max_tokens"] == 8192

    async def test_truncation_exception_exhausts_retries(self):
        """Persistent raised truncation doubles, then jumps to the ceiling, then re-raises."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=RuntimeError(
                "Could not parse response content as the length limit was reached"
            )
        )
        config = make_config(max_tokens=1024)
        try:
            await self._invoke(inst, config)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

        # 1 original + 2 doubling rungs + 1 ceiling jump.
        assert inst.ainvoke.await_count == 1 + MAX_TRUNCATION_RETRIES + 1
        assert config["configurable"]["max_tokens"] == MAX_OUTPUT_TOKENS_CEILING

    async def test_truncation_exception_jumps_to_ceiling_after_doubling(self):
        """After the doubling rungs, truncation jumps to the large ceiling."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            side_effect=[
                RuntimeError(
                    "Could not parse response content as the length limit was reached"
                ),
                RuntimeError(
                    "Could not parse response content as the length limit was reached"
                ),
                RuntimeError(
                    "Could not parse response content as the length limit was reached"
                ),
                AIMessage(content="full", response_metadata={"finish_reason": "stop"}),
            ]
        )
        config = make_config(max_tokens=1024)
        out = await self._invoke(inst, config)

        # 1024 -> 2048 (doubling 1) -> 4096 (doubling 2) -> 32768 (jump) -> success.
        assert inst.ainvoke.await_count == 4
        assert out.content == "full"
        assert config["configurable"]["max_tokens"] == MAX_OUTPUT_TOKENS_CEILING

    async def test_truncation_at_ceiling_no_more_retries(self):
        """Once the ceiling retry also truncates, no further retries occur."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            return_value=AIMessage(
                content="cut", response_metadata={"finish_reason": "length"}
            )
        )
        config = make_config(max_tokens=1024)
        out = await self._invoke(inst, config)

        # 1 original + 2 doublings + 1 ceiling jump = 4 attempts.
        assert inst.ainvoke.await_count == 1 + MAX_TRUNCATION_RETRIES + 1
        assert out.content == "cut"
        assert config["configurable"]["max_tokens"] == MAX_OUTPUT_TOKENS_CEILING

    async def test_truncation_jump_uses_endpoint_context(self):
        """A large endpoint max_model_len lets the jump exceed the fixed ceiling."""
        with mock.patch(
            "klea_utils.nodes.base.get_endpoint_model_limits",
            return_value=ModelLimits(context=262144, output=None, input=None),
        ):
            inst = mock.Mock()
            inst.ainvoke = mock.AsyncMock(
                side_effect=[
                    RuntimeError(
                        "Could not parse response content as the length limit was reached"
                    ),
                    RuntimeError(
                        "Could not parse response content as the length limit was reached"
                    ),
                    RuntimeError(
                        "Could not parse response content as the length limit was reached"
                    ),
                    AIMessage(
                        content="full", response_metadata={"finish_reason": "stop"}
                    ),
                ]
            )
            # base_url present so the endpoint lookup applies; small prompt so
            # the endpoint headroom (~262k) is far above the 32768 fallback.
            config = make_config(max_tokens=1024)
            config["configurable"]["base_url"] = (
                "https://inf01.arc-llm.condenser.arc.ucl.ac.uk/v1/"
            )
            config["configurable"]["model_provider"] = "openai"
            node = make_node(inst)
            node._last_prompt = StringPromptValue(text="hi")
            out = await node._invoke_llm(inst, StringPromptValue(text="hi"), config)

        # 1024 -> 2048 -> 4096 (doublings) -> jump to endpoint headroom, not 32768.
        assert inst.ainvoke.await_count == 4
        assert out.content == "full"
        assert config["configurable"]["max_tokens"] > MAX_OUTPUT_TOKENS_CEILING

    async def test_truncation_jump_ignores_models_dev_output_cap(self):
        """The jump trusts the endpoint and is not shrunk by a lower models.dev cap.

        The retry path passes use_endpoint=True for shrink/double, but the
        jump bypasses the resolver entirely (single endpoint lookup), so a
        models.dev output cap lower than the endpoint context must not
        under-cut the raise.
        """
        with mock.patch(
            "klea_utils.nodes.base.get_endpoint_model_limits",
            return_value=ModelLimits(context=262144, output=None, input=None),
        ) as endpoint_lookup:
            inst = mock.Mock()
            inst.ainvoke = mock.AsyncMock(
                side_effect=[
                    RuntimeError(
                        "Could not parse response content as the length limit was reached"
                    ),
                    RuntimeError(
                        "Could not parse response content as the length limit was reached"
                    ),
                    RuntimeError(
                        "Could not parse response content as the length limit was reached"
                    ),
                    AIMessage(
                        content="full", response_metadata={"finish_reason": "stop"}
                    ),
                ]
            )
            config = make_config(max_tokens=1024)
            config["configurable"]["base_url"] = "https://example.com/v1/"
            config["configurable"]["model_provider"] = "openai"
            node = make_node(inst)
            node._last_prompt = StringPromptValue(text="hi")
            out = await node._invoke_llm(inst, StringPromptValue(text="hi"), config)

        # 1024 -> 2048 -> 4096 (doublings) -> jump (not capped by models.dev).
        assert inst.ainvoke.await_count == 4
        assert out.content == "full"
        assert config["configurable"]["max_tokens"] > MAX_OUTPUT_TOKENS_CEILING

    async def test_truncation_exception_at_catalog_cap_no_retry(self):
        """Raised truncation at the catalog output cap surfaces immediately."""
        with mock.patch(
            "klea_utils.llm.get_model_limits",
            return_value=ModelLimits(context=128000, output=4096),
        ):
            inst = mock.Mock()
            inst.ainvoke = mock.AsyncMock(
                side_effect=RuntimeError(
                    "Could not parse response content as the length limit was reached"
                )
            )
            config = make_config(max_tokens=4096)
            try:
                await self._invoke(inst, config)
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError")

        assert inst.ainvoke.await_count == 1
        assert config["configurable"]["max_tokens"] == 4096

    async def test_truncation_at_catalog_cap_no_retry(self):
        """Truncation at the catalog output cap does not waste a retry."""
        with mock.patch(
            "klea_utils.llm.get_model_limits",
            return_value=ModelLimits(context=128000, output=4096),
        ):
            inst = mock.Mock()
            inst.ainvoke = mock.AsyncMock(
                return_value=AIMessage(
                    content="cut", response_metadata={"finish_reason": "length"}
                )
            )
            config = make_config(max_tokens=4096)
            out = await self._invoke(inst, config)

        assert inst.ainvoke.await_count == 1
        assert out.content == "cut"
        assert config["configurable"]["max_tokens"] == 4096

    async def test_truncation_list_finish_reason(self):
        """List-form finish_reason (some providers) is handled."""
        inst = mock.Mock()
        inst.ainvoke = mock.AsyncMock(
            return_value=AIMessage(
                content="full",
                response_metadata={"finish_reason": ["stop"]},
            )
        )
        out = await self._invoke(inst)
        assert inst.ainvoke.await_count == 1
        assert out.content == "full"

    async def test_structured_output_fallback_on_rejection(self):
        """Structured-output rejection falls back to a plain invoke."""
        inst = mock.Mock()
        wrapped = mock.Mock()
        inst.with_structured_output.return_value = wrapped
        wrapped.ainvoke = mock.AsyncMock(
            side_effect=RuntimeError("response_format not supported")
        )
        inst.ainvoke = mock.AsyncMock(
            return_value=AIMessage(
                content="ok", response_metadata={"finish_reason": "stop"}
            )
        )

        node = make_node(inst, output_schema=_OutputSchema)
        config = make_config()
        out = await node._invoke_llm(inst, StringPromptValue(text="hi"), config)

        assert wrapped.ainvoke.await_count == 1
        assert inst.ainvoke.await_count == 1
        assert out.content == "ok"

    async def test_structured_output_non_rejection_propagates(self):
        """A non-rejection structured-path error is re-raised."""
        inst = mock.Mock()
        wrapped = mock.Mock()
        inst.with_structured_output.return_value = wrapped
        wrapped.ainvoke = mock.AsyncMock(
            side_effect=RuntimeError("Error code: 429 - rate limit")
        )
        inst.ainvoke = mock.AsyncMock(return_value=AIMessage(content="ok"))

        node = make_node(inst, output_schema=_OutputSchema)
        try:
            await node._invoke_llm(inst, StringPromptValue(text="hi"), make_config())
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")

        assert wrapped.ainvoke.await_count == 1
        assert inst.ainvoke.await_count == 0
