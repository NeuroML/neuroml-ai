#!/usr/bin/env python3
"""
Base node classes for LangGraph processing nodes

File: klea_utils/nodes/base.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from __future__ import annotations

import inspect
import json
import logging
from functools import cached_property
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

from langchain.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.utils.function_calling import convert_to_json_schema
from pydantic import BaseModel

from klea_utils.graph.base import model_overrides_ctx
from klea_utils.plogging import mask_sensitive

from ..errors import LLMInvocationErrorCategory, PromptTemplateError
from ..llm import (
    add_memory_to_prompt,
    classify_llm_invocation_error,
    get_provider_allowed_fields,
    load_prompt,
    parse_output_with_thought,
)
from .abstract import AbstractLLMNode


class BaseLLMNode[TSchema: BaseModel](AbstractLLMNode[TSchema]):
    """Base class for LangGraph nodes that load prompts from files.

    Extends AbstractLLMNode with:
    - File-based prompt loading via load_prompt()
    - Optional memory support (appends memory content to system prompt)
    - Auto-derived prompt registry location from subclass file path

    Prompt files are expected to be named ``{prefix}_system.md`` and
    ``{prefix}_user.md``.

    Subclasses can override ``prompt_prefix`` or ``prompt_registry_location``
    via the setter if the defaults (lowercase class name / sibling ``prompts/``)
    are not appropriate.
    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        llm_models: dict[str, Any],
        output_schema: type[TSchema] | None,
        memory: bool = False,
        num_history_messages: int = 10,
    ):
        """Initialize with file-based prompt loading and memory support.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param llm_models: ``{role: LLMModel}`` dict (from ``BaseLangGraph.llm_models``)
        :param output_schema: Pydantic schema for structured output
        :param memory: Whether to append memory content to the system prompt
        :param num_history_messages: Number of recent messages to include
        """
        super().__init__(logger, label, llm_models, output_schema=output_schema)

        self._prompt_prefix: str | None = None
        self._prompt_registry_location: Path | None = None
        self.memory = memory
        self.num_history_messages = num_history_messages

    @property
    def prompt_prefix(self) -> str:
        """Return the prompt file prefix.

        Falls back to the lowercase class name if not explicitly set.
        """
        if self._prompt_prefix is not None:
            return self._prompt_prefix
        return self.__class__.__name__

    @prompt_prefix.setter
    def prompt_prefix(self, value: str) -> None:
        """Set the prompt file prefix."""
        self._prompt_prefix = value

    @property
    def prompt_registry_location(self) -> Path:
        """Return path to the prompts directory.

        Falls back to a sibling ``prompts/`` directory relative to the
        subclass file if not explicitly set.
        """
        if self._prompt_registry_location is not None:
            return self._prompt_registry_location

        subclass_file = inspect.getfile(self.__class__)
        loc = Path(subclass_file).parent / "prompts"
        self.logger.debug(f"No prompt registry location set. Falling back to {loc}")
        return loc

    @prompt_registry_location.setter
    def prompt_registry_location(self, value: Path) -> None:
        """Set the prompts directory path."""
        self._prompt_registry_location = value

    def _pre_exec(self, state: BaseModel) -> bool:
        """Pre-execution check. Override to conditionally skip node execution.

        Return False to skip execution (returns empty dict).
        Return True (default) to proceed with the standard flow.
        """
        return True

    @property
    def output_schema(self) -> type[TSchema] | None:
        """Return Pydantic schema for structured output if required"""
        return self._output_schema

    @output_schema.setter
    def output_schema(self, value: type[TSchema] | None) -> None:
        """Set Pydantic schema for structured output"""
        self._output_schema = value

    @cached_property
    def output_schema_json(self) -> dict[str, Any]:
        """Return JSON schema string for use in prompts."""
        return convert_to_json_schema(self.output_schema) if self.output_schema else {}

    def _configure_llm(self) -> tuple[Runnable, RunnableConfig]:
        """Configure LLM and build per-invoke config.

        Returns the raw ``instance`` (a ``_ConfigurableModel``) without
        wrapping it  ---  structured output is applied inside ``_invoke_llm``
        so that providers that reject ``response_format`` can fall back to
        prompt-based structured output.

        :returns: (llm_instance, config_dict) where config_dict is a
            ``RunnableConfig`` with ``configurable`` populated.
        """
        inst = self._llm_entry.instance
        config = self._build_invoke_config()
        self.logger.debug(f"{self.model_type = }\n{config = }")
        return inst, config

    def _build_invoke_config(self) -> RunnableConfig:
        """Build the per-invoke RunnableConfig.

        Delegates the full merge (role defaults -> context overrides ->
        node defaults, including model-string parsing) to
        ``LLMModel.build_config()``, then applies provider field
        filtering to strip fields invalid for the resolved provider.
        """
        role_overrides = model_overrides_ctx.get().get(self.model_type, {})
        self.logger.debug(
            f"{mask_sensitive(model_overrides_ctx.get()) = }\n"
            f"{self.model_type = }\n"
            f"{mask_sensitive(role_overrides) = }\n"
            f"{self.model_defaults = }"
        )

        # Delegate merge + model parsing to LLMModel.
        config = self._llm_entry.build_config(
            context_overrides=role_overrides,
            node_defaults=self.model_defaults,
        )

        # Get the merged configurable dict for provider field filtering.
        overrides: dict[str, Any] = config["configurable"]

        # --- Provider field filtering ---
        active_provider = overrides.get("model_provider") or "openai"
        provider_allowed = get_provider_allowed_fields(active_provider)
        allowed = provider_allowed | {"model", "model_provider"}
        overrides = {k: v for k, v in overrides.items() if k in allowed}
        self.logger.debug(
            f"After provider field filtering ({active_provider = }):\n{mask_sensitive(overrides) = }"
        )

        return cast(RunnableConfig, {"configurable": overrides})

    async def _invoke_llm(
        self, llm: Runnable, prompt: PromptValue, config: RunnableConfig
    ) -> AIMessage | dict[str, Any]:
        """Async invoke LLM with optional structured output + fallback.

        Wraps the configurable model with ``with_structured_output``
        when an output schema exists.  If the provider rejects the
        ``response_format`` parameter (e.g. some custom OpenAI-compatible
        endpoints), falls back to a plain invoke  ---  the prompt already
        contains the JSON schema as text instructions.
        """
        inst = self._llm_entry.instance
        if self.output_schema:
            llm_wrapped = inst.with_structured_output(
                self.output_schema, method="json_schema", include_raw=True
            )
            try:
                output = await llm_wrapped.ainvoke(prompt, config=config)
            except Exception as exc:
                if (
                    classify_llm_invocation_error(exc)
                    is LLMInvocationErrorCategory.STRUCTURED_OUTPUT_REJECTED
                ):
                    self.logger.warning(
                        "Structured output not supported, falling back to prompt-based"
                    )
                    output = await inst.ainvoke(prompt, config=config)
                else:
                    raise
        else:
            output = await inst.ainvoke(prompt, config=config)
        self.logger.debug(f"{output = }")
        return output

    def _process_output(self, output: AIMessage | dict[str, Any]) -> Any:
        """Common output processing with error handling"""
        result: TSchema | AIMessage | None = None
        schema = self.output_schema

        if schema:
            # but answer is returned as message instead of json/dict
            if isinstance(output, AIMessage):
                result, _ = parse_output_with_thought(output, schema)
                if isinstance(result, dict):
                    result = schema(**result)
            else:
                assert isinstance(output, dict)
                if output["parsing_error"]:
                    self.logger.warning(
                        f"LLM parsing error, using fallback: {output['parsing_error']}"
                    )
                    result, _ = parse_output_with_thought(output["raw"], schema)
                else:
                    result = output["parsed"]
                    if isinstance(result, dict):
                        result = schema(**result)
                    else:
                        if not isinstance(result, schema):
                            self.logger.critical(
                                f"Unexpected output type: {type(result)}"
                            )
                            result = self._get_default_error_result()

            self.logger.debug(f"Processed output: {result}")
        else:
            assert isinstance(output, AIMessage)
            result = output
            self.logger.debug(
                f"No output schema. Returning unprocessed output: {result}"
            )

        return result

    def _invoke_prompt(
        self, prompt_template: ChatPromptTemplate, variables: Any | dict[str, Any]
    ) -> PromptValue:
        """Format prompt with state-specific parameters"""
        prompt = prompt_template.invoke(variables)
        self.logger.debug(f"{prompt =}")
        return prompt

    def _get_system_prompt(self, state: BaseModel) -> str:
        """Load system prompt from file, optionally adding memory summary and output schema."""
        system_prompt = self._load_prompt_file(f"{self.prompt_prefix}_system")

        if self.output_schema:
            # we pass this as part of the prompt because not all models/end
            # points support passing schemas separately, or respect `with
            # structured output`.  This is the safest, most general way of
            # doing it.
            system_prompt += dedent(
                f"""
                ## Output schema (strict)

                Respond in JSON following this schema:

                {json.dumps(self.output_schema_json).replace("{", "{{").replace("}", "}}")}
                """
            )

        if self.memory:
            memory_addition = self._get_memory_addition(state)
            system_prompt += memory_addition

        self.logger.debug(f"{system_prompt =}")
        return system_prompt

    def _get_human_prompt(self, state: BaseModel) -> str:
        """Load human prompt from file."""
        human_prompt = self._load_prompt_file(f"{self.prompt_prefix}_user")

        self.logger.debug(f"{human_prompt =}")
        return human_prompt

    def _load_prompt_file(self, prompt_name: str) -> str:
        """Load a prompt file from the registry.

        :param prompt_name: Prompt file name (without extension)
        :returns: Prompt text content
        """
        return load_prompt(
            prompt_name=prompt_name,
            prompt_registry_location=self.prompt_registry_location,
        )

    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate with system and human messages."""
        if len(system_prompt) and len(human_prompt):
            prompt_template = ChatPromptTemplate(
                [("system", system_prompt), ("human", human_prompt)]
            )
        elif len(system_prompt) and not len(human_prompt):
            prompt_template = ChatPromptTemplate([("system", system_prompt)])
        elif len(human_prompt) and not len(system_prompt):
            prompt_template = ChatPromptTemplate([("human", human_prompt)])
        else:
            raise PromptTemplateError(
                "No prompts provided. Cannot create prompt template!"
            )

        self.logger.debug(f"{prompt_template =}")
        return prompt_template

    def _get_memory_addition(self, state: BaseModel) -> str:
        """Hook for subclasses to append memory content into the system prompt.

        Override this method to provide memory-specific content.
        The default implementation returns an empty string.
        """
        return add_memory_to_prompt(
            messages=state.messages,  # type: ignore
            context_summary=state.context_summary,  # type: ignore
            num_history_messages=self.num_history_messages,
        )
