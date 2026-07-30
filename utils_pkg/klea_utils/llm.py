#!/usr/bin/env python3
"""
LLM related utils

File: klea_rag/llm.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from __future__ import annotations

import logging
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any, NamedTuple, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from langgraph.types import RunnableConfig
from pydantic import BaseModel

from .plogging import mask_sensitive, setup_logger

logger = setup_logger(__name__)


class ParsedModelName(NamedTuple):
    """Parsed components of a model name string."""

    provider: str | None
    model_name: str
    suffix: str | None


def parse_model_name(raw: str) -> ParsedModelName:
    """Split a model name into provider, model identifier, and suffix.

    Follows the ``provider:model_id`` convention.  The provider is
    expected to be explicitly included; no provider inference is done.

    With three segments the third is treated as a ``suffix``
    (provider hint, model tag, base URL, etc.) *unless* the provider
    is ``ollama``, for which the second and third segments form the
    model name (``model_name:tag``).

    Examples:

    * ``ollama:bge-m3:latest`` -> provider=ollama, model=bge-m3:latest, suffix=None
    * ``huggingface:org/model:auto`` -> provider=huggingface, model=org/model, suffix=auto
    * ``custom:model:https://example.com/v1`` -> provider=custom, model=model, suffix=https://example.com/v1
    * ``openai:gpt-4o`` -> provider=openai, model=gpt-4o, suffix=None
    * ``bge-m3`` -> provider=None, model=bge-m3, suffix=None

    :param raw: Model name with optional provider prefix
    :returns: Parsed model name components
    """
    parts = raw.split(":", 2)

    if len(parts) == 1:
        return ParsedModelName(provider=None, model_name=raw, suffix=None)

    provider = parts[0].lower()

    if len(parts) == 2:
        return ParsedModelName(provider=provider, model_name=parts[1], suffix=None)

    if provider == "ollama":
        return ParsedModelName(
            provider=provider, model_name=f"{parts[1]}:{parts[2]}", suffix=None
        )

    return ParsedModelName(provider=provider, model_name=parts[1], suffix=parts[2])


def check_ollama_model(logger, model, exit=False):
    """Check if ollama model is available

    :param logger: logger instance
    :type logger: logging
    :param model: ollama model name
    :type model: str
    :param exit: if we should call sys.exit if check fails
    :type exit: bool
    :returns: None

    :throws ollama.ResponseError: if `model` is not available
    :throws ConnectionError: if cannot connect to an Ollama server

    """
    import ollama

    try:
        _ = ollama.show(model)
    except ollama.ResponseError:
        logger.error(f"Could not find ollama model: {model}")
        logger.error("Please ensure you have pulled the model")
        if exit:
            sys.exit(-1)
    except ConnectionError:
        logger.error("Could not connect to Ollama.")
        if exit:
            sys.exit(-1)


def parse_output_with_thought[TSchema: BaseModel](
    message: AIMessage, schema: type[TSchema]
) -> tuple[TSchema, str]:
    """Parse AI message with thought to a dict based on given schema"""
    # Lazy: JsonOutputParser pulls in langchain parsers
    from langchain_core.output_parsers import JsonOutputParser

    thought = ""
    answer = ""

    if isinstance(message.content, str):
        if "</think>" in message.content:
            splits = message.content.split("</think>")
            thought = splits[0].strip()
            answer = splits[1].strip()
        else:
            answer = message.content

        parser = JsonOutputParser()
        parser.pydantic_object = schema()
        result = parser.parse(answer)
    else:
        message.content = content_to_str(message.content)
        # Now a string -- re-run the string parsing above.
        if "</think>" in message.content:
            splits = message.content.split("</think>")
            thought = splits[0].strip()
            answer = splits[1].strip()
        else:
            answer = message.content

        parser = JsonOutputParser()
        parser.pydantic_object = schema()
        result = parser.parse(answer)

    logger.debug(f"{thought = }")
    logger.debug(f"{answer = }")

    return result, thought


def split_output_by_section(
    text: str, section_start_marker: str, section_end_marker: str | None = None
):
    """Split out thoughts and actual responses from AI responses"""
    if not text:
        logger.warning("Empty message.content. Nothing to do.")
        return "", ""

    if not section_start_marker:
        logger.warning("No starting marker. Nothing to do.")
        return "", ""

    if not section_end_marker:
        section_end_marker = None

    delimited, other = [], []

    # prepare pattern
    markers = [re.escape(section_start_marker)]
    if section_end_marker:
        markers.append(re.escape(section_end_marker))
    pattern = f"({'|'.join(markers)})"

    # split
    splits = re.split(pattern, text)

    # process splits
    # by default, we're outside the delimiters to begin with
    is_in = False

    # do we have both markers?
    found_start_marker = section_start_marker in text
    found_end_marker = section_end_marker in text if section_end_marker else False

    if not found_start_marker and not found_end_marker:
        logger.debug("No markers found. Nothing to do.")
        return "", text

    # end marker, but no start: we start inside the delimted region
    if found_end_marker and not found_start_marker:
        is_in = True

    for part in splits:
        if part == section_start_marker:
            is_in = True
        elif part == section_end_marker:
            is_in = False
        else:
            if is_in:
                delimited.append(part)
            else:
                other.append(part)

    delimited_text = "".join(delimited).strip()
    other_text = "".join(other).strip()

    # Add notes
    if found_start_marker and (section_end_marker and not found_end_marker):
        other_text += "\nNOTE: NO END MARKER FOUND"
    elif found_end_marker and not found_start_marker:
        other_text += "\nNOTE: NO START MARKER FOUND"

    return delimited_text, other_text


def content_to_str(
    content: str | list[dict | str] | None,
) -> str:
    """Normalise an ``AIMessage.content`` value to a plain string.

    AIMessage.content can be a plain string, a list of content blocks
    (when the LLM returns tool calls or structured output), or ``None``.
    This helper always returns a string suitable for downstream text
    processing (regex, ``in`` checks, prompt interpolation, etc.).

    :param content: The raw ``.content`` value from an AIMessage.
    :returns: A plain string.
    """
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content)


def prompt_value_to_messages(prompt: PromptValue) -> list[dict]:
    """Convert a ``PromptValue`` to a clean list of message dicts.

    Each dict has ``role`` and ``content`` keys, suitable for JSON
    serialisation in the inspector debug panel.

    :param prompt: The LangChain ``PromptValue`` (filled, variables
        already substituted).
    :returns: A list of ``{"role": "...", "content": "..."}`` dicts.
    """
    return [
        {"role": msg.type, "content": content_to_str(msg.content)}
        for msg in prompt.to_messages()
    ]


def extract_llm_output_content(output: AIMessage | dict) -> str:
    """Extract plain-text content from an LLM output.

    Handles both ``AIMessage`` (non-structured output) and
    ``dict`` (structured output with ``raw`` / ``parsed`` keys):

    * ``AIMessage`` -- returns ``content_to_str(message.content)``.
    * ``dict`` -- extracts the ``raw`` ``AIMessage`` from a structured
      output response and returns its content; falls back to
      ``output["parsed"]`` and finally ``str(output)``.

    :param output: The raw output from ``llm.invoke()``.
    :returns: A plain-text string.
    """
    if isinstance(output, AIMessage):
        return content_to_str(output.content)
    if isinstance(output, dict):
        raw = output.get("raw")
        if isinstance(raw, AIMessage):
            return content_to_str(raw.content)
        parsed = output.get("parsed")
        if parsed is not None:
            return str(parsed)
    return str(output)


def check_model_works(model, timeout=30, retries=5):
    """Check if a model works since it is not tested when loaded"""
    assert timeout >= 0

    # Pick the right token-limit param for the provider so we keep the health
    # check cheap without triggering warnings about unknown kwargs.
    llm_type = getattr(model, "_llm_type", "")
    if "huggingface" in llm_type:
        token_param = "max_new_tokens"
    elif "ollama" in llm_type:
        token_param = "num_predict"
    else:
        token_param = "max_tokens"

    configurable = {token_param: 5}

    # One-shot probe for response_format/json_schema support.  A transient
    # failure here is harmless  ---  we just fall back to prompt-based structured
    # output (already in _get_system_prompt).  Every model will pass the plain
    # ping retry loop below even if this probe fails.
    try:
        from pydantic import BaseModel

        class _ProbeSchema(BaseModel):
            answer: str

        probe = model.with_structured_output(
            _ProbeSchema, method="json_schema", include_raw=False
        )
        probe.invoke(
            "ping",
            config={"timeout": timeout, "configurable": configurable},
        )
        model._supports_structured_output = True
        logger.info("Model supports structured output")
    except Exception:
        model._supports_structured_output = False

    for attempt in range(retries):
        logger.info(f"Checking model. Attempt #{attempt + 1}/{retries}")
        try:
            result = model.invoke(
                "ping",
                config={
                    "timeout": timeout,
                    "configurable": configurable,
                },
            )
            logger.info(f"Model available (attempt {attempt + 1}/{retries}): {result}")
            return True, f"Model available (attempt {attempt + 1}/{retries})"
        except StopIteration as e:
            return (
                False,
                f"{e.__class__.__name__}: check if any inference providers are available for the selected model",
            )
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e.__str__()}"
            logger.warning(
                f"Attempt #{attempt + 1}/{retries}: model unavailable: {error_msg}"
            )
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                logger.error(f"Model unavailable after {retries} attempts: {error_msg}")
                return (
                    False,
                    f"Model unavailable after {retries} attempts: {error_msg}",
                )

    return False, "Unknown error"


def setup_embedding(model_name_full, logger):
    # Lazy: init_embeddings and HuggingFaceEndpointEmbeddings pull in
    # langchain provider packages that may not be installed
    from langchain.embeddings import init_embeddings

    parsed = parse_model_name(model_name_full)

    if parsed.provider == "custom":
        logger.debug(
            "Using openai-compatible embedding model %s at endpoint %s",
            parsed.model_name,
            parsed.suffix,
        )
        model_var = init_embeddings(
            parsed.model_name,
            provider="openai",
            base_url=parsed.suffix,
        )
    elif parsed.provider == "huggingface":
        # "local" suffix -> HuggingFaceEmbeddings (local sentence-transformers).
        # Any other suffix -> HuggingFaceEndpointEmbeddings (HF Inference API).
        if parsed.suffix == "local":
            logger.debug(
                "Using huggingface local embedding model: %s",
                parsed.model_name,
            )
            from langchain_huggingface import HuggingFaceEmbeddings

            model_var = HuggingFaceEmbeddings(model_name=parsed.model_name)
        else:
            logger.debug(
                "Using huggingface endpoint embedding model: %s (provider: %s)",
                parsed.model_name,
                parsed.suffix or "auto",
            )
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            model_var = HuggingFaceEndpointEmbeddings(
                model=parsed.model_name,
                provider=parsed.suffix or "auto",
                task="feature-extraction",
            )
    else:
        logger.debug("Using embedding model: %s", model_name_full)
        if parsed.provider == "ollama":
            check_ollama_model(logger, parsed.model_name)
        model_var = init_embeddings(model_name_full)

    assert model_var
    logger.info(f"Using embedding model: {model_name_full}")
    return model_var


def looks_like_structured_output_error(exc: Exception) -> bool:
    """Heuristic: is this exception caused by structured output rejection?

    Some providers (e.g. certain custom OpenAI-compatible endpoints) do
    not support the ``response_format`` / ``json_schema`` parameter.
    Rather than failing hard, we detect this by checking the error for
    known indicators and fall back to prompt-based structured output.

    Real errors (auth failure, model-not-found, rate limits) are left to
    propagate  ---  they use different HTTP status codes and error strings
    not matched by these heuristics.
    """
    msg = str(exc).lower()

    # OpenAI-style BadRequestError with structured-output rejection
    if "400" in msg and "invalid_request_error" in msg:
        return True

    # Provider explicitly rejects the structured output parameter
    if any(kw in msg for kw in ("response_format", "json_schema")):
        return True

    # LangChain's own message when structured output fails
    if "structured output" in msg:
        return True

    return False


# Extra fields accepted by provider model constructors that are not part
# of the Pydantic model class (e.g. kwargs passed to a factory method).
_PROVIDER_EXTRA_FIELDS: dict[str, set[str]] = {
    # ChatHuggingFace.from_model_id() accepts backend, provider, etc.
    # which flow through to HuggingFaceEndpoint but are not fields on
    # ChatHuggingFace itself.
    "huggingface": {"backend", "provider"},
}


def get_provider_allowed_fields(provider: str) -> set[str]:
    """Return the set of init-param names accepted by a given provider's model class.

    Uses LangChain's internal provider registry to look up the Pydantic
    model class and introspect its fields (including aliases so that
    both ``api_key`` and ``openai_api_key`` pass through).

    Falls back to an empty set if the provider is not registered in
    LangChain's built-in providers.  Raises ``ImportError`` if the
    provider's integration package is not installed  ---  callers should
    handle this at configuration time, not silently fall through.

    The caller should always include ``{"model", "model_provider"}`` on
    top of the returned set since those are consumed by
    ``_ConfigurableModel`` before reaching the model constructor.
    """
    from langchain.chat_models.base import _get_chat_model_creator

    try:
        creator = _get_chat_model_creator(provider)
    except ValueError:
        # Provider not in LangChain's built-in registry  ---  not an error,
        # the caller will include model/model_provider which is sufficient.
        return set()

    cls = getattr(creator, "keywords", {}).get("cls")
    if cls is None:
        return set()

    fields: set[str] = set()
    for name, field in cls.model_fields.items():
        fields.add(name)
        if field.alias:
            fields.add(field.alias)
    return fields | _PROVIDER_EXTRA_FIELDS.get(provider, set())


def create_configurable_model(logger: logging.Logger):
    """Set up a configurable chat model.

    Creates a ``_ConfigurableModel`` with no default model.  Model,
    provider, and all other parameters (``base_url``, ``api_key``,
    ``temperature``, etc.) are specified per-invoke via the
    ``config["configurable"]`` dict passed to ``ainvoke()``.

    This enables runtime model switching  ---  each ``ainvoke()`` call
    creates a fresh underlying model instance for the given provider,
    so there is no stale configuration leakage between calls.

    The lookup function ``check_model_works`` is deliberately **not**
    called here  ---  we prefer a "leap before you look" approach so that
    startup is fast and model availability is checked only at query time.
    """
    from langchain.chat_models import init_chat_model

    model_var = init_chat_model(
        model=None,
        configurable_fields="any",
    )
    logger.info("Configurable chat model created (provider/model set per invoke)")

    return model_var


class LLMModel(BaseModel):
    """Container for a single LLM model instance and its runtime configuration.

    ``instance`` holds the model object (typically a ``_ConfigurableModel``
    returned by ``init_chat_model``).  ``role_defaults`` stores role-wide
    default parameters (e.g. ``max_tokens``, ``temperature``) that apply to
    every node sharing this role, unless overridden by node or user config.

    ``build_config()`` performs a four-layer merge:

    **Layer 0**  ---  ``role_defaults``: role-wide parameters (e.g.
    ``{"max_tokens": 4096}``).

    **Layer 1**  ---  ``model_name``: the default model identifier from
    the graph config.

    **Layer 2**  ---  ``context_overrides``: per-request fields from the API
    (``model``, ``api_key``, etc.).  Only applied when ``modifiable=True``,
    and skipping any keys frozen by node defaults.

    **Layer 3**  ---  ``node_defaults``: frozen per-node defaults (always win).

    ``modifiable`` controls whether the model can be changed at runtime
    (both the API and web UI reject modifications to locked roles).
    Set to ``False`` to lock a role (e.g. guard) against user overrides
    in managed deployments.
    """

    model_name: str = ""
    instance: Any
    role_defaults: dict[str, Any] = {}
    modifiable: bool = True

    def build_config(
        self,
        context_overrides: dict[str, Any] | None = None,
        node_defaults: dict[str, Any] | None = None,
    ) -> RunnableConfig:
        """Merge up to four layers of model configuration into a ``RunnableConfig``.

        Layer order (lowest -> highest priority):

        0. ``self.role_defaults``   ---   role-wide parameters
        1. ``self.model_name``      ---   role model from graph config
        2. ``context_overrides``    ---   per-request user overrides
        3. ``node_defaults``        ---   frozen per-node defaults

        :param context_overrides: Per-request fields from the API
            (e.g. ``model``, ``api_key``).  Only applied when
            ``self.modifiable is True``, and skipping any keys
            present in ``node_defaults``.
        :param node_defaults: Frozen per-node defaults (e.g.
            ``{"temperature": 0.3}``).  Always win.
        :returns: A ``RunnableConfig`` with the ``configurable`` key
            populated.
        """
        logger.debug(
            f"{self.modifiable = }\n"
            f"{self.role_defaults = }\n"
            f"{self.model_name = }\n"
            f"{mask_sensitive(context_overrides or {}) = }\n"
            f"{node_defaults = }"
        )

        # Layer 0: role-wide defaults from graph config
        overrides: dict[str, Any] = dict(self.role_defaults)
        logger.debug(f"Layer 0 (role defaults):\n{mask_sensitive(overrides) = }")

        # Layer 1: role model identifier
        overrides["model"] = self.model_name
        logger.debug(f"Layer 1 (model):\n{mask_sensitive(overrides) = }")

        # Layer 2: context overrides (only if modifiable).
        # Skip any keys the node has frozen in model_defaults.
        if self.modifiable and context_overrides:
            for k, v in context_overrides.items():
                if node_defaults and k in node_defaults:
                    logger.debug(
                        f"Skipping context override '{k}' (frozen by node defaults)"
                    )
                    continue
                overrides[k] = v
            logger.debug(f"Layer 2 (context):\n{mask_sensitive(overrides) = }")

        # Layer 3: node defaults  ---  always win
        if node_defaults:
            overrides.update(node_defaults)
            logger.debug(f"Layer 3 (node defaults):\n{mask_sensitive(overrides) = }")

        # Parse the final model string into LangChain-compatible components.
        # Klea stores full provider-prefixed model strings internally (e.g.
        # "custom:gpt-4o:https://endpoint/v1"), but the _ConfigurableModel
        # expects the bare model name plus separate model_provider and base_url.
        # parse_model_name is defined in this module  ---  no lazy import needed.
        parsed = parse_model_name(overrides["model"])
        overrides["model"] = parsed.model_name
        if parsed.provider == "custom":
            overrides["model_provider"] = "openai"
        elif parsed.provider:
            overrides["model_provider"] = parsed.provider
        if parsed.suffix:
            overrides["base_url"] = parsed.suffix
        logger.debug(f"After model string parse:\n{mask_sensitive(overrides) = }")

        # Inject HuggingFace from_model_id kwargs derived from the model
        # string suffix.  These are not fields on ChatHuggingFace itself
        # (they flow through to HuggingFaceEndpoint) so they'd be filtered
        # out later  ---  we set them here so they survive provider filtering.
        if overrides.get("model_provider") == "huggingface" and parsed.suffix:
            if parsed.suffix == "local":
                overrides.setdefault("backend", "pipeline")
            else:
                overrides.setdefault("backend", "endpoint")
                overrides.setdefault("provider", parsed.suffix)
            logger.debug(
                f"HuggingFace kwargs injected:\n{mask_sensitive(overrides) = }"
            )

        # Map generic "api_key" to provider-specific token field names so a
        # single user-facing field works across all providers.
        if "api_key" in overrides:
            overrides.setdefault("huggingfacehub_api_token", overrides["api_key"])
            logger.debug(f"After api_key mapping:\n{mask_sensitive(overrides) = }")

        # Wrap in the "configurable" key expected by _ConfigurableModel.
        return cast(RunnableConfig, {"configurable": overrides})


def get_last_n_conversations(
    all_messages, start: int = 0, stop: int | None = None
) -> tuple[str, list[HumanMessage], list[AIMessage]]:
    """Get recent converstations between start and stop indices

    :param all_messages: all the messages
    :param start: start index
    :param stop: stop index
    :returns: (conversation, list of human messages, list of ai messages)

    """
    conv_messages = list(
        filter(
            lambda x: isinstance(x, (HumanMessage, AIMessage)),
            all_messages[start:stop],
        )
    )
    human_messages = []
    ai_messages = []
    conversation = ""
    for msg in conv_messages:
        if isinstance(msg, HumanMessage):
            conversation += f"{msg.pretty_repr()}"
            human_messages.append(msg)
        else:
            conversation += f": {msg.pretty_repr()}"
            ai_messages.append(msg)

    logger.debug(f"{conversation = }")

    return (
        conversation.replace("{", "{{").replace("}", "}}"),
        human_messages,
        ai_messages,
    )


# TODO: num_history_messages is currently part of the orchestrator, but nodes wont have access to it.
def add_memory_to_prompt(context_summary: str, messages, num_history_messages) -> str:
    """Add memory to system prompt.

    Adds the context summary and recent conversation

    :param state: agent state
    :returns: "memory" string to add to the system prompt

    """
    ret_string = ""

    directive = dedent("""

        ## Previous context

        IMPORTANT:

        - Consider both the latest user message AND the conversation history.

    """)

    if len(context_summary):
        ret_string += dedent(f"""

        ### Context summary

        Here is a concise summary of the past conversation to maintain continuity:

        {context_summary}

        """)

    conversation, _, _ = get_last_n_conversations(
        messages, (-1 * num_history_messages), None
    )
    if len(conversation):
        ret_string += dedent(f"""
        ### Recent messages

        Here are recent messages between the user and the assistant:

        {conversation}

        -----
        """)

    if len(ret_string):
        ret_string += directive

    return ret_string


@lru_cache(maxsize=10000)
def load_prompt(prompt_name: str, prompt_registry_location: str):
    """Load a prompt from file called prompt_name.md

    :param str: prompt file name
    :param prompt_registry_location: location of prompts folder/registry
    :returns: loaded prompt text

    """
    prompt_path = Path(f"{prompt_registry_location}/{prompt_name}.md")
    if not prompt_path.exists():
        raise FileNotFoundError(f"{prompt_path} was not found")

    with open(prompt_path, "r") as f:
        return f.read()
