#!/usr/bin/env python3
"""
LLM related utils

File: klea_rag/llm.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import NamedTuple, Type

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    message: AIMessage, schema: Type[TSchema]
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

    for attempt in range(retries):
        print(f"Checking model. Attempt #{attempt + 1}/{retries}")
        try:
            result = model.invoke(
                "ping",
                config={
                    "timeout": timeout,
                    "configurable": configurable,
                },
            )
            print(f"Model available (attempt {attempt + 1}/{retries}): {result}")
            return True, f"Model available (attempt {attempt + 1}/{retries})"
        except StopIteration as e:
            return (
                False,
                f"{e.__class__.__name__}: check if any inference providers are available for the selected model",
            )
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e.__str__()}"
            print(f"Attempt #{attempt + 1}/{retries}: model unavailable: {error_msg}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print(f"Model unavailable after {retries} attempts: {error_msg}")
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

    if parsed.provider == "huggingface":
        logger.debug(f"Using huggingface model: {parsed.model_name}")

        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        hf_token = os.environ.get("HF_TOKEN", None)
        assert hf_token

        model_var = HuggingFaceEndpointEmbeddings(
            model=parsed.model_name,
            provider=parsed.suffix or "auto",
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
    else:
        if parsed.provider == "ollama":
            check_ollama_model(logger, parsed.model_name)
        model_var = init_embeddings(model_name_full)

    assert model_var
    logger.info(f"Using embedding model: {model_name_full}")
    return model_var


# Shared configurable fields for init_chat_model. Common to all providers;
# provider-specific fields are appended per-branch.
_COMMON_CONFIG_FIELDS: tuple[str, ...] = (
    "model",
    "model_provider",
    "temperature",
    "max_tokens",
    "api_key",
    "base_url",
)


def setup_llm(model_name_full: str, logger: logging.Logger):
    """Set up a chat model"""
    # Lazy: init_chat_model and huggingface classes pull in provider
    # packages that may not be installed (langchain-ollama, langchain-huggingface, etc.)
    from langchain.chat_models import init_chat_model

    parsed = parse_model_name(model_name_full)

    # fall back to openai for custom end points, assuming they are openai compatible
    if parsed.provider == "custom":
        model_var = init_chat_model(
            parsed.model_name,
            model_provider="openai",
            configurable_fields=_COMMON_CONFIG_FIELDS + ("model_provider",),
            base_url=parsed.suffix,
        )
    elif parsed.provider == "huggingface":
        hf_token = os.environ.get("HF_TOKEN")
        assert hf_token

        logger.debug(f"Using huggingface model: {parsed.model_name}")
        logger.debug(f"Got HuggingFace Token: {hf_token[:2]}...{hf_token[-2:]}")

        # init_chat_model for huggingface calls ChatHuggingFace.from_model_id(),
        # which for backend="endpoint" creates a HuggingFaceEndpoint internally.
        # All remaining kwargs flow through to HuggingFaceEndpoint.__init__().
        # "model", "model_provider", and "temperature" are configurable at
        # runtime so the user can switch model/repo_id without restarting.
        model_var = init_chat_model(
            parsed.model_name,
            model_provider="huggingface",
            configurable_fields=_COMMON_CONFIG_FIELDS
            + ("provider", "backend", "huggingfacehub_api_token", "max_new_tokens"),
            backend="endpoint",
            provider=parsed.suffix or "auto",
            huggingfacehub_api_token=hf_token,
            max_new_tokens=32768,
            do_sample=False,
            repetition_penalty=1.03,
        )
    else:
        extra_fields: tuple[str, ...] = ()
        if parsed.provider == "ollama":
            check_ollama_model(logger, parsed.model_name)
            extra_fields = ("num_predict",)

        model_var = init_chat_model(
            model_name_full,
            configurable_fields=_COMMON_CONFIG_FIELDS + extra_fields,
        )

    state, msg = check_model_works(model_var, timeout=60)
    if not state:
        # handle special case where some models do not support "cheapest" on HF
        if (
            parsed.provider == "huggingface"
            and "Provider 'cheapest' not supported" in msg
        ):
            logger.error(f"Model does not work: {state}, {msg}")
            logger.debug("Replacing 'cheapest' with 'auto' and retrying")
            return setup_llm(model_name_full.replace(":cheapest", ":auto"), logger)

        logger.error(f"Model does not work: {state}, {msg}")
        assert state

    logger.info(f"Using chat model: {model_name_full}")

    return model_var


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
