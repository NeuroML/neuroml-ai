#!/usr/bin/env python3
"""
Misc utils

File: neuroml_ai/utils.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
import time
import sys
import logging
import ollama
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser


class LoggerNotInfoFilter(logging.Filter):
    """Allow only non INFO messages"""

    def filter(self, record):
        return record.levelno != logging.INFO


class LoggerInfoFilter(logging.Filter):
    """Allow only INFO messages"""

    def filter(self, record):
        return record.levelno == logging.INFO


logger_formatter_info = logging.Formatter(
    "%(name)s (%(levelname)s) >>> %(message)s\n\n"
)
logger_formatter_other = logging.Formatter(
    "%(name)s (%(levelname)s) in '%(funcName)s' >>> %(message)s\n\n"
)


def check_ollama_model(logger, model):
    """Check if ollama model is available

    :param model: ollama model name
    :type model: str
    :returns: None

    :throws ollama.ResponseError: if `model` is not available
    :throws ConnectionError: if cannot connect to an Ollama server

    """
    try:
        _ = ollama.show(model)
    except ollama.ResponseError:
        logger.error(f"Could not find ollama model: {model}")
        logger.error("Please ensure you have pulled the model")
        sys.exit(-1)
    except ConnectionError:
        logger.error("Could not connect to Ollama.")
        sys.exit(-1)


def parse_output_with_thought(message: AIMessage, schema) -> dict:
    """Parse AI message with thought to a dict based on given schema"""
    if "</think>" in message.content:
        splits = message.content.split("</think>")
        answer = splits[1].strip()
    else:
        answer = message.content

    parser = JsonOutputParser()
    parser.pydantic_object = schema()
    result = parser.parse(answer)
    return result


def split_thought_and_output(message: AIMessage):
    """Split out thoughts and actual responses from AI responses"""
    if "</think>" in message.content:
        splits = message.content.split("</think>")
        answer = splits[1].strip()
        thoughts = splits[1].strip()
    else:
        answer = message.content.strip()
        thoughts = ""
    return thoughts, answer


def check_model_works(model, timeout=30, retries=3):
    """Check if a model works since it is not tested when loaded"""
    for attempt in range(retries):
        try:
            # Use a very simple prompt with short max_tokens
            _ = model.invoke("test", config={"timeout": timeout})
            return True, f"Model available (attempt {attempt + 1})"
        except Exception as e:
            error_msg = str(e)
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                return False, f"Model unavailable after {retries} attempts: {error_msg}"
    return False, "Unknown error"

def setup_llm(model_name_full, logger, embedding=False):
    """Set up a chat model"""
    if model_name_full.lower().startswith("huggingface:"):
        from langchain_huggingface import HuggingFaceEndpoint

        _, model_name, provider = model_name_full.split(":")
        logger.debug(f"Using huggingface model: {model_name}")

        hf_token = os.environ.get("HF_TOKEN_NML_AI", None)
        logger.debug(f"{hf_token =}")
        assert hf_token

        llm = HuggingFaceEndpoint(
            repo_id=f"{model_name}",
            provider="auto",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=hf_token
        )

        if not embedding:
            model_var = init_chat_model(
                model_name,
                model_provider="huggingface",
                llm=llm,
                configurable_fields=("temperature"),
            )
        else:
            model_var = init_embeddings(model_name, model_provider="huggingface", llm=llm)
    else:
        if model_name_full.lower().startswith("ollama:"):
            check_ollama_model(
                logger, model_name_full.lower().replace("ollama:", "")
            )

        if not embedding:
            model_var = init_chat_model(
                model_name_full, configurable_fields=("temperature")
            )
        else:
            model_var = init_embeddings(model_name_full)

    assert model_var

    if not embedding:
        state, msg = check_model_works(model_var)
        assert state
        logger.info(f"Using chat model: {model_name_full}")
    else:
        logger.info(f"Using embedding model: {model_name_full}")

    return model_var
