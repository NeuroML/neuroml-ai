#!/usr/bin/env python3
"""
Misc utils

File: neuroml_ai/utils.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import sys
import logging
import ollama


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

