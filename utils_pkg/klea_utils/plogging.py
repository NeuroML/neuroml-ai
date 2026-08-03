#!/usr/bin/env python3
"""
Logging related utils

File: klea_utils/plogging.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

#: Klea logger namespaces that are turned up to DEBUG by
#: ``setup_root_logger``.  Everything else (third-party libraries) inherits
#: the root logger's INFO level, so their DEBUG output is filtered at the
#: source without having to enumerate them.
KLEA_LOG_NAMESPACES = (
    "klea_utils",
    "klea_rag",
    "klea_code",
    "neuroml_mcp",
)


class LoggerNotInfoFilter(logging.Filter):
    """Allow only non INFO messages"""

    def filter(self, record):
        return record.levelno != logging.INFO


class LoggerInfoFilter(logging.Filter):
    """Allow only INFO messages"""

    def filter(self, record):
        return record.levelno == logging.INFO


logger_formatter_info = logging.Formatter(
    "%(asctime)s %(name)s (%(levelname)s) >>> %(message)s\n\n"
)
logger_formatter_other = logging.Formatter(
    "%(asctime)s %(name)s (%(levelname)s) in '%(funcName)s' >>> %(message)s\n\n"
)
logger_formatter_file = logging.Formatter(
    "%(asctime)s %(name)s (%(levelname)s) in '%(funcName)s' >>> %(message)s"
)


def setup_root_logger(
    app_name: str,
    stderr_level: int = logging.DEBUG,
    log_dir: str | Path | None = None,
) -> logging.Logger:
    """Configure the root logger once per process.

    Idempotent: if the root logger already has handlers, this is a no-op
    and the existing configuration is returned unchanged.

    Adds, on the root logger:

    * a stdout handler for INFO messages (simple format)
    * a stderr handler for all other levels at ``stderr_level`` (format
      includes the function name)
    * an optional ``RotatingFileHandler`` at ``{log_dir}/{app_name}.log``
      logging all levels at DEBUG when ``log_dir`` is provided

    The root logger is set to INFO.  The Klea logger namespaces (see
    ``KLEA_LOG_NAMESPACES``) and the application logger (``app_name``) are
    raised to DEBUG so our own logs are captured in full.  Because module
    loggers propagate to the root logger by default, a single call from each
    application entry point routes all Klea logs (library modules, graph
    nodes, API routers) through the same console and file handlers.
    Third-party libraries inherit the root's INFO level, so their DEBUG
    output is filtered at the source without enumerating them.

    :param app_name: Application name, used as the log file name to keep
        per-app logs separate (e.g. ``"klea-rag"``).
    :param stderr_level: Level for the stderr handler (default ``DEBUG``)
    :param log_dir: Directory for the log file.  ``None`` disables file
        logging.
    :returns: The configured root logger
    """
    root = logging.getLogger()
    if root.handlers:
        return root

    root.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(LoggerInfoFilter())
    stdout_handler.setFormatter(logger_formatter_info)
    root.addHandler(stdout_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(stderr_level)
    stderr_handler.addFilter(LoggerNotInfoFilter())
    stderr_handler.setFormatter(logger_formatter_other)
    root.addHandler(stderr_handler)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path / f"{app_name}.log",
            maxBytes=1_000_000,
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logger_formatter_file)
        root.addHandler(file_handler)

    # Turn up verbosity for our own namespaces only.  Third-party libraries
    # inherit root's INFO level, so their DEBUG is filtered without listing
    # them.  Only applied when the logger's level is still NOTSET so an
    # explicit configuration always wins.
    for name in (*KLEA_LOG_NAMESPACES, app_name):
        klea_logger = logging.getLogger(name)
        if klea_logger.level == logging.NOTSET:
            klea_logger.setLevel(logging.DEBUG)

    return root


def mask_sensitive(
    data: dict[str, Any],
    sensitive_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Return a copy with sensitive values masked for logging.

    Shows only the last 4 characters of each value to prevent secrets
    (API keys, tokens) from appearing in plaintext in log output.

    :param data: The dict to sanitize.
    :param sensitive_keys: Keys whose values should be masked.
        Defaults to ``{"api_key"}``.
    :returns: New dict with masked values.
    """
    safe = dict(data)
    for key in sensitive_keys or {"api_key"}:
        if safe.get(key):
            val = str(safe[key])
            safe[key] = f"...{val[-4:]}"
    return safe
