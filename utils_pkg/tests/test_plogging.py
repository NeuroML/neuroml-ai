#!/usr/bin/env python3
"""
Tests for logging setup (plogging).

File: tests/test_plogging.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import tempfile
import unittest
from logging.handlers import RotatingFileHandler

from klea_utils.plogging import (
    KLEA_LOG_NAMESPACES,
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    setup_root_logger,
)


class TestSetupRootLogger(unittest.TestCase):
    """Tests for the process-wide root logger configuration."""

    def setUp(self):
        # The root logger is process-global and setup_root_logger is
        # idempotent, so save/restore root state around every test to avoid
        # leaking handlers (and namespace level overrides) into other tests.
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)
        for name in (
            *KLEA_LOG_NAMESPACES,
            "testapp",
            "urllib3",
            "httpx",
            "mcp",
            "aiosqlite",
        ):
            logging.getLogger(name).setLevel(logging.NOTSET)

    def tearDown(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in self._saved_handlers:
            root.addHandler(handler)
        root.setLevel(self._saved_level)

    def test_console_handlers_added(self):
        """INFO goes to stdout, other levels to stderr, root at INFO."""
        setup_root_logger("testapp")
        root = logging.getLogger()

        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
        ]
        self.assertEqual(len(stream_handlers), 2)

        info_handlers = [
            h
            for h in stream_handlers
            if any(isinstance(f, LoggerInfoFilter) for f in h.filters)
        ]
        other_handlers = [
            h
            for h in stream_handlers
            if any(isinstance(f, LoggerNotInfoFilter) for f in h.filters)
        ]
        self.assertEqual(len(info_handlers), 1)
        self.assertEqual(len(other_handlers), 1)
        self.assertEqual(root.getEffectiveLevel(), logging.INFO)

    def test_file_handler(self):
        """A rotating file handler is created and receives all levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_root_logger("testapp", log_dir=tmpdir)
            root = logging.getLogger()

            file_handlers = [
                h for h in root.handlers if isinstance(h, RotatingFileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            self.assertTrue(file_handlers[0].baseFilename.endswith("testapp.log"))
            self.assertEqual(file_handlers[0].level, logging.DEBUG)

            logger = logging.getLogger("klea_utils.test")
            logger.debug("debug line")
            logger.info("info line")
            logger.warning("warning line")
            for handler in file_handlers:
                handler.flush()

            with open(file_handlers[0].baseFilename) as f:
                content = f.read()
            self.assertIn("debug line", content)
            self.assertIn("info line", content)
            self.assertIn("warning line", content)

    def test_no_file_handler_without_log_dir(self):
        """Without log_dir, only the two console handlers are added."""
        setup_root_logger("testapp")
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
        self.assertEqual(len(file_handlers), 0)

    def test_idempotent(self):
        """A second call returns the root logger without adding handlers."""
        root = logging.getLogger()
        setup_root_logger("app1")
        num_handlers = len(root.handlers)

        result = setup_root_logger("app2", log_dir="/tmp/should-not-exist")
        self.assertIs(result, root)
        self.assertEqual(len(root.handlers), num_handlers)

    def test_klea_namespaces_debug(self):
        """Our own logger namespaces are turned up to DEBUG."""
        setup_root_logger("testapp")
        for name in (*KLEA_LOG_NAMESPACES, "testapp"):
            self.assertEqual(logging.getLogger(name).getEffectiveLevel(), logging.DEBUG)

    def test_third_party_inherits_info(self):
        """Third-party loggers inherit the root INFO level, not DEBUG."""
        setup_root_logger("testapp")
        for name in ("urllib3", "httpx", "mcp", "aiosqlite"):
            self.assertEqual(logging.getLogger(name).getEffectiveLevel(), logging.INFO)

    def test_third_party_explicit_level_not_overridden(self):
        """An explicitly configured third-party level is left alone."""
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        setup_root_logger("testapp")
        self.assertEqual(logging.getLogger("httpx").getEffectiveLevel(), logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
