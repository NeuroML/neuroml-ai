#!/usr/bin/env python3
"""
Server entry point for the Klea Agent API.

File: klea_agent/api/server.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.api.server import make_serve_app

serve_app = make_serve_app("klea_agent.api.main:app", default_port=8006)
