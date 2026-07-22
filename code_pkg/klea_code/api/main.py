#!/usr/bin/env python3
"""
Main API script

File: code_pkg/klea_code/api/main.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.api.app import make_app
from klea_utils.api.chat import create_chat_router
from klea_utils.api.health import create_health_router
from klea_utils.api.models import create_models_router

from klea_code.klea_code import KleaCode


def _create_kleacode() -> KleaCode:
    return KleaCode(checkpoint="inmemory")


app = make_app(
    graph_factory=_create_kleacode,
    title="Klea Code API",
    version="0.0.1",
    routers=[create_chat_router(), create_health_router(), create_models_router()],
)
