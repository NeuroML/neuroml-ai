#!/usr/bin/env python3
"""
Shared FastAPI app factory for Klea packages.

File: klea_utils/api/app.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from platformdirs import PlatformDirs

from klea_utils.api.session_store import SessionStore
from klea_utils.graph.base import BaseLangGraph
from klea_utils.paths import get_data_dir, init_dir


def make_app(
    graph_factory: Callable[[], BaseLangGraph],
    title: str = "Klea API",
    version: str = "0.1.0",
    routers: list[APIRouter] | None = None,
) -> FastAPI:
    """Create a FastAPI instance with a standard lifespan.

    The lifespan:

    1. Opens a persistent :class:`SessionStore` at
       ``{data_dir}/sessions.db`` for chat metadata, model overrides,
       and message history.
    2. Instantiates and sets up the graph via *graph_factory*
    3. Stores the graph and session store on ``app.state``

    :param graph_factory: Callable that returns a configured
        :class:`~klea_utils.graph.base.BaseLangGraph` instance
    :param title: API title (appears in OpenAPI docs)
    :param version: API version (appears in OpenAPI docs)
    :param routers: List of APIRouters to include on the app
    :returns: Configured FastAPI app
    """

    # TODO: skipping sqlite bits for testing with an arg/flag

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.is_ready = False

        db_path = init_dir(get_data_dir(PlatformDirs("klea"))) / "sessions.db"
        app.state.chat_sessions = SessionStore(str(db_path))

        graph = graph_factory()
        await graph.setup()
        app.state.graph = graph
        app.state.is_ready = True

        yield

        app.state.is_ready = False
        app.state.chat_sessions.close()

    app = FastAPI(lifespan=lifespan, title=title, version=version)

    for router in routers or []:
        app.include_router(router)

    return app
