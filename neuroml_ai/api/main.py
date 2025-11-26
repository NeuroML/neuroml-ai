#!/usr/bin/env python3
"""
Main API script

File: neuroml_ai/api/main.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


from neuroml_ai.mcp.server import codegen
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from neuroml_ai.rag.rag import NML_RAG
from neuroml_ai.api.chat import router
# from neuroml_ai.mcp.util import create_mcp_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    codegen_path=Path(codegen.__file__).absolute()
    print(f"{codegen_path =}")
    server_parameters = StdioServerParameters(
        command="python",
        args=[str(codegen_path)]
    )


    read, write = await stdio_client(server_parameters).__aenter__()
    mcp_client = await ClientSession(read, write).__aenter__()
    await mcp_client.initialize()
    tools = await mcp_client.list_tools()
    print(f"Available tools: {[tool.name for tool in tools.tools]}")

    nml_rag = NML_RAG(mcp_client)
    await nml_rag.setup()

    app.state.rag = nml_rag
    app.state.mcp_client = mcp_client
    app.state.mcp_read = read
    app.state.mcp_write = write

    try:
        yield
    finally:
        await app.state.mcp_client.__aexit__(None, None, None)
        await app.state.mcp_read.__aexit__(None, None, None)
        await app.state.mcp_write.__aexit__(None, None, None)

app = FastAPI(lifespan=lifespan)
app.include_router(router)
