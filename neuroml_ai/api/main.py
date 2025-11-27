#!/usr/bin/env python3
"""
Main API script

File: neuroml_ai/api/main.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


from fastmcp import Client
from fastapi import FastAPI
from neuroml_ai.rag.rag import NML_RAG
from neuroml_ai.api.chat import router
# from neuroml_ai.mcp.util import create_mcp_client

app = FastAPI()

@app.on_event("startup")
async def startup():

    client_url = "http://127.0.0.1:8000/mcp"
    mcp_client = Client(client_url)

    # check that client is up
    async with mcp_client:
        await mcp_client.ping()
        tools = await mcp_client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

    nml_rag = NML_RAG(mcp_client)
    await nml_rag.setup()

    app.state.rag = nml_rag
    app.state.mcp = Client

app.include_router(router)
