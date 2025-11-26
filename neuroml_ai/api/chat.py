#!/usr/bin/env python3
"""
Chat end points

File: neuroml_ai/api/chat.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


from fastapi import APIRouter, Request

router = APIRouter()

@router.post("/query")
async def query(request: Request, query: str):

    rag = request.app.state.rag
    return {"result": await rag.run_graph_invoke(query)}
