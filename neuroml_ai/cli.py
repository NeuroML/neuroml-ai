#!/usr/bin/env python3
"""
Main runner interface for nml-ai

File: neuroml_ai/cli.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import typer
import logging
from rag import NML_RAG


nml_ai_app = typer.Typer()


@nml_ai_app.command()
def nml_ai_cli(query: str):
    """NeuroML AI cli wrapper function"""
    nml_ai = NML_RAG(
        chat_model="ollama:qwen3:1.7b",
        embedding_model="bge-m3",
        logging_level=logging.INFO
    )
    nml_ai.setup()
    nml_ai.run_graph(query)


if __name__ == "__main__":
    nml_ai_app()
