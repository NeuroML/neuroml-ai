#!/usr/bin/env python3
"""
Main runner interface for nml-ai

File: neuroml_ai/cli.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import typer
import logging
from neuroml_ai.rag import NML_RAG


nml_ai_app = typer.Typer()


@nml_ai_app.command()
def nml_ai_cli(query: str = typer.Option(None, help="Query for command line mode"), chat_model: str = "ollama:qwen3:1.7b",
               embedding_model: str = "bge-m3", gui: bool = False):
    """NeuroML AI cli wrapper function"""

    if not gui and not query:
        typer.echo("Error: either query or gui must be given")
        raise typer.Exit(code=1)

    if not gui:
        nml_ai = NML_RAG(
            chat_model=chat_model,
            embedding_model=embedding_model,
            logging_level=logging.INFO
        )
        nml_ai.setup()
        nml_ai.run_graph(query)
    else:
        # streamlit app
        pass


if __name__ == "__main__":
    nml_ai_app()
