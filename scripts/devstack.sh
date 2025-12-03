#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
# File:  scripts/devstack.sh
#
# Script for testing the stack


export NML_AI_CHAT_MODEL="huggingface:Qwen/Qwen3-30B-A3B-Instruct-2507:cheapest"
export NML_AI_EMBEDDING_MODEL="huggingface:BAAI/bge-m3:cheapest"

uv pip install -e .[dev]

echo "Re-starting MCP server"
pgrep -fa nml-mcp && pkill -f --signal SIGINT nml-mcp || echo "No running NeuroML MCP instance found"
nml-mcp &

echo "Starting fastapi"
fastapi dev neuroml_ai/api/main.py --port 8005 &

echo "Starting streamlit frontend"
streamlit run neuroml_ai/streamlit_ui.py
