#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
# File : 
#


uv pip install -e .[dev]

echo "Re-starting MCP server"
pgrep -fa nml-mcp && pkill -f --signal SIGINT nml-mcp || echo "No running NeuroML MCP instance found"
nml-mcp &


fastapi dev neuroml_ai/api/main.py --port 8005
