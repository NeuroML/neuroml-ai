#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
# File:  docker/deploy.sh
#
# Script for docker deployments


echo "Re-starting MCP server"
nml-mcp &

echo "Starting fastapi"
fastapi dev neuroml_ai/api/main.py --port 8005 &

echo "Starting streamlit frontend"
streamlit run neuroml_ai/streamlit_ui.py
