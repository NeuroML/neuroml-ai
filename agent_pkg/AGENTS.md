# AGENTS.md - Code AI Package

AI assisted coding/workflow system using LangChain/LangGraph.

## Package Overview

Package: `klea_code`
CLI entry: `klea-code`

## Development Commands

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Linting and Formatting
```bash
# Run ruff for linting and fixing
ruff check . --fix
ruff format .

# Sort imports specifically
ruff check . --select I --fix
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_code_ai.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Package Structure
```
klea_code/
├── api/             # FastAPI server (thin wrappers around klea_utils routers)
│   ├── main.py      # FastAPI app creation
│   └── server.py    # Typer serve command
├── config.py        # Configuration loading (env file + JSON)
├── klea_code.py     # Main CodeAgent orchestrator (extends BaseLangGraph)
├── nodes/           # LangGraph nodes for code generation workflows
│   ├── answer_user.py
│   ├── evaluator.py
│   ├── explore_planner.py
│   ├── goal_setter.py
│   ├── init_graph.py
│   ├── planner.py
│   ├── tools_caller.py
│   ├── tools_picker.py
│   └── tools_router.py
│   └── prompts/     # Prompt markdown templates per node
├── schemas.py       # Pydantic schemas
├── tools/           # Bundled MCP tools
│   └── bundled.py
└── ui/
    └── cli.py       # Typer CLI entry point (klea-code, klea-code-serve)
```

### Key Technologies
- LangChain/LangGraph for agent orchestration
- FastMCP for MCP tool integration
- Typer for CLI
- NiceGUI/Streamlit UI shared from klea_utils

## Code Style

### File Organization
- **Header**: All Python files should have a copyright header
- **Docstrings**: Use reStructuredText format
- **Module structure**: `__init__.py` files should be minimal or empty

### Import Conventions
```python
# 1. Standard library imports
import asyncio
import os
from typing import Any, Dict, List

# 2. Third-party imports
from langchain_core.messages import HumanMessage, AIMessage
from fastmcp import Context
from pydantic import BaseModel

# 3. Local imports
from klea_code.schemas import CodeGenRequest
from klea_code.nodes import generate_code_node
```

### Naming Conventions
- **Functions**: snake_case (`generate_code_node`, `validate_neuroml_model`)
- **Classes**: PascalCase (`CodeGenRequest`, `LangGraphAgent`)
- **Variables**: snake_case (`model_name`, `code_output`)
- **Constants**: UPPER_CASE (`DEFAULT_MODEL`, `MAX_RETRIES`)
