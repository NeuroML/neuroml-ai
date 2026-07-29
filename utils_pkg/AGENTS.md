# AGENTS.md - Utils Package

Shared utilities for Klea packages.

## Package Overview

Package: `klea_utils`

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
pytest tests/test_utils.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Package Structure
```
klea_utils/
├── api.py          # Legacy API utilities (validate_url, check_api_is_ready)
├── api/            # FastAPI app factory and endpoint routers
│   ├── app.py      # make_app() -- FastAPI factory with lifespan (graph + session store)
│   ├── chat.py     # /query/stream SSE endpoint for streaming graph execution
│   ├── health.py   # /health endpoint for readiness probes
│   ├── messages.py # message history CRUD per chat session
│   ├── models.py   # per-session runtime model switching endpoints
│   ├── server.py   # make_serve_app() -- Typer app wrapping uvicorn
│   ├── sessions_db.py # SQLite session store for chat persistence
│   ├── sessions.py # chat session CRUD (list, rename, delete)
│   ├── sse.py      # SSE streaming client (async gen for NiceGUI/TUI, sync for Streamlit)
│   └── utils.py    # URL validation, API readiness check
├── cli/            # Shared CLI infrastructure
│   └── parser.py   # make_parser() -- standard argparse for all frontends
├── errors.py       # Custom exception classes
├── graph/          # LangGraph orchestrator base
│   └── base.py     # BaseLangGraph abstract class (setup, run, compile template)
├── llm.py          # LLM utilities: configurable models, provider introspection,
│                   #   model name parsing, three-layer config merge, HuggingFace
│                   #   auto-derivation, structured output fallback
├── nodes/          # Shared LangGraph node classes
│   ├── abstract.py # AbstractLangGraphNode, NodeStreamData (streaming contract)
│   ├── answer_general.py # General-purpose answer node (no retrieval needed)
│   ├── base.py     # LangGraphNode, LLMModel, _ConfigurableModel
│   ├── fixed_answer.py   # Node that returns a fixed/canned response
│   ├── guard.py    # GuardNode -- content safety check (skippable when model empty)
│   ├── guard_router.py   # GuardRouterNode -- routes based on guard_decision
│   └── summarise_memory.py # Memory summarisation node
│   └── prompts/    # Prompt markdown templates per node
├── paths.py        # platformdirs wrapper for OS-appropriate cache/data/config dirs
├── plogging.py     # Logging setup, mask_sensitive (API key masking), logfmt output
├── stores/         # Vector store management
│   ├── config.py   # Pydantic models for store configuration
│   ├── ingestion.py # Document ingestion pipeline (chunking, embedding, storage)
│   ├── retrieval.py # Retrieval from configured backends
│   └── utils.py    # Shared store helpers
├── tools.py        # MCP CallToolResult helpers (textualize content blocks)
├── ui/             # User interface frontends
│   ├── tui/        # Textual/TUI chat client (repl.py)
│   ├── vs_create.py # CLI for vector store creation (klea-vs-create)
│   └── web/        # Web frontends
│       ├── nicegui/ # NiceGUI web UI (3-column layout, inspector, model config)
│       └── streamlit/ # Streamlit web UI
```

### Key Technologies
- FastAPI for REST/SSE API layer
- LangGraph for state machine orchestration
- LangChain-core for LLM abstractions (init_chat_model, configurable models)
- NiceGUI for reactive web UI (optional frontend)
- Streamlit for lightweight web UI (alternative frontend)
- HuggingFace for embeddings and inference endpoints
- Chroma / PGVector / Qdrant for vector store backends (URI-style config)
- httpx for async HTTP and SSE streaming
- tenacity for retry logic

## Code Style

### File Organization
- **Header**: All Python files should have a copyright header
- **Docstrings**: Use reStructuredText format
- **Module structure**: `__init__.py` files should be minimal or empty

### Import Conventions
```python
# 1. Standard library imports
import logging
from typing import Any, Dict, List

# 2. Third-party imports
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
import httpx

# 3. Local imports
from klea_utils.plogging import setup_logging
from klea_utils.llm import get_default_model
```

### Naming Conventions
- **Functions**: snake_case (`setup_logging`, `get_default_model`)
- **Classes**: PascalCase (`NeuroMLLogger`, `LLMConfig`)
- **Variables**: snake_case (`model_name`, `api_key`)
- **Constants**: UPPER_CASE (`DEFAULT_MODEL`, `LOG_FORMAT`)

### Guidelines
- Utilities should be framework-agnostic where possible
- Provide sensible defaults but allow configuration
- Document environment variable dependencies
- Include type hints for all public functions
