# Changelog

## v0.4.0 (wip)

### Streaming infrastructure

- Added `BaseLangGraph.run_graph_astream_events()` — LangGraph v3 protocol
  yielding structured `progress`, `token`, and `complete` events from
  both LLM (``messages`` channel) and non-LLM (``custom`` channel) nodes
- Per-node timing logged server-side for debugging performance
- SSE streaming endpoint `POST /query/stream` on the shared FastAPI router
- Both frontends (CLI and Streamlit) now consume `/query/stream` with
  per-node progress labels

### API refactor

- Consolidated all API code into `klea_utils`: `create_chat_router()`,
  `create_health_router()`, `make_app()`, `make_serve_app()`
- Per-package wrappers (rag/code) are now ~15 lines each
- Added missing `klea-code-serve` CLI to code_pkg

### UI refactor

- Consolidated Streamlit and TUI code into `klea_utils/ui/web/streamlit/`
  and `klea_utils/ui/tui/` — shared `run_streamlit_app()` and `run_repl()`
- No per-package `streamlit_ui.py` needed — resolved via `importlib`
- Node progress shown as compact `st.caption()` in Streamlit, `yaspin`
  spinner with node labels in CLI

### Bug fixes

- `run_graph_stream`: sync ``for`` on async generator → ``async for``
- `run_graph_astream_events`: handle Pydantic ``BaseModel`` (not just dict)
  in the ``values`` channel when extracting final ``message_for_user``
- `AnswerGeneral._update_state`, `GenerateRetrievalQuery._update_state`:
  handle list-form content blocks from newer langchain-ollama
- `ToolsCaller.execute`: only emit custom stream event when node
  actually executes (not when pre-exec check fails)

### Dependencies

- `fastapi[standard]`, `typer`, `cachetools`, `uvicorn` moved from
  rag/code into klea_utils ``install_requires``
- `langchain-huggingface` moved to new ``[huggingface]`` extra
- New ``[ollama]`` extra (langchain-ollama + ollama)
- Chainable extras in rag/code (``rag[huggingface]`` pulls
  ``klea_utils[huggingface]``)
- Heavy imports in `llm.py` made lazy (init_chat_model,
  HuggingFace, JsonOutputParser)

---

## v0.3.0 (2026-07-02)

- Consolidated `label` attribute on all nodes (~25 concrete classes)
  — labels double as LangGraph node names for UI progress display
- Added `write_custom_stream()` helper to `AbstractLangGraphNode`
- Lazy imports in CLI modules for fast `--help`
- Improved error handling and logging throughout

---

## v0.2.0 (development, unreleased)

- Refactored vector stores into sub-package with Chroma, PGVector,
  Qdrant backends
- Added document ingestion pipeline (VSBuilder, chunking, metadata)
- New `klea-vs-create` CLI for building vector stores
- MCP client integration with FastMCP for tool execution
- Per-domain vector store and MCP server configuration
- Evaluator node with re-ranking and query modification loop
- Structured output schemas for LLM nodes
- Memory support (context summary, conversation history)
- Session checkpointing via InMemorySaver

---

## v0.1.0 (2026-06-30)

Initial release of klea_utils and klea_rag to PyPI.

- BaseLangGraph orchestrator with setup/run template methods
- RAG graph: guard, classify, retrieve, generate, evaluate pipeline
- Support for ollama and HuggingFace inference providers
- Chroma, PGVector, Qdrant vector store backends
- CLI client and Streamlit web interface
- Health check endpoints
- Sphinx documentation deployed at https://neuroklea.org
