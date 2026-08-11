# Changelog

## v0.4.0 (wip)

### Automatic metadata extraction

- New `klea_utils.biblio` package: tiered bibliographic metadata extraction
  that pre-fills the per-file `DEFAULT` entries of
  `metadata-map.template.json` during chunking
- Tiers, most authoritative first: DOI resolution via Crossref/OpenAlex/
  Semantic Scholar (round-robin across calls, fallback on rate limits,
  disk-cached), PDF Info dict via pypdfium2, Docling structured signals,
  layout-region regex, document front-matter regex
- Internal `_metadata_complete` / `_sources` keys flag whether automation
  fully populated the metadata and which tiers contributed; they are never
  shown to the answer LLM
- `KLEA_INGEST_MAILTO` env var opts into the DOI APIs' polite pool for
  higher rate limits
- PDF OCR can be disabled with `klea-stores-create --no-ocr`, speeding up
  conversion of text-based PDFs significantly

### Configurable model system

- Replaced `setup_llm()` with `create_configurable_model()` using LangChain's
  `init_chat_model(model=None, configurable_fields="any")` -- model and provider
  specified per-invoke via config dict, no stale config leakage across switches
- `LLMModel.build_config` three-layer merge: `role_defaults` -> context overrides
  -> `node_defaults` (frozen), with model string parsed into LangChain-compatible
  components
- Dynamic provider field filtering via `get_provider_allowed_fields()` --
  introspects provider Pydantic model at runtime, replaces hardcoded
  `PROVIDER_CONFIG_FIELDS`/`EXCLUDES` dicts
- `model_defaults` class attribute on all 15+ node subclasses, replacing
  per-constructor `temperature` param
- Structured output fallback moved into `_invoke_llm` with try/except for
  providers that reject `response_format`
- `mask_sensitive` in plogging -- reusable helper to mask API keys in logs
- HuggingFace auto-derivation from model string suffix (`local` -> `backend`,
  remote -> `provider`) survives provider field filtering
- `LLMModel` moved from `graph/base.py` to `llm.py` alongside `parse_model_name`,
  `create_configurable_model`, `get_provider_allowed_fields`

### Optional guard node

- When `guard_model` is empty in config, `GuardNode._pre_exec` returns `False`
  (skip the guard entirely)
- `guard_decision` default changed from `"unsafe"` to `"safe"` in RAG state/init
  node so the pipeline proceeds when guard is disabled

### NiceGUI web UI

- New 3-column layout: left drawer (chat list, inspector, new chat), center
  (messages), right drawer (status pane with per-node progress)
- Model configuration dialog for runtime model switching
- Inspector dialog with debug entries (heading, summary, timing, expandable JSON)
  -- snapshot-based, disabled during streaming, enabled on complete/error
- Chat bubbles render markdown (`ui.markdown` instead of `ui.html`)
- Status pane with per-node state sections, collapsible details
- Auto-generated chat names via `coolname`

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

### Hybrid BM25 keyword retrieval

- New `BM25RetrieverManager` (``klea_utils/stores/retrieval/bm25.py``)
  providing classic keyword search over a pickled chunk corpus
- New shared `BaseKleaRetriever` base class for retriever managers
  (per-store ``k`` tracking, lazy loading); `VSRetriever` refactored
  onto it; retrieval split into a subpackage
  (``retrieval/{base,vs,bm25}.py``)
- Domains can now configure `bm25_stores` in addition to `vector_stores`
  (either/both/neither)
- `klea-stores-create build|store --bm25-store <path>` writes the
  combined chunked corpus for BM25 retrieval (replaces `VSBuilder` /
  `klea-vs-create` naming)
- Retrieval fuses vector-store and BM25 results with Reciprocal Rank
  Fusion; original per-source scores preserved in `_source_scores`
  metadata and shown to the answer LLM (`serialize_vs_retrieval`)
- New `rrf_merge` / `format_source_scores` helpers in
  `klea_utils/stores/utils.py`
- New deps: `langchain-community`, `rank_bm25`

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
