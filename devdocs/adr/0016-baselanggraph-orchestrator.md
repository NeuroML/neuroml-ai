---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# BaseLangGraph as single model/MCP/vector-store orchestrator (Template Method pattern)

## Context and Problem Statement

Klea has two orchestration targets that share the same lifecycle:
``klea_rag`` (domain-configurable RAG pipeline) and ``klea_agent``
(general-purpose coding agent), both LangGraph ``StateGraph`` instances
served as FastAPI apps.  Without a common orchestrator each app would
duplicate: env-file -> ``BaseSettings`` -> ``AppConfig`` loading,
``LLMModel`` role setup, ``MCPConfig``/``Client`` building, vector-store
and BM25 loading, SQLite/in-memory checkpoint selection, and the
``setup`` -> ``compile`` -> ``run/a/stream`` seam, plus the per-domain
``tools_info``/``domain_mcp_configs`` mapping and the
``query_domains`` schema generation.  This is the Template Method
(GoF) pattern: the base defines the skeleton of ``setup()`` and defers
``_setup_models`` / ``_configure_resources`` / ``_create_graph`` to
subclasses.

## Decision Drivers

* DRY across ``klea_rag`` and ``klea_agent`` (both currently subclass
  the same base; future packages should do the same).
* Env schema must be generated from ``llm_models`` roles (``{role}_model``
  -> ``KLEA_*_<ROLE>_MODEL``) as a single source of truth, so a missing
  role is visible in the env file documentation.
* Cross-package ``ty`` resolution must see all four ``*_pkg/``
  imports via ``ty.toml`` ``extra-paths``; the base lives in the shared
  ``klea_utils`` lib.
* Template order must be fixed (MCP must list tools before
  ``tools_info`` is built; vector stores must load after the embedding
  model is known) but remain subclass-customisable per phase.

## Considered Options

* **A. Per-app duplication** -- each app copies the ``setup`` logic from
  an early ``code_ai`` package.  Rejected: the ``llm_models`` Single
  Source Of Truth (SSOT) refactor (``2325c9d``) and the ``httpx`` lifespan
  consolidation showed duplication churn across two graphs.
* **B. Composition helpers (Strategy-like free functions)** -- expose ``load_env``,
  ``create_mcp_client``, ``load_vector_stores`` as standalone helpers
  that each app calls (composition / Strategy).  Rejected: call order
  would be re-established per app and divergence would be silent; no
  single place to document the lifecycle or to extend (e.g. ``_setup_checkpointer``,
  ``_pre_graph``).
* **C. Abstract ``BaseLangGraph`` Template Method (chosen)** -- ``utils_pkg/
  klea_utils/graph/base.py:70`` ``BaseLangGraph`` owns the lifecycle as a
  ``@final async def setup()`` that calls ``_pre_setup``,
  ``_setup_checkpointer``, ``_setup_models`` (abstract), ``_load_env``,
  ``_apply_model_names``, ``_apply_provider_defaults``,
  ``_configure_resources`` (abstract), ``_check_required_models``,
  ``_create_mcp_client``, ``_get_mcp_tools``/``_build_tools_info``,
  ``_get_vector_stores`` (generates ``QueryDomainSchema`` from
  ``RetrieverConfig.domains``), ``_pre_graph``, ``_create_graph``
  (abstract), ``_post_setup``.  Apps subclass and implement three
  abstracts: ``_setup_models``, ``_configure_resources``, and
  ``_create_graph`` (plus ``config_class``/``env_var``/``env_prefix``/
  ``graph_name`` class vars).  The base also owns ``_bundled_server_config``,
  ``model_overrides_ctx``, ``_CustomChannelEnabler`` streaming, and the
  ``run_graph_*`` ``RunnableConfig`` seam.

## Decision Outcome

Chosen option: "C. ``BaseLangGraph`` Template Method in ``klea_utils`` (GoF Template Method: subclasses override hooks, base controls lifecycle via ``@final setup()``)".

* Location: ``utils_pkg/klea_utils/graph/base.py:70`` ``BaseLangGraph``;
  ``klea_agent/klea_agent.py:17`` ``KleaAgent(BaseLangGraph)`` and
  ``rag_pkg/klea_rag/rag.py:43`` ``RAG(BaseLangGraph)`` implement the
  abstracts.  ``BaseLangGraph`` is also re-exported via
  ``klea_utils.graph.base`` (``AGENTS.md:139``).
* Model Single Source Of Truth (SSOT): ``llm_models: dict[str, LLMModel]`` is populated in
  ``_setup_models``; ``_build_env_class`` derives the
  ``BaseSettings`` env schema from its keys, so the env file is
  authoritative.  ``_apply_model_names`` + ``_apply_provider_defaults``
  materialise ``model_name``/``provider_defaults`` after ``_load_env``.
* ``_create_mcp_client`` -> ``Client(MCPConfig)`` plus
  ``_get_mcp_tools``/``_build_tools_info`` that prefix tools by server
  name (``server_tool``) and build ``tools_info: dict[str, dict[str,
  ToolInfo]]`` per domain; ``domain_mcp_configs: dict[str, MCPConfig]``
  is also populated for per-domain ``ToolsPicker`` description.
* ``_get_vector_stores`` builds ``VSRetriever`` (needs the embedding
  ``LLMModel``) and ``BM25RetrieverManager``; it also creates
  ``QueryDomainSchema = create_model("QueryDomainSchema", query_domains=
  list[Literal[tuple(all_domains)]])`` for ``ClassifyQuestion``.  The
  embedding model's ``required`` flag is adjusted after
  ``_configure_resources`` (vector-only check via ``_has_vector_stores``).
* Streaming: ``run_graph_astream_events`` / ``graph_stream`` via
  ``_CustomChannelEnabler`` is the single seam consumed by
  ``klea_utils/api/sse.py`` (``klea-rag-serve`` SSE) and direct clients;
  see ADR-0013.

### Consequences

* Good, because new packages (e.g. a future ``code_pkg`` regression) can
  add an orchestrator by subclassing three methods and inheriting env,
  MCP, vector-store, and streaming lifecycles -- no copy-paste.
* Good, because cross-package changes (e.g. ``httpx`` lifespan in
  ADR-0005, bundled stdio in ADR-0004) land once in the base and are
  visible to both graphs.
* Good, because ``ty`` and ``ruff`` have a single import root
  (``ty.toml`` extra-paths) rather than per-app path tricks.
* Bad (inherent to pattern): inheritance couples
  ``klea_rag``/``klea_agent`` to ``klea_utils.graph.base`` -- a base
  change touches every app.  The three abstracts keep the seam
  explicit, but the template order is fixed and ``@final setup()``
  cannot be overridden.  One cannot have Template Method without this
  rigidity; hooks (``_pre_setup``, ``_pre_graph``, subclass
  abstracts) are the intended extension points.
* Bad, because abstract-method contracts are not enforced at import
  time beyond ``abstractmethod``; a missing ``_setup_models`` fails
  only on ``setup()``.

### Confirmation

* ``ty --extra-paths`` for all four packages via ``ty.toml``; ``ruff``
  clean for ``graph/base.py``; ``docs: make html`` still renders the
  pipeline figures that reference ``BaseLangGraph``.
* ``mcp_pkg: pytest -v`` and ``utils_pkg: pytest -m "not localonly"``
  still exercise the shared ``BaseLangGraph`` path (probe
  ``model_overrides_ctx``, ``tools_info``, ``RetrieverConfig``).
* Manual: ``RAG().setup`` → ``M -> VS -> compile`` order logged as
  ``Resolving model configuration`` + ``Vector stores loaded``.

## Pros and Cons of the Options

### BaseLangGraph template method (chosen)

* Good, because DRY lifecycle with single override point per phase
* Good, because env schema stays single source of truth via ``llm_models``
* Bad (inherent to pattern): template order is fixed via ``@final`` --
  one cannot have Template Method without this rigidity; hooks are the
  extension points

### Per-app duplication

* Good, because minimal abstraction
* Bad, because lifecycle drifts per app

## More Information

* Code: ``utils_pkg/klea_utils/graph/base.py:70`` (``BaseLangGraph``),
  ``llm.py:808`` (``create_configurable_model`` used by the base),
  ``rag_pkg/klea_rag/rag.py:43`` / ``klea_agent/klea_agent.py:17``
  (subclasses), ``AGENTS.md:139`` ``BaseLangGraph`` pointer,
  ``devdocs/system/c4-container.md:119`` (every app subclasses
  ``BaseLangGraph`` and is served as FastAPI).
* Related: ``ADR-0004`` (bundled stdio wired via the base),
  ``ADR-0005`` (httpx lifespan composed via the base), ``ADR-0006``
  (monorepo layout that makes the shared base possible), ``ADR-0013``
  (inspection stream owned by the base).
* Commits: ``2325c9d`` (``llm_models`` Single Source Of Truth (SSOT)), ``46c46df``/``96db080``
  (bundled wiring through base), ``b73c3a5``/``65139e3`` (model
  checking + docs flow).
* Codified ``2026-08-28``; base extracted early ``2026-03..04`` during
  ``rag_pkg``/``code_ai`` split and hardened as the template method in
  ``2026-08-17``.
