---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# SQLite checkpointer and session store for graph resumption

## Context and Problem Statement

LangGraph graphs are stateful across turns: the next ``ainvoke`` must
see prior messages, the compiled graph's checkpoints, and per-user/chat
model overrides (``model``, ``api_key``, ``base_url`` per ADR-0014).
Early Klea used LangGraph's ``InMemorySaver`` only, so restarts lost
history and deployments with multiple users could not isolate sessions.
LangGraph's checkpointing and Klea's own chat persistence were also
divergent: the checkpointer lived in the graph while chat history lived
in NiceGUI's ``app.storage.user`` (ephemeral, per-process).

How should checkpoints and chat/session state be persisted, scoped per
user/chat, and reused across graph resumptions?

## Decision Drivers

* Resumption: ``graph.ainvoke`` / ``graph.astream_events`` with
  ``RunnableConfig(configurable={"thread_id": ...})`` must replay prior
  checkpoints.
* Per-user/chat isolation: a public HuggingFace Space and local single-
  user must both isolate ``user_id``/``chat_id`` and per-chat model
  overrides (``KLEA_*_*_MODEL`` per session vs global env).
* Web/TUI persistence: chat history, ``dark_mode``, and the active
  ``chat_id`` must survive server restarts and browser reloads; NiceGUI
  ``.nicegui/storage-user-*.json`` is only a pointer to server-side state.
* Operational simplicity: the chosen store should not require an
  external DB (Redis/Postgres) on ``cpu-basic``.

## Considered Options

* **A. ``InMemorySaver`` only** -- the graph's default checkpointer
  holds checkpoints in process memory; chat history lives in
  ``app.storage.user``.  Rejected: restart loses checkpoints and chat
  history; ``app.storage.user`` is not durable across processes and
  cannot scope ``user_id``/``chat_id`` server-side.
* **B. ``AsyncSqliteSaver`` checkpointer + SQLite session store (chosen)**
  -- ``BaseLangGraph._setup_checkpointer`` selects between
  ``InMemorySaver`` (``checkpoint="inmemory"``), ``AsyncSqliteSaver``
  (``checkpoint="sqlite"`` at ``{user_data_dir}/checkpoints.db`` via
  ``aiosqlite``), or ``None`` (``checkpoint="none"`` → ``memory=False``).
  A dedicated ``klea_utils/api/sessions_db.py`` ``SessionStore``
  (SQLite) owns ``chat_sessions`` (including ``model overrides`` JSON
  blob per chat) and ``messages`` tables, exposed via
  ``klea_utils/api/sessions.py`` ``/sessions`` / ``/chat`` / ``/messages``
  endpoints.  ``user_id`` is excluded from the backend and derived from
  the ``.nicegui/storage-user-*.json`` pointer only.
* **C. External DB (Redis/Postgres) for both** -- Rejected: adds
  operational cost for ``cpu-basic`` demos and institutional
  deployments that already have path-based user-data dirs via
  ``platformdirs``.

## Decision Outcome

Chosen option: "B. ``AsyncSqliteSaver`` checkpointer + SQLite session store".

* ``utils_pkg/klea_utils/graph/base.py:549`` ``_setup_checkpointer``
  branches on ``self.checkpointer_mode`` (from ``BaseLangGraph.__init``
  ``checkpoint``).  ``AsyncSqliteSaver`` opens
  ``{user_data_dir}/checkpoints.db`` (``init_dir`` + ``aiosqlite``);
  ``InMemorySaver`` is the default; ``None`` disables checkpointing.
* ``klea_utils/api/sessions_db.py`` ``SessionStore`` owns the two
  tables; ``api/sessions.py`` serves the CRUD endpoints used by the
  NiceGUI/TUI runners and the ``klea-rag-serve`` SSE path.
* Checkpointing and session memory are orthogonal: ``memory`` flag
  (``checkpoint != "none"``) governs whether nodes receive
  ``memory=True`` (turn-faithful prompts via ``BaseMessage`` objects
  per ADR-0018); the checkpointer governs LangGraph resumption.
* ``BaseLangGraph.setup`` calls ``_setup_checkpointer`` before
  ``_load_env`` so the checkpointer is available even when env
  resolution fails.

### Consequences

* Good, because graph resumption works across restarts (LangGraph
  checkpoints are durable) and chat sessions survive server restarts
  (``sessions.db``) rather than living only in NiceGUI's ephemeral
  ``.nicegui/`` pointer files.
* Good, because per-chat model overrides (``model``, ``api_key`` per
  ``ADR-0014``) are stored per ``chat_id`` in ``chat_sessions`` and
  replayed via ``RunnableConfig(configurable={"thread_id": ...})``.
* Good, because no external DB is required on ``cpu-basic``; the
  store is a local ``.db`` file in the platform user-data dir.
* Bad, because SQLite is single-host: true horizontal scaling would
  need an external checkpointer/session store.
* Bad, because ``checkpoints.db`` and ``sessions.db`` add on-disk
  state that must be pruned externally; there is no automatic
  retention window today.

### Confirmation

* ``klea_utils/api/sessions_db.py`` unit coverage for CRUD
  (``chat_sessions`` including overrides blob, ``messages``) and
  ``graph/base.py:549`` ``AsyncSqliteSaver`` wiring.
* ``docs: make html`` still renders the pipeline figure;
  ``ty`` extra-paths for ``graph/base.py`` + ``api/sessions*``.
* Manual: ``klea-rag-serve`` with ``checkpoint="sqlite"`` writes
  ``checkpoints.db``; browser reload with ``.nicegui/`` pointer reopens
  the same ``chat_id``/model overrides; ``InMemorySaver`` and
  ``checkpoint="none"`` paths still work.

## Pros and Cons of the Options

### AsyncSqliteSaver + SQLite session store (chosen)

* Good, because durable resumption across restarts
* Good, because per-chat model overrides survive restarts
* Good, because no external DB on ``cpu-basic``
* Bad, because single-host SQLite (not horizontally scalable)

### InMemorySaver only

* Good, because zero disk state
* Bad, because restart loses checkpoints and chat history

## More Information

* Code: ``utils_pkg/klea_utils/graph/base.py:549`` (``_setup_checkpointer``),
  ``api/session_store.py`` / ``api/sessions.py`` / ``api/sessions_db.py``
  (session store + endpoints), ``api/app.py`` (FastAPI wiring),
  ``graph/base.py:40`` (``model_overrides_ctx`` per-session overrides
  consumed via ``RunnableConfig``).
* Related: ``ADR-0014`` (runtime model switching stored per chat),
  ``ADR-0018`` (``BaseMessage``-object memory), ``ADR-0013`` (inspection
  stream shows ``usage`` per node).
* Commits: ``d0b6dbe`` (add sqlite checkpointer), ``6fa52e4`` (add db
  CRUD), ``901aaff`` (add session endpoints), ``2772039`` (update chat
  to use session store), ``0255bea``/``a74487b`` (session plumbing).
* Codified ``2026-08-28``; checkpointer/session store landed
  ``2026-07-22..23`` during the session-persistence sprint.
