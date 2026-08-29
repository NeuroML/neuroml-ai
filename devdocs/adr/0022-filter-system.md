---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Filter system: declarative fields, per-domain scoping, and dialect translation

## Context and Problem Statement

Klea RAG's retrieval must support structured filtering (journal,
year, domain-specific fields like ``topic`` or ``tags``) in addition
to semantic similarity.  Early RAG had hard-coded filter fields in
``RetrievalQueryOutput`` and the LLM prompt, so adding a new field
required code changes, and a single global filter leaked across
domains (``PapersA``'s ``journal: Nature`` also constrained
``PapersB``).  Each vector store backend (Chroma, Qdrant, pgvector)
has a different filter syntax, so the LLM's logical filter must be
translated per backend; BM25 has its own tuple-to-field path.

How should filters be declared, generated, scoped, and translated
uniformly?

## Decision Drivers

* No code change to add a filter field: declaration in config file
  should suffice.
* Per-domain scoping: a filter on ``topic`` for ``DomainA`` must not
  contaminate ``DomainB`` whose store does not have that field.
* Per-backend translation: one logical filter (``field == value`` with
  ``value_type``) must be rendered into Chroma ``where``, Qdrant
  ``Filter``, pgvector JSONB, and BM25 ``tuple->filter_fields``.
* Person-name recall: ``authors``/``persons`` queries must match
  partial names (``Sinha`` matches ``Ankur Sinha``).

## Considered Options

* **A. Hard-coded filter fields in code** -- ``RetrievalQueryOutput``
  has ``journal``, ``year``, ``authors``, etc. as explicit Pydantic
  fields; the prompt template lists them.  Rejected: adding ``topic``
  requires a code change and redeploy; global filter object has no
  domain concept so cross-domain leak is inherent.
* **B. Single global declared filter list, no per-domain scoping** --
  unify ``filter_fields`` at ``GeneralConfig`` level and forward the
  whole logical filter to every retriever.  Rejected: a ``topic``
  filter meant for ``DomainA`` would also constrain ``DomainB`` whose
  store does not have ``topic``, producing false ``0`` results with
  native filter push-down.
* **C. Declarative per-domain ``filter_fields`` + ``filter_fields``-
  aware scoping + translator matrix (chosen)** -- ``RetrieverConfig``
  declares ``PerDomainConfig.filter_fields`` as ``list[FilterFieldInfo]
  `` (``name``, ``description``, ``value_type`` ``string``/``int``/
  ``float``/``list``).  ``klea_utils/stores/filters.py`` provides typed
  helpers: ``sanitize``/``normalize`` per ``value_type``, ``expand_person_names``
  store-time person-name expansion (full + per-word lowercase variants,
  applied in ``_apply_store_metadata_policy``), ``normalize_config_filters``
  (logical -> ``FilterClause`` list), and ``restrict_metadata_filter``
  / ``translate`` per-backend (Chroma/Qdrant/pgvector dialect + BM25
  tuple path).  At generation time ``GenerateRetrievalQuery`` receives
  ``filter_fields_by_domain`` and the prompt lists *only* the declared
  fields for that domain; at retrieval time
  ``RetrieveInfoNode`` scopes the combined logical filter per domain
  (``restrict_metadata_filter``) so each domain only sees the clauses
  on its declared fields, dropping the rest.  Documented in
  ``docs/concepts/rag.rst`` and validated by ``STORES_TEST_CONFIG``.

## Decision Outcome

Chosen option: "C. Declarative per-domain filter system".

* Config: ``utils_pkg/klea_utils/stores/config.py:58``
  ``FilterFieldInfo`` (``name``, ``description``, ``value_type``) and
  ``PerDomainConfig.filter_fields``; empty list means unfiltered
  retrieval (``tuple->filter_fields`` inherited from ``retrieval`` helpers
  enforces ``tuple`` scoping for BM25).  ``RetrievalQueryOutput`` now has
  ``search_query`` + ``filters`` (raw LLM operand dict) + ``config_filters``
  (canonical ``FilterClause``) rather than typed hard-coded fields.
* Generation: ``rag_pkg/klea_rag/nodes/generate_retrieval_query.py``
  receives ``filter_fields_by_domain: dict[str, list[FilterFieldInfo]]``
  (built in ``rag.py:242`` ``_configure_resources``) and renders only
  the declared fields per domain in the LLM prompt; the LLM emits
  logical filters in operand form (``$in``, ``$gte``, ``$contains``,
  etc.) keyed by field name.
* Scoping: ``klea_utils/stores/filters.py`` ``restrict_metadata_filter``
  takes the combined ``config_filters`` and drops any clause whose
  field is not declared for that domain; each domain therefore
  queries only its relevant subset.
* Translation: ``translate_chroma_filter`` / ``translate_qdrant_filter`` /
  ``translate_pgvector_filter`` + BM25 ``tuple`` plumbing render the
  scoped clauses into the native dialect; ``value_type`` drives
  rendering (``string``/``int``/``float`` → ``$in`` / range vs ``list``
  → ``$contains`` all).  Person-name expansion is store-time, not
  query-time, so ``Sinha`` matches via the stored variants.
* Storage: ``utils_pkg/klea_utils/stores/utils.py`` ``normalize_text``
  (NFKC + soft-hyphen) + ``sanitize`` + ``expand_person_names`` in
  ``_apply_store_metadata_policy`` so both Chroma metadata and the BM25
  corpus carry the variants.

### Consequences

* Good, because adding a filter field is a one-line config edit
  (``filter_fields: [{name: topic, description: ..., value_type: string}]``)
  plus store regeneration -- no code change, no prompt template change.
* Good, because per-domain scoping prevents cross-domain contamination;
  a ``topic`` filter never constrains a domain that does not declare
  ``topic``.
* Good, because one logical filter fans out to all backends via the
  single ``translate`` matrix; new backends require one translator,
  not a new node.
* Bad, because a declared field only filters effectively when the
  underlying stores actually carry that metadata key; retrieval on a
  store missing the key yields nothing for that clause, so the store
  must be regenerated with the key (breaking re-store).
* Bad, because ``FilterFieldInfo`` + ``value_type`` discipline adds a
  small config authoring burden vs a free-form string.

### Confirmation

* ``STORES_TEST_CONFIG`` (``utils_pkg/tests/test_stores_retrieval.py``)
  exercises ``VSRetriever`` + ``BM25RetrieverManager`` with scoped
  filters; ``restrict_metadata_filter`` unit tests assert per-domain
  subsetting (``k*3`` + native filter ``0`` → fixed).
* ``ty`` ``extra-paths`` for ``RetrieverConfig`` + ``FilterFieldInfo``;
  ``ruff`` clean for ``stores/filters.py``/``config.py``/``generate_retrieval_query``/
  ``retrieve_info``.
* Manual: ``GenerateRetrievalQuery`` with ``filter_fields_by_domain``
  emits only declared fields; ``RetrieveInfoNode`` routes only the
  declared subset per domain (cross-domain ``topic`` leak → ``0`` fixed).

## Pros and Cons of the Options

### Declarative per-domain filter system (chosen)

* Good, because no code change to add a field
* Good, because per-domain scoping avoids contamination
* Good, because one logical filter → N dialects
* Bad, because declared field still needs per-store metadata present

### Hard-coded fields

* Good, because minimal abstraction
* Bad, because every new field requires code change and global filter leaks

## More Information

* Code: ``stores/config.py:58`` (``FilterFieldInfo``/``PerDomainConfig``),
  ``stores/filters.py`` (typed helpers + ``translate`` matrix),
  ``stores/utils.py`` (``normalize_text``/``expand_person_names``),
  ``rag/nodes/generate_retrieval_query.py`` (per-domain prompt),
  ``nodes/retrieve_info.py`` (per-domain ``restrict_metadata_filter`` loop).
* Related: ``ADR-0011`` (multiple domains -- this ADR governs *how*
  filters are scoped *within* those domains); ``docs/concepts/rag.rst``
  174 (filter-fields discipline); ``docs/troubleshooting.rst``
  (store ``name``/``path`` mismatch).
* Commits: ``c9f5d17``/``fc95034``/``310e741``/``a598242`` (filter
  translators), ``c47afbe``/``d0b8957`` (store-create batch + RRF),
  ``7bb3c2e``/``7c892c5``/``d5f8ce3``/``da244ef`` (filter fields).
* Codified ``2026-08-28``; landed ``2026-08-21`` filter-facade work.
