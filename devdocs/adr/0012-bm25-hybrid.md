---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# BM25 hybrid retrieval for exact string matches

## Context and Problem Statement

Dense vector similarity generalises meaning but misses the exact string
matches that academic and technical domains rely on.  Neuron names
(e.g. ``Purkinke cell`` variants), ion/channel symbols (``Cav3.1``,
``Nav1.6``, ``HCN``), model identifiers, parameter keys, and Biblio
tokens are often lexically distinct but semantically close -- the
embedding collapses them while the user expects an exact hit.

Should Klea add a lexical complement to vector search, and how should
the two be combined without calibrating incomparable scores?

## Decision Drivers

* Answer quality: exact names and symbols must be surfaced even when
  they are not the nearest semantic neighbour.
* Score incompatibility: vector cosine lies in ``[0, 1]`` (1 = most
  similar); BM25 is an unbounded raw keyword relevance (higher = more
  matching terms, e.g. ``5.1``).  The two are not on the same scale so
  a weighted linear merge requires arbitrary calibration.
* Deduplication: the same chunk may appear in both stores and must be
  merged without double counting.
* Declarative per-domain wiring: a domain may configure
  ``vector_stores``, ``bm25_stores``, both, or neither; the pipeline
  should respect that without forking the graph.
* Recency as tie-breaker: academic work builds on and often corrects
  earlier results, so a newer document should outrank an older one of
  equal relevance.

## Considered Options

* **A. Vector only** -- original system: single ``VSRetriever`` per
  domain.  Rejected: fails on exact symbols (e.g. ``Cav3.1`` vs
  ``Cav3`` collapse, neuron/channel names).
* **B. Weighted score merge** -- normalise both scores to ``[0, 1]`` and
  blend with a learned or hand-tuned weight.  Rejected: calibration is
  arbitrary per corpus; BM25 range depends on corpus length and
  term statistics, vector cosine on embedding model -- no stable
  weighting exists.
* **C. RRF rank fusion with per-source scores as debug context (chosen)**
  -- each domain may declare ``bm25_stores`` alongside
  ``vector_stores``.  At query time ``RetrieveInfoNode`` (via
  ``BaseKleaRetriever``) queries all of the domain's stores; results are
  fused with **Reciprocal Rank Fusion** (``1 / (60 + rank)`` per
  store), deduped by content, then top ``k`` kept.  Raw per-source
  scores are preserved as ``_source_scores`` metadata for inspection
  but never shown to the answer LLM.  A small recency bias
  (``rerank_by_recency: 0.9 * relevance + 0.1 * time`` where
  ``time = (year - year_min)/(year_max - year_min)`` after normalising
  the RRF score to ``[0, 1]``; no year -> ``0.5``) is blended after
  fusion, so newer documents outrank older ones of equal relevance while
  relevance still dominates.
* **D. BM25 only** -- variant of C for BM25-only domains.  Already
  covered: a domain with no ``vector_stores`` and only ``bm25_stores``
  requires no embedding model (``RAG._has_vector_stores`` checks).
  The same RRF path handles it (single store, no fusion needed).

## Decision Outcome

Chosen option: "C. RRF rank fusion with per-source scores as debug context".

* Storage: ``klea-stores-create --bm25-store /path/to/corpus.pkl``
  (or ``store --bm25-store``) writes the combined chunked documents to
  a single pickle corpus (always written; move the file afterwards).
  ``store-lint <corpus.pkl>`` runs deterministic, LLM-free checks
  (summary, suspicious near-empty chunks, structural checks, contiguous
  ``--samples`` windows; printed automatically after ``store`` when a
  BM25 corpus is written).
* Config: ``PerDomainConfig`` now has ``bm25_stores: list[BM25StoreConfig]``
  parallel to ``vector_stores``; each entry's ``path`` points at the
  pickle corpus file; ``name`` must match ``--collection`` just like
  vector stores.  ``filter_fields`` interact with BM25 the same as with
  vector stores (BM25 path has a ``tuple->filter_fields`` plumbing
  noted but no separate filter call needed; filtering still scopes to
  per-domain retrievers via ``restrict_metadata_filter``).
* Retrieval: ``rag_pkg/klea_rag/nodes/retrieve_info.py`` consumes
  ``retrievers: list[BaseKleaRetriever]`` (``VSRetriever`` and/or
  ``BM25RetrieverManager`` as ``BaseKleaRetriever`` sub-types) built in
  ``rag.py:303`` ``_configure_resources``.  Every query runs all of the
  domain's stores and merges via ``rrf_merge`` (``k=60``), dedupes, and
  truncates to the global ``max_refs_size`` char budget (not a doc
  count).  Duplicate chunks are removed; ordering is by RRF (then
  recency blend), never by raw score comparison.
* Ingestion: ``klea_utils/stores/bm25.py`` ``BM25RetrieverManager`` wraps
  ``langchain-community``'s ``rank_bm25`` ``BM25Retriever``; chapter
  heading metadata is preserved as chunk ``metadata`` (the corpus pickle
  carries it).  The component lives under ``stores/retrieval/`` as a
  ``BaseKleaRetriever`` sub-type so ``docs/concepts/rag.rst``
  ``retrieve_info`` plus the ``generate_retrieval_query`` tool path
  need not special-case BM25.
* Graph: the RAG graph still ``always retrieve`` (ADR-0008) -- BM25 and
  vector retrievers are both children of the same ``RetrieveInfoNode``
  that sits behind ``_splitter_label``; no new branching.

### Consequences

* Good, because exact symbols and names (``Cav3.1``, ``Nav1.6``,
  ``Pvalb``, model IDs) that dense retrieval would dilute are surfaced
  by the lexical path; cross-cutting queries (ADR-0011) cite them
  alongside semantic hits in one RRF-ordered list.
* Good, because RRF needs no score calibration (only ranks) and
  preserves raw scores as ``_source_scores`` for debugging without
  confusing the answer LLM.
* Good, because BM25-only domains need no embedding model
  (``_has_vector_stores`` gates ``llm_models["embedding"].required``),
  and mixed domains require no per-corpus weight.
* Good, because the same ingestion CLI flags serve both stores:
  ``--bm25-store`` writes the pickle alongside the vector store in one
  pass; ``store-lint`` gives an offline corpus health check.
* Bad, because BM25 corpus is a single pickle file that holds the
  entire chunk text (no vector store sharding) -- large corpora pay a
  creation-time pickle dump and a load-time unpickle cost.  Mitigated
  by batched corpus writes (per ``embed_batch_size`` in
  ``ingestion.py``) and by keeping the corpus as ``.pkl`` on
  ``deployments/huggingface/`` ``git lfs``/``xet`` rather than in the
  monorepo.
* Bad, because RRF ``k=60`` and the ``0.9/0.1`` recency blend are fixed
  constants; they tune ordering for the current academic corpora but
  are not exposed as per-domain config today.
* Bad, because hybrid retrieval doubles store hits per domain query
  (one vector + one BM25 call) and therefore adds embedding + rank
  latency; the same ``max_refs_size`` budget still caps the final
  context, but fetch cost is genuinely higher.

### Confirmation

* ``utils_pkg/tests/test_stores_retrieval.py`` exercises
  ``VSRetriever``/``BM25RetrieverManager`` + RRF + ``max_refs_size``
  truncation via ``STORES_TEST_CONFIG`` (``stores-tests.json``); the
  per-source ``_source_scores`` is asserted as preserved metadata.
* ``docs/concepts/rag.rst:174`` Hybrid retrieval (vector + BM25) +
  ``docs/tutorials/create-and-use-rag.rst`` (``--bm25-store`` flag)
  document the same ``RRF 1/(60+rank)`` + ``0.9/0.1`` + ``scale`` note;
  ``klea-stores-create --help`` advertises ``--bm25-store`` and
  ``store-lint``.
* ``ty`` ``extra-paths`` still resolves ``RetrieverConfig`` with
  ``bm25_stores`` from ``rag_pkg``; ``docs: make html`` renders the
  hybrid section without score-comparison guidance.
* Live: ``klea-rag-serve`` logs per-retriever ``k`` and the RRF-ordered
  fused set before ``AnswerFromContext``; queries for ``Nav1.6`` and
  similar symbols now cite BM25-ranked hits that vector alone missed.

## Pros and Cons of the Options

### Vector only

* Good, because single retriever, single cost
* Bad, because fails on exact symbols (channel/neuron/model IDs)

### Weighted score merge

* Good, because would let vector/BM25 weights be tuned
* Bad, because scores are incomparable (cosine vs raw BM25); no
  stable weight across corpora

### RRF rank fusion with per-source scores as debug context (chosen)

* Good, because exact symbols surfaced without calibration
* Good, because raw scores preserved as ``_source_scores`` debug context
  but not shown to the answer LLM
* Good, because BM25-only domains need no embedding model; mixed domains
  need no per-corpus weight
* Bad, because whole-corpus pickle cost (mitigated by batched writes)
* Bad, because ``k=60`` + ``0.9/0.1`` constants not yet per-domain
  configurable
* Bad, because double store hits per domain query (vector + BM25)

## More Information

* Code: ``utils_pkg/klea_utils/stores/retrieval/bm25.py`` (``BM25RetrieverManager``),
  ``klea_utils/stores/retrieval/vs.py`` (``VSRetriever``),
  ``klea_utils/stores/retrieval/base.py`` (``BaseKleaRetriever``),
  ``klea_utils/stores/ingestion.py`` (batched pickle corpus writes +
  ``store-lint``), ``rag_pkg/klea_rag/nodes/retrieve_info.py``
  (per-domain ``VSRetriever``/``BM25RetrieverManager`` loop + ``rrf_merge`` +
  ``max_refs_size`` truncation), ``rag_pkg/klea_rag/rag.py:303`` (``_configure_resources``
  wiring), ``rag_pkg/klea_rag/config.py`` (``PerDomainConfig.bm25_stores``),
  ``utils_pkg/klea_utils/ui/stores_create.py`` (``--bm25-store`` flag).
* Related: ``ADR-0008`` (always retrieve -- this ADR governs *which*
  stores that retrieval spans); ``ADR-0011`` (multiple query domains --
  BM25 benefits most when multiple domains are cited together);
  ``ADR-0001`` (worker-isolated chunking that also writes the BM25
  corpus); ``docs/concepts/rag.rst:174`` and
  ``docs/troubleshooting.rst`` (``chroma.sqlite3`` vs corpus file).
* Decisions codified ``2026-08-28``; BM25 work landed in ``2026-08-07``
  ``2da12eb incorporate bm25 stores`` / ``7500ca8`` / ``a7bdb41``
  alongside ``BaseKleaRetriever`` extraction.
