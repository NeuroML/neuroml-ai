---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Always retrieve for RAG queries

## Context and Problem Statement

A retrieval-augmented generation (RAG) pipeline can handle a user
query in two ways: (a) always run retrieval first and generate the
answer from the retrieved context, or (b) expose retrieval as a tool
call and let the LLM decide per query whether retrieval is needed.
Option (b) is natural for an agentic loop, but it means grounding,
citation, and corpus control become advisory -- the LLM may skip
retrieval, fall back to its frozen training data, or retrieve with a
poorly formed tool query.

Klea RAG is a RAG: its core promise is grounded, cited answers from a
curated corpus.  Should retrieval be mandatory or LLM-decided?

## Decision Drivers

* Grounding guarantee: the answer must come from the configured vector
  stores when they are relevant; training-data leakage must be
  opt-in, not accidental.
* Citations and reference material: the answer LLM needs the retrieved
  chunks rendered as ``serialize_reference_material`` blocks (per-file
  bibliographic header + numbered chunks) to cite and to stay
  transparent.
* Evaluation loop: ``Evaluator``/``RouteEvaluator`` assume there is
  retrieved context to judge; a skipped retrieval starves the
  generate->retrieve->answer->evaluate cycle.
* Tool use is not gone: RAG domains can still attach MCP servers whose
  tools (``ToolsPicker``/``ToolsCallerNode``) run *in parallel* with
  query generation (see ``rag.py:402`` ``_splitter_label``), so live
  data and store data complement each other.

## Considered Options

* **A. Expose retrieval as a tool call (LLM decides)** -- retrieval
  becomes one of the ``ToolsPicker`` choices; the LLM may or may not
  call it.  Rejected: the LLM sometimes skips retrieval for queries
  that look answerable from memory; when it does retrieve, the tool
  query and filters may be under-specified; citation coverage becomes
  variable and the evaluation loop has nothing to evaluate.
* **B. Always retrieve, then generate (chosen)** -- every domain-routed
  query fans out via the ``_splitter_label`` node (``rag.py:389``) to
  both ``GenerateRetrievalQuery`` and ``ToolsPicker``/``ToolsCallerNode``;
  ``RetrieveInfoNode`` (``rag_pkg/klea_rag/nodes/retrieve_info.py``)
  queries all of the domain's retrievers (``VSRetriever`` and/or
  ``BM25RetrieverManager`` via ``BaseKleaRetriever``) with RRF fusion
  and ``max_refs_size`` truncation, then ``AnswerFromContext`` generates
  from that context.  ``AnswerGeneral`` (``answer_general.py``) is only
  reached via explicit ``RouteQuery``/``RouteEvaluator`` branches
  (``non_domain_query``, ``fallback``) when retrieval is not
  applicable or is exhausted.

## Decision Outcome

Chosen option: "B. Always retrieve, then generate".

* ``rag.py:389`` ``_splitter_label`` node fans every ``domain_query``
  to ``GenerateRetrievalQuery`` *and* ``ToolsPicker`` in parallel;
  ``RetrieveInfoNode`` + ``ToolsCallerNode`` both feed into
  ``AnswerFromContext``.  This preserves the RAG contract while still
  giving the LLM tool access for live web/MCP data.
* ``RetrieveInfoNode`` (``retrieve_info.py``) uses the domain's
  ``filter_fields``-scoped ``config_filters`` and the shared
  ``max_refs_size`` budget; ``VSRetriever``/``BM25RetrieverManager``
  are built from ``RetrieverConfig`` per ``_configure_resources``.
* Fallback paths remain explicit: ``RouteQuery(non_domain_chat)`` and
  ``RouteEvaluator(fallback_to_training_data, max_retrieval_attempts,
  max_rewrite_attempts)`` decide when the pipeline reaches
  ``AnswerGeneral`` or ``FixedAnswer`` refusals, never the LLM's ad
  hoc tool-choice.
* The agent (``klea_agent``) is orchestration that *may* use retrieval
  as a tool.  Klea RAG itself never delegates retrieval to free tool
  choice.

### Consequences

* Good, because every domain-routed answer is grounded in the configured
  stores, citable, and bounded by ``max_refs_size``; the LLM cannot
  silently bypass the corpus.
* Good, because the evaluation loop (``Evaluator`` -> ``RouteEvaluator``:
  ``retrieve_more_info``/``rewrite_answer``/``modify_query``/
  ``best_effort``) has a stable contract: there is always context to
  re-rank, rewrite, or fall back from.
* Good, because MCP tools remain available (picker/caller run in
  parallel) without jeopardising grounding; store retrieval and live
  retrieval are complementary, not competing.
* Bad, because queries that could be answered from general knowledge
  still pay retrieval cost (embedding + vector/BM25 lookup).  For
  domains that truly need a no-retrieval fast path, the correct
  mechanism is a dedicated domain or the ``non_domain_chat`` branch,
  not per-query LLM gating.
* Bad, because a misconfigured ``filter_fields`` or a missing store
  (name/path mismatch, ``STORES_TEST_CONFIG`` drift) is exercised on
  *every* query -- but this visibility is also a forcing function for
  configuration hygiene (see ``docs/troubleshooting.rst`` Vector
  store mismatch).

### Confirmation

* ``concepts/rag.rst`` pipeline states the mandatory retrieval stage;
  ``retrieve_info.py`` + ``generate_retrieval_query.py`` + ``answer_from_context.py``
  are the three nodes that implement it.
* ``utils_pkg/tests/test_stores_retrieval.py`` + ``test_tools_caller.py``
  exercise both retrieval and tool-caller branches off the splitter;
  ``rag_pkg/tests/test_classify_question.py`` covers ``RouteQuery``
  domain vs non-domain routing to ``AnswerGeneral``/``FixedAnswer``.
* ``ty`` cross-package resolution for ``RetrieverConfig`` and
  ``FilterFieldInfo`` is via ``ty.toml`` extra-paths; ``docs: make html``
  renders the pipeline figure ``rag-lang-graph.png``.
* Live: ``klea-rag-serve`` logs per-domain retrieval (``domain``,
  ``k``, ``max_refs_size``) for every domain query before
  ``AnswerFromContext`` is called.

## Pros and Cons of the Options

### Always retrieve, then generate (chosen)

* Good, because grounding and citation are guaranteed for domain queries
* Good, because ``Evaluator``/``RouteEvaluator`` have a stable
  retrieve->answer->evaluate contract
* Good, because MCP tools still run in parallel (live + store data)
* Bad, because general-knowledge queries still pay retrieval cost (use
  ``non_domain_chat`` or a separate domain for that case)

### Retrieval as LLM tool call (LLM decides)

* Good, because trivial queries could skip retrieval
* Bad, because LLM may skip retrieval even when the corpus is relevant
* Bad, because citation coverage becomes variable and evaluation is
  starved

## More Information

* Pipeline: ``rag_pkg/klea_rag/rag.py:389`` (splitter) + ``nodes/generate_retrieval_query.py``,
  ``nodes/retrieve_info.py``, ``nodes/answer_from_context.py``,
  ``nodes/evaluator.py``/``nodes/route_evaluator.py``, ``nodes/tools_picker.py``/
  ``nodes/tools_caller.py``; ``utils_pkg/klea_utils/stores/retrieval/vs.py``
  /``bm25.py``/``base.py`` + ``ty.toml``.
* Related: ``ADR-0010`` (single vs multiple query domains -- multiple
  domains now augment this decision without leaving ``always retrieve``);
  ``ADR-0009`` (configurable no-answer fallback, evaluator warnings);
  ``ADR-0012`` (BM25 hybrid exact matches); ``docs/concepts/rag.rst`` and
  ``docs/troubleshooting.rst`` (empty-results branch).
* Decisions codified ``2026-08-28``; underlying wiring predates ADRs (early
  ``2026-03..04`` ``rag_pkg`` graph extraction, ``2026-08-21`` filter/BM25
  work).
