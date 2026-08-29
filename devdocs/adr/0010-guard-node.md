---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Cheap guard node for production deployments

## Context and Problem Statement

All production deployments that are open to the internet --
HuggingFace demo Spaces (``NeuroML/NeuroKLEA``, ``OpenWormLLM``) are the
current example, but any public ``klea-rag-serve`` deployment is the same
-- are subject to spam, with or without bots.  Human spam is also an
issue.  Bots crawled the ``/query``
endpoint with spam, costing inference budget and polluting evaluation
loops, which first surfaced the problem.  A full request-level rate
limiter (e.g. per-IP token bucket in front of the FastAPI app) would
require infrastructure many deployment targets do not have (the Spaces
container has none), and content-based blocking at the graph level was
missing.

No guard existed originally -- the graph was ``classify -> retrieve ->
answer -> evaluate`` with no ``guard`` role in ``BaseLangGraph.llm_models``.
The spam wave forced the question of how to reduce spam.  The ADR
records the decision to add a dedicated gate: a configurable guard node
implemented via ``GuardNode`` + ``GuardRouterNode`` as the first graph
step, with a new ``guard`` role in ``BaseLangGraph``/``RAG``.  Should
the graph gain that step at all, and under what conditions should it
run?

## Decision Drivers

* Must be cheap: spam is high-volume (bot or human); the gate must be
  a small model with minimal context, or public deployments will not
  enable it.
* Must not be strictly required: local single-user RAG should run
  without a guard model.
* Must be skippable when not configured, without forking the graph or
  changing node wiring per deployment -- so the same binary serves
  HuggingFace demos, production, and local single-user with different
  env vars.
* Must compose with the existing ``START -> InitRAGState -> Guard ->
  GuardRouter -> ClassifyQuestion / Decline`` topology and be useful
  beyond HuggingFace (any internet-facing production deployment).

## Considered Options

* **A. No guard node; rely on external rate limiting** -- front the
  FastAPI app with an IP-based limiter or platform's built-in
  protections.  Rejected: HF Spaces has no such knob on any tier and
  public Spaces are open to the internet; the same exposure exists for
  any internet-facing production deployment, so without a guard spam
  still reaches the LLM.  Even where a limiter exists, bots and agents
  find ways around it (rotating IPs, burst patterns); it is a
  deployment/infra-level tweak, not a software-scope fix for Klea, and
  would still require a content gate inside the graph.
* **B. Hard-required guard node (always runs)** -- every query must pass
  ``GuardNode``.  Rejected: local ``klea-rag-serve`` with no embedding
  model or small model footprint would be forced to pull
  ``llama-guard3:1b`` even when not needed.
* **C. Cheap, skippable guard node (chosen)** -- introduce
  ``GuardNode`` (``utils_pkg/klea_utils/nodes/guard.py``) +
  ``GuardRouterNode`` (``guard_router.py``) as the first graph step.
  When ``llm_models["guard"].model_name`` is empty, the node short-
  circuits via ``_pre_exec`` (returns ``"safe"``); the router then
  takes the ``safe -> ClassifyQuestion`` edge immediately.  When set
  (e.g. ``KLEA_RAG_GUARD_MODEL=ollama:llama-guard3:1b``), the node
  calls the guard model with the query and routes ``safe``
  vs ``unsafe``.  ``unsafe`` goes to ``FixedAnswer(Declining query)``
  ("I cannot respond to this query. Please try another.") per
  ``rag_pkg/klea_rag/rag.py:191``.
* **D. Filter at the API layer** -- reject in ``klea_utils/api/chat.py``
  before graph ``ainvoke``.  Rejected implicitly: the graph already
  carries the guard roles and the same logic is shared by ``klea_agent``
  (also ``llm_models["guard"]``), so centralising in the graph avoids
  duplication and keeps the topology visible (``rag-lang-graph.png``).

## Decision Outcome

Chosen option: "C. Cheap, skippable guard node".

* New role and nodes: ``GuardNode`` (``guard.py:21``) ``model_type =
  "guard"``, ``model_defaults = {"temperature": 0.3, "max_output_tokens":
  2048}``, ``BaseLLMNode``.  ``_pre_exec`` returns ``bool(model_name)``;
  when falsy the node emits ``"safe"`` via ``_get_default_error_result``.
  ``_update_state`` maps ``"unsafe" in content`` -> ``{"guard_decision":
  "unsafe"}`` else ``"safe"``.  Error guard returns ``"safe"`` as
  well so transient LLM errors do not block the demo.  ``GuardRouterNode``
  (``guard_router.py:19``) reads ``state.guard_decision`` (default
  ``"safe"``) and returns that string as the conditional edge key.
  Both were new: before this decision ``BaseLangGraph.llm_models`` had
  no ``guard`` entry and the graph was ``classify -> retrieve -> ...``.
* Graph wiring added via ``BaseLangGraph`` / ``RAG`` updates: ``rag_pkg/
  klea_rag/rag.py:98`` introduces ``llm_models["guard"]`` as
  ``required=False``/``modifiable=False`` so ``_check_required_models``
  warns but does not prevent startup, and per-chat overrides cannot
  change the guard; ``rag.py:191`` inserts ``START -> InitRAGState ->
  Guard -> GuardRouter --safe--> ClassifyQuestion`` vs ``--unsafe-->
  Declining query -> END`` (agent similarly).  This was not a
  pre-existing topology.
* Configuration: ``KLEA_RAG_GUARD_MODEL`` / ``KLEA_AGENT_GUARD_MODEL``
  per ``docs/install.rst:263`` (empty value disables).  ``AGENTS.md``
  env schema is generated from ``llm_models`` roles, so the env var is
  ``KLEA_<APP>_GUARD_MODEL`` automatically.

### Consequences

* Good, because any internet-facing deployment -- public HF demos
  (``NeuroML/NeuroKLEA``, ``OpenWormLLM``) and future production -- gets a
  cheap spam gate without external infra; spam
  that is obviously unsafe (bot or human) is short-circuited before
  ``ClassifyQuestion`` / retrieval / answer generation.
* Good, because local single-user RAG pays nothing when no guard model
  is configured -- the node returns ``"safe"`` without an LLM call.
* Good, because the graph topology is uniform across deployments
  (``init -> guard -> router -> classify`` always exists), so the same
  binary serves HuggingFace demos, production, and local single-user with
  different env vars.
* Bad, because the guard is an extra LLM call on every query.  The cost
  is kept cheap by using a small guard model and by checking only the
  initial query (not each retrieval or evaluation loop iteration), but
  it does add latency and inference cost versus no guard.
* Bad, because the small guard LLM can mislabel queries: a false
  positive (safe query -> ``unsafe``) causes an unnecessary hard
  refusal ("I cannot respond to this query") even though retrieval
  would have answered; a false negative (unsafe/spam -> ``safe``)
  lets spam through with no benefit but still pays the guard cost.
  Heuristic ``safe/unsafe`` is a content filter, not a forensic
  classifier, and domain-specific academic phrasing can trigger
  over-blocking.
* Bad, because the guard is content-based, not rate-based: spam that
  looks safe still consumes a guard-model inference before classification.
  A per-IP token bucket would still be the correct complement for
  volumetric abuse by bots or humans.
* Bad, because failure is fail-open by design: ``_get_default_error_result``
  returns ``"safe"`` so transient guard-model errors (timeout, HTTP
  ``5xx``/``429``, missing ``HF_TOKEN``) let the query through rather
  than blocking the deployment.  This trades availability for safety
  and is intentional per ``guard.py:82`` but must be understood as a
  blind spot.
* Bad, because only the initial query is checked; later user turns or
  tool-retrieved content within the same graph run are not re-screened.
  The ``"unsafe" in content`` substring match in ``guard.py:75`` is
  also brittle (case/wording dependent) compared to a structured
  schema.
* Bad, because the guard model itself is a dependency to provision
  (``ollama pull llama-guard3:1b`` / HuggingFace gated model via
  ``HF_TOKEN``) and its ``safe/unsafe`` signal is heuristic, not a
  hard policy -- it is a filter for abusive content, not a forensic
  spam classifier.
* Good, because the classifier and the configurable no-answer fallback
  (``ADR-0009``) also help against abuse of free LLM resources: users
  trying to abuse public-facing systems with out-of-domain queries get a
  ``cannot answer this query`` refusal when ``fallback_to_training_data``
  is off, so the abuse path also ends without a full answer generation.
  This is not part of this ADR but composes with it.

### Confirmation

* ``rag.py:191`` + ``guard.py``/``guard_router.py`` wiring verified via
  ``rag_pkg/tests`` (classify/guard routing) and live ``klea-rag-serve``
  on ``cpu-basic``: with ``KLEA_RAG_GUARD_MODEL=""`` every query takes
  the ``safe`` path immediately (no guard LLM call); with
  ``llama-guard3:1b`` set, spam queries take ``unsafe -> Declining
  query``.
* ``docs: make html`` still renders the pipeline figure;
  ``ty`` cross-package ``guard`` role via ``ty.toml`` extra-paths is
  satisfied; ``install.rst:263`` documents the empty-to-disable contract.

## Pros and Cons of the Options

### Cheap, skippable guard node (chosen)

* Good, because spam gate is cheap and deployable on any internet-facing
  deployment
* Good, because local RAG pays nothing when guard model is absent
* Good, because topology is uniform (no per-deployment graph fork)
* Good, because the classifier + no-answer fallback (``ADR-0009``)
  complement the guard: out-of-domain abuse gets a refusal without a
  full generation when ``fallback_to_training_data`` is off
* Bad, because extra LLM call on every query (kept cheap by small model
  + initial-query-only check, but still latency/cost)
* Bad, because small LLM can mislabel (false positive -> hard refusal;
  false negative -> spam passes; see Consequences)
* Bad, because fail-open on guard error lets queries through on
  transient failures
* Bad, because content-based only; volumetric or human spam that looks
  safe still costs a guard inference until a rate limiter is added

### No guard node; external rate limiting

* Good, because zero graph cost (infra-level tweak, not Klea code)
* Bad, because no such limiter exists for HF Spaces and any limiter is
  not fool-proof -- bots/agents rotate IPs and find ways around it; it
  complements but does not replace a content gate inside Klea's
  software scope

### Hard-required guard node

* Good, because every query is screened
* Bad, because local single-user must pull a guard model

## More Information

* Code: ``utils_pkg/klea_utils/nodes/guard.py:21`` (node),
  ``utils_pkg/klea_utils/nodes/guard_router.py:19`` (router),
  ``rag_pkg/klea_rag/rag.py:98`` (``guard`` ``required=False``,
  ``modifiable=False``), ``rag_pkg/klea_rag/rag.py:191`` (graph wiring),
  ``klea_utils/llm.py`` provider wiring for ``KLEA_*_GUARD_MODEL``,
  ``AGENTS.md`` env-generation from ``llm_models`` roles.
* Related: ``ADR-0008`` (always retrieve -- guard sits before retrieval,
  so spam never pays embedding/vector cost); ``ADR-0009`` (fallback
  decisions also route through evaluation, not guard); ``docs/concepts/rag.rst``
  6-stage pipeline (Guard/Classify/Retrieval/...), ``docs/troubleshooting.rst``
  (``map-lint``/``store-lint`` still run when guard is disabled);
  ``docs/install.rst:263`` empty-to-disable contract is deployment-agnostic
  (HF demos enable it, local single-user may leave it empty).
* Decisions codified ``2026-08-28``; no guard existed before
  (``2026-04..05`` RAG graph was ``classify -> retrieve -> answer ->
  evaluate`` with no ``guard`` role in ``BaseLangGraph``); introduced
  and made skippable during the ``2026-08`` HF public demo spam wave
  (commit ``d288d94 feat: allow skipping guard node`` makes the
  ``_pre_exec`` gate explicit; earlier guard wiring from ``2026-04-14``
  ``4293404``/``f84b751``/``4a97e40`` predates the cheap-skip change).
