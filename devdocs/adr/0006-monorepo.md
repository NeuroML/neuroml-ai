---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Monorepo for all Klea packages

## Context and Problem Statement

Klea is four installable Python packages (``utils_pkg/klea_utils`` shared
lib, ``rag_pkg/klea_rag``, ``agent_pkg/klea_agent``, ``mcp_pkg/neuroml_mcp``)
plus vector-store and deployment artefacts.  ``klea_rag`` and
``klea_agent`` both depend on ``klea_utils``; the MCP servers, vector-store
backends, and the RAG/agent graphs are developed together and change
atomically (e.g. ``klea_utils.mcp.tool_impls`` -> ``bundled_tools``
wrappers -> ``BaseLangGraph._bundled_server_config``; ``filter_fields``
changes that touch ingestion, retrieval, and node prompts).

Should Klea be one repository or one repository per package?

## Decision Drivers

* Atomic cross-cutting changes: a ``ToolInfo`` or ``tool_impls``
  change should not require coordinated commits/PRs across two repos
  and a version-range chase (``klea_rag``/``klea_agent`` already depend
  on ``klea_utils``; see ``AGENTS.md:12``).
* Shared toolchain: single ``ruff.toml``/``ty.toml`` (``extra-paths`` for
  all four packages), single ``scripts/run_tests.sh`` (``pytest -v -n
  auto`` per ``*_pkg/tests``), single ``CHANGELOG.md`` and ``devdocs/``
  as a single source of truth (C4 model, store-create pipeline,
  permission design note).
* CI and local verification ergonomics: one place to express the
  incremental review-driven workflow (apply -> ``ruff check --fix``/
  ``ruff format``/``ty``/``--help``/``pytest`` -> diff for review ->
  wait) and the package-specific testing quirks (``pytest`` must run
  from inside the package directory; ``mcp_pkg`` ``addopts = -n 1``,
  ``localonly`` LLM marker).
* Vector-store and deployment assets are large and not suitable for the
  main repository.

## Considered Options

* **A. Polyrepo -- one repo per package** -- each ``*_pkg/`` is its own
  GitHub repo with its own ``CHANGELOG.md``, ``devdocs/``, and CI.
  Rejected: cross-package atomic changes become multi-PR
  ``klea_utils`` release -> ``klea_rag``/``klea_agent`` bump cycles;
  shared ``ty`` ``extra-paths`` and the monograph ``CHANGELOG.md``
  would be lost; ``devdocs/system/store-create.md`` spans ingestion +
  store + agent/RAG graph so would be homeless.
* **B. Monorepo (chosen)** -- one Klea repository contains ``utils_pkg``,
  ``rag_pkg``, ``agent_pkg``, ``mcp_pkg`` (each ``setup.cfg`` with its
  own ``version``), ``ty.toml``, ``ruff.toml``, ``scripts/run_tests.sh``,
  and ``devdocs/``.  Packages remain separately installable
  (``setuptools`` + ``setup.cfg``) and separately publishable.  This is
  the current layout (``AGENTS.md:32`` Packages at a glance).
* **C. Monorepo with a single version for all packages** -- variant of
  B where ``CHANGELOG.md`` and ``git tag`` move in lockstep for the
  whole repo.  Rejected: ``klea_utils`` and ``klea_rag`` are at
  different 0.x.y pre-1.0 cadences; a docstring fix in ``neuroml_mcp``
  should not bump the library version.
* **D. Monorepo with an integrated ``osb-mcp``** -- variant of B where
  the external OSB MCP server (``OSB/OSB-AI/mcp`` on HuggingFace)
  is vendors into this repo.  Rejected: slippery-slope risk -- see
  Consequences; ``osb-mcp`` now lives in the HuggingFace
  ``OSB/OSB-AI`` repo.

## Decision Outcome

Chosen option: "B. Monorepo with per-package ``setup.cfg`` versions".

* Layout: ``utils_pkg/`` (``klea_utils``), ``rag_pkg/`` (``klea_rag``),
  ``agent_pkg/`` (``klea_agent``), ``mcp_pkg/`` (``neuroml_mcp``),
  ``ty.toml`` (``extra-paths`` for all four), ``ruff.toml``,
  ``scripts/run_tests.sh``, ``CHANGELOG.md``, ``devdocs/`` per
  ``AGENTS.md:144`` structure.
* Versioning and publishing: version is tracked per package in each
  ``setup.cfg`` (``AGENTS.md:174``).  Releases are per-package: ``git tag
  v<version>`` + ``git push --tags`` triggers the OIDC trusted publisher
  workflow that builds and publishes to PyPI; only ``klea_utils`` and
  ``klea_rag`` are published today (``AGENTS.md:177``).  After a release
  the package is bumped to its next dev version.  There is no single
  GitHub Release for the whole repo -- only per-package tags on PyPI.
  ``CHANGELOG.md`` stays at the repo root and covers all packages with
  concise entries (one short bullet per user-visible change; Breaking
  changes first, then Added/Changed/Fixed/Dependencies; see
  ``AGENTS.md:184``).
* Storage boundary: vector stores and corpora are never committed to the
  monorepo.  They live in the HuggingFace deployment repos (e.g.
  ``deployments/huggingface/`` ``vector-stores/`` with ``.gitattributes``
  ``git lfs``/``git xet`` patterns) and are addressed as URI-style paths
  (``chroma:/``, ``qdrant:http://``, ``pgvector:postgresql://`` per
  ``AGENTS.md:141``).  The monorepo itself stays cloneable without the
  data.
* Organisational flag boundary: ``neuroml_mcp`` stays in the Klea monorepo
  and the ``NeuroML`` org for now because Klea is built under that flag;
  ``osb-mcp`` was deliberately kept out and lives in the OSB-AI
  HuggingFace repo.  This is an acknowledged case-by-case judgement,
  not a blanket rule.
* Workflow: the incremental review-driven loop (``AGENTS.md:14``) and
  testing quirks (run ``pytest`` from inside the package directory;
  ``mcp_pkg`` ``-n 1``; ``localonly`` requires an LLM and self-skips;
  ``utils_pkg/tests/test_stores_retrieval.py`` ``STORES_TEST_CONFIG``)
  apply repo-wide in one place.

### Consequences

* Good, because atomic cross-package edits are one PR with one diff
  (e.g. ``klea_utils/mcp/tool_impls/permission.py`` + its wrappers +
  ``graph/base.py`` ``_bundled_server_config`` + docs).
* Good, because ``ty`` cross-package resolution and single
  ``scripts/run_tests.sh`` remain trivial (one invocation exercises all
  four ``*_pkg/tests``; package table stays in one ``AGENTS.md``).
* Good, because ``devdocs/`` (C4 model, store-create pipeline, ADRs)
  and the monograph ``CHANGELOG.md`` are discoverable in one place
  (``devdocs/README.md`` + ``docs/developer-info.rst``).
* Bad, because per-package releases need more CI tweaking (per-package
  ``setup.cfg`` version + ``git tag v<version>`` + ``git push --tags`` +
  per-package trusted publisher vs a single GitHub Release).  A repo-level
  Release page would now be ambiguous -- releases are really PyPI releases
  per package via tags.
* Bad, because monorepo is a slippery slope: every adjacent tool could be
  argued into the repo (``osb-mcp`` vs ``neuroml-mcp`` above).
  ``neuroml_mcp``'s continued presence here is admittedly an
  organisational-flag judgement, so the boundary must be re-examined when
  new servers arise.
* Bad, because large artefacts (vector stores, pickled stores, model
  files) cannot be kept alongside code; they must be versioned and
  cached externally (HuggingFace ``deployments/huggingface/`` via
  ``git lfs``/``xet``; ``klea_utils.paths`` + ``.klea-cache/`` for local
  chunk caches), so repo clone does not imply a ready deployment.

### Confirmation

* ``ty`` cross-package resolution preserved via ``ty.toml`` ``extra-paths``
  for all four packages; ``ruff check --select I --fix``/``ruff format``
  still run from any package dir.
* ``bash scripts/run_tests.sh`` runs ``pytest -v -n auto`` per ``*_pkg/tests``
  (``mcp_pkg`` correctly ``-n 1`` via its ``pyproject.toml`` ``addopts``).
* Per-package ``setup.cfg`` ``version`` bump + ``git tag v<version>`` +
  ``git push --tags`` + OIDC trusted publisher -> PyPI verified on
  ``klea_utils``/``klea_rag``; ``CHANGELOG.md`` entries stay per-package
  with the prescribed grouping.
* ``deployments/huggingface/.gitattributes`` correctly marks
  ``vector-stores/**`` as ``filter=lfs``/``filter=xet`` and the deployment
  clones stay small; ``docs: make html`` still resolves cross-package
  imports via ``ty.toml``.

## Pros and Cons of the Options

### Monorepo with per-package versions (chosen)

* Good, because atomic cross-package changes are one PR/diff.
* Good, because shared ``devdocs/``, ``CHANGELOG.md``, ``ty``/``ruff``,
  and ``scripts/run_tests.sh`` are single source of truth.
* Bad, because per-package release needs extra CI + ``setup.cfg`` bump
  care and only yields ``git tag``-driven PyPI publishes (no monolithic
  GitHub Release page).
* Bad, because large data (vector stores) cannot live in the repo and
  must be versioned elsewhere (HuggingFace).
* Bad, because boundary discipline is required (``osb-mcp`` vs
  ``neuroml_mcp`` slippery slope -- must be judged case by case).

### Polyrepo

* Good, because each package has clean release/tag semantics on its own
  repo.
* Bad, because cross-package atomic edits become multi-repo PRs and
  version-range chases; shared ADRs and C4 model are homeless.

### Monorepo with single version for all packages

* Good, because one ``git tag`` publishes everything in lockstep.
* Bad, because docs-only ``neuroml_mcp`` fix would bump ``klea_utils``
  library version -- wrong pre-1.0 cadence.

## More Information

* Package table: ``AGENTS.md:32`` (and ``docs/index.rst:13``); commands:
  ``AGENTS.md:48`` (``uv pip install -r requirements-dev.txt``,
  ``scripts/run_tests.sh``, ``ruff``/``ty``); versioning/releases:
  ``AGENTS.md:174``; ``ty.toml`` extra-paths; ``utils_pkg/klea_utils/paths.py``
  platformdirs layout.
* Related: ``devdocs/adr/0004-bundled-stdio-server.md`` (atomic
  ``utils_pkg -> rag_pkg/agent_pkg`` bundled change that motivates the
  monorepo), ``devdocs/system/c4-container.md`` (packages as containers),
  ``devdocs/README.md`` structure, ``.agents/2026-08-16.md`` (initial
  ``devdocs`` creation).
* Vector-store storage: ``deployments/huggingface/.gitattributes``
  ``filter=lfs``/``filter=xet`` for ``vector-stores/**``; logical
  monorepo ``deployments/huggingface/`` submodule vs HuggingFace Space
  repo.
* Boundary: ``osb-mcp`` -> ``OSB/OSB-AI`` HuggingFace repo (external) vs
  ``neuroml_mcp`` staying under the ``NeuroML`` flag here -- see
  Consequences for the acknowledged judgement.
* Codified ``2026-08-28``; monorepo itself predates ADRs (early 2026-03
  ``rag_pkg``/``agent_pkg`` extraction promoting ``klea_utils`` to shared
  lib) so original commit range spans the first monorepo commits.
