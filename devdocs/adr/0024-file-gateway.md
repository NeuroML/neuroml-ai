---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# File-gateway pattern for external-repository tools

## Context and Problem Statement

Klea's ``nml-mcp`` NeuroML server and the RAG's MCP layer need uniform
file gateway helpers for external repositories (GitHub, FigShare, DANDI
Archive, BioModels).  Each repository has its own paginated API and file
listing (e.g. GitHub ``/contents``, FigShare ``/articles``, DANDI
paginated asset listing -- BFS over ``datasets/<id>/versions/<v>/assets/``
with continuation tokens).  Naively each tool would
re-implement ``httpx`` session handling, SSRF gating, and retry, and
the ``agent`` would need to sniff URLs to decide which tool to call.

How should Klea provide a uniform, externally-safe file-gateway
contract that ``tool_picker`` can expose declaratively via tags?

## Decision Drivers

* Uniform contract: every file gateway must return the same shape
  ``{source, url, version, versions, files, error}`` so the caller
  (``ToolsCallerNode`` + LLM) can adapt without per-repo parsing.
* External-repository safety: HTTP fetches must be gated
  (``check_ssrf``/``is_private_or_reserved``) and reuse the shared
  ``httpx`` session (``ADR-0005``) via ``http_session`` lifespan.
* Testability without sockets: tool tests need ``SessionLike`` fakes
  (``stream``/``get``) per ``AGENTS.md`` HTTP conventions.
* Discoverability: the tool set should be selectable via the same
  ``ToolInfo.tags`` (``include_tags``/``exclude_tags``) and
  ``ToolAnnotations`` (``readOnlyHint``/``destructiveHint``) seam as the
  bundled tools (``ADR-0004``), not a bespoke flag.
* No URL-sniffing dispatch: ``detect_source(url)`` would conflate
  ``context`` (``github:…``) with ``version``; per-source tools let the
  LLM already know the source.

## Considered Options

* **A. Per-repo MCP tools calling ad-hoc ``httpx``** -- each tool owns
  its own ``AsyncClient``, retry, and SSRF check.  Rejected: the same
  ``httpx[http2]`` + ``make_http_session_lifespan`` + ``_make_retryer_httpx``
  + ``SessionLike`` test seam from ``ADR-0005`` would be re-established
  per repository.
* **B. Central dispatcher with URL-sniffing** -- one ``file_gateway_tool``
  that inspects the URL for ``github``/``figshare``/``dandi``/``biomodels``
  and dispatches.  Rejected: ``context`` (deployment, repo) is not
  ``version``; the sniff conflates them and forces the LLM to paste raw
  URLs instead of using tagged tools.
* **C. Unified ``tool_impls/repositories/`` + per-source tools sharing a
  general-pattern wrapper contract (chosen)** -- ``klea_utils/mcp/
  tool_impls/repositories/`` adds a subpackage:
  ``sources.py`` (``_get_json``: honest ``User-Agent``, shared
  ``_make_retryer_httpx`` retryer, SSRF guard, ``httpx.AsyncClient``
  from ``ctx.lifespan_context["http_session"]``) plus
  ``github.py``/``figshare.py``/``dandi.py``/``biomodels.py``, each
  exposing ``*_list_versions`` and ``*_list_files`` that return the
  uniform ``{source, url, version, versions, files, error}`` dict.
  ``ToolInfo.tags`` expose them as ``{web, download, github}`` etc.,
  filtered via ``general.bundled_tools`` + ``MCPConfig`` per ``ADR-0004``.
  The ``agent``/``RAG`` picker sees per-source tools, not a dispatcher,
  so the LLM already knows the source.  ``download_file(s)`` remains a
  separate tool; ``case 3`` (multi-file ``download_files``) uses a
  bounded ``Semaphore`` helper when needed.

## Decision Outcome

Chosen option: "C. Unified ``tool_impls/repositories/`` pattern with
per-source tools".

* Location: ``utils_pkg/klea_utils/mcp/tool_impls/repositories/``
  ``sources.py`` (``_get_json`` + shared retry/SSRF + ``get``),
  ``github.py`` (``_asset``/``_folder``), ``figshare.py`` (``_figshare_*
  ``), ``dandi.py`` (paginated asset listing, BFS over the nested asset
  tree with continuation tokens), ``biomodels.py`` (``_biomodels_*``)
  + ``download_file.py`` ``download_files`` batch helper
  (``Semaphore``-bounded when needed).
* Contract: ``repository_file_gateway`` is the *pattern*, not a
  single tool; each per-source pair (``github_list_versions``/
  ``github_list_files`` etc.) returns ``{source: str, url: str,
  version: str, versions: list[str], files: list[FileInfo], error: str}``.
  ``error`` is the ``to_result``-driven ``is_error`` signal (per
  ``ADR-0003``); success is ``error==""``.
* Tool exposure: ``mcp_pkg/neuroml_mcp/tools/`` registers the
  per-source tools with ``@tool_meta(ToolInfo(tags={"web","download",
  "github"}, read_only=True))`` etc.; they are tag-filterable as
  ``include_tags: ["github"]`` like any bundled tool.
* Dropped OSB jargon: earlier ``context`` param renamed to neutral
  ``version`` (commit ``5409d55``); URL-sniffing ``detect_source``
  removed so the LLM's source knowledge is explicit.

### Consequences

* Good, because adding a provider (e.g. ModelDB) is one new
  ``repositories/<provider>.py`` + two ``@tool_meta`` wrappers via the
  same contract; no per-tool HTTP/SSRF/retry re-implementation.
* Good, because per-source tools let the LLM know the source
  declaratively (tag filtering) rather than pasting URLs into a
  dispatcher.
* Good, because ``httpx`` integration is inherited (``install.rst`` HuggingFace
  inference API, ``mcp/lifespan`` session) and SSRF is single-point
  (``sources.py``) so tests need only ``SessionLike`` ``get`` fakes.
* Bad, because each provider still needs two wrappers (versions + files)
  with small per-API pagination quirks (GitHub ``/contents`` vs DANDI
  paginated asset listing).
* Bad, because the uniform ``files`` shape hides provider-specific
  metadata (e.g. GitHub ``sha``) behind a single ``FileInfo`` type;
  richer provenance would need a per-provider extension.

### Confirmation

* ``utils_pkg/tests/test_osb_sources.py`` (``_asset``/``_folder`` helpers,
  ``github``/``figshare``/``dandi``/``biomodels`` version+files
  contracts) plus ``test_mcp_registry.py:22`` ``ToolInfo`` tag/``_meta``
  assertions.
* ``ty`` ``extra-paths`` for ``tool_impls/repositories``; ``ruff``
  clean for the subpackage; ``docs: make html`` still renders the
  ``mcp`` concept with the external-repository tag vocabulary.
* Live: ``agent_pkg`` OSB-AI tools (``github_list_files`` etc.) return
  the uniform dict via ``SessionLike`` fakes; the dispatch layer never
  sees provider-specific shapes.

## Pros and Cons of the Options

### Unified repositories/ pattern with per-source tools (chosen)

* Good, because one contract for all providers (``{…error}``)
* Good, because per-source tools let the LLM know the source via tags
* Good, because HTTP/SSRF/test seam are shared (``ADR-0005``)
* Bad, because each provider still needs two thin wrappers

### Per-repo ad-hoc httpx

* Good, because minimal abstraction
* Bad, because per-tool HTTP/SSRF/retry duplication

### Central dispatcher with URL sniffing

* Good, because one tool
* Bad, because conflates ``context`` vs ``version``; LLM must paste URLs

## More Information

* Code: ``utils_pkg/klea_utils/mcp/tool_impls/repositories/``
  (``sources.py``, ``github.py``/``figshare.py``/``dandi.py``/
  ``biomodels.py``, ``errors.py``), ``mcp/tool_impls/download_file.py``
  (``download_files`` batch helper), ``mcp_pkg/neuroml_mcp/tools/*``
  (per-source ``@tool_meta`` wrappers), ``mcp/lifespan.py`` (``http_session``).
* Related: ``ADR-0005`` (``httpx`` single stack that this pattern reuses),
  ``ADR-0003`` (``to_result`` ``isError`` for uniform ``error``),
  ``ADR-0007`` (``check_path_access`` + ``check_ssrf`` that this pattern
  composes with).
* Commits: ``5409d55`` (unified ``repositories/`` subpackage +
  ``context``→``version`` rename + ``detect_source`` removal),
  ``aebe544``/``f9305da``/``ea1ec3e``/``31897d3`` (per-provider
  ``*_list_versions``/``*_list_files``).
* Codified ``2026-08-28``; landed ``2026-08-25`` during the OSB-AI
  repository-gateway sprint.
