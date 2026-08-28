---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Single HTTP stack on httpx with shared retry and lifespan session

## Context and Problem Statement

Klea's RAG, agent, MCP servers, and store-creation all make outbound HTTP
calls (LLM providers, HuggingFace, NeuroML-DB, OSB, DOI resolvers, web
fetch/download).  For a period two stacks coexisted: ``httpx`` in the API
and ``aiohttp`` in the MCP servers/tools.  ``aiohttp`` required a
separate ``aiohttp.TCPConnector`` pool per server, per-tool
retry/backoff, and distinct ``SessionLike`` fakes for tests.  Some
outbound tools (``web_fetch``, ``download_file``) also had ad-hoc
retry and SSRF handling.

How should Klea consolidate its HTTP client code?

## Decision Drivers

* One set of connection pooling, HTTP/2, and TLS behaviour across RAG,
  agent, MCP, and store creation (not per-server tuning).
* Shared retry/backoff that honours ``429`` and ``5xx`` as transient but
  not other ``4xx`` (so bad queries fail fast).
* SSRF guard (``klea_utils.mcp.tool_impls.ssrf``) must compose with the
  HTTP session, not be re-implemented per tool.
* Testability: tool tests should use one ``SessionLike`` protocol with
  ``stream``/``get`` (``httpx``-shaped fakes), not two.
* Must not reintroduce ``aiohttp`` -- its connector lifecycle (``await
  connector.close()``) complicated MCP lifespans.

## Considered Options

* **A. Keep both stacks** -- ``httpx`` for the API, ``aiohttp`` for MCP.
  Rejected: two pool lifecycles, two retry impls, two test fakes, and
  ``aiohttp`` pool cleanup races in FastMCP lifespans.
* **B. Migrate everything to ``aiohttp``** -- standardise on the MCP
  server's original client.  Rejected: ``httpx`` already provided ``http2``
  and unified ``Limits``; ``aiohttp`` lacked ``AsyncClient``-style
  ``http2=True`` negotiation that HuggingFace Spaces and Ollama benefit
  from.
* **C. Consolidate on ``httpx`` with shared helpers (chosen)** -- remove
  ``aiohttp`` everywhere; introduce ``klea_utils.api.utils._make_retryer_httpx``
  (``tenacity.AsyncRetrying`` with ``TransportError``/``TimeoutError``/``429``/``5xx``)
  and ``klea_utils.mcp.lifespan.make_http_session_lifespan`` (yields
  ``{http_session: httpx.AsyncClient}``).  FastMCP lifespans compose with
  ``|``; tools read ``ctx.lifespan_context["http_session"]``.  Keep the
  ``SessionLike`` protocol for tests (``stream``/``get``).

## Decision Outcome

Chosen option: "C. ``httpx`` everywhere with shared retry + lifespan
session".

* ``utils_pkg/klea_utils/api/utils.py:33`` -- ``_make_retryer_httpx(attempts, timeout)``
  (generous retry/budget defaults; ``attempts`` takes precedence over
  ``timeout``) and ``check_api_is_ready`` / ``_get_ready`` using it.
  Session helpers defer ``import httpx`` so importing the module never
  requires ``httpx``.
* ``utils_pkg/klea_utils/mcp/lifespan.py:26`` -- ``make_http_session_lifespan(session_key="http_session")``
  returns a FastMCP ``@lifespan`` that yields a shared ``httpx.AsyncClient``
  (``http2=True``, generous ``Limits``/timeout for bursty multi-user MCP
  servers; specific values are configurable defaults in code).
  Shared servers reuse it via ``bundled.py:34``
  ``lifespan=make_http_session_lifespan()``; apps wrap it in their
  ``FastMCP`` instance lifespan chain.
* ``utils_pkg/setup.cfg`` / ``requirements*`` -- ``aiohttp`` removed;
  ``httpx[http2]`` added; ``tenacity`` retained.
* ``AGENTS.md`` HTTP conventions codified: ``httpx`` preferred, ``aiohttp``
  removal noted, shared ``tool_impls`` + ``lifespan`` pattern mandated,
  ``http_session`` lifespan key documented.
* Tool tests updated to httpx-shaped fakes: ``mcp_pkg/tests`` and
  ``utils_pkg/tests/test_bundled_server.py`` implement ``SessionLike``
  (``stream``/``get``) rather than ``aiohttp`` fakes.

### Consequences

* Good, because one pool, one ``Limits`` tune, and one retry policy
  govern all outbound traffic (LLM, stores, MCP tools).
* Good, because ``http2=True`` and keep-alive reuse reduce TLS handshakes
  under bursty load (HuggingFace Spaces, Ollama).
* Good, because ``SessionLike`` fakes give a single test seam for
  ``web_fetch``/``download_file`` without touching real sockets.
* Good, because ``make_http_session_lifespan`` composes with ``|`` so the
  bundled ``klea-mcp`` server and per-MCP-server lifespans share the same
  session.
* Bad, because ``httpx`` adds ``h2``/``httpcore`` transitive deps (already
  required by ``fastmcp``/``langchain`` so not new weight in practice).
* Bad, because any remaining ``aiohttp``-specific tuning (e.g. per-host
  limits) must be re-expressed as ``httpx.Limits``.

### Confirmation

* ``mcp_pkg: pytest -v`` still 13 passed; ``utils_pkg: pytest -m "not
  localonly"`` 848 passed (httpx fakes, no sockets).
* ``docs: make html`` ``build succeeded``; CLI ``--help`` still defers
  heavy imports (``httpx`` import stays lazy inside retry helpers).
* Manual: ``Client(bundle_server).call_tool("web_fetch", ...)`` with an
  httpx-shaped ``http_session`` fake returns ``ToolResult``; ``check_ssrf``
  still gates the initial URL (redirect re-check remains a known
  best-effort limitation per ``devdocs/system/mcp-permissions.md:92``).
* Lint/type: ``ruff``/``ty`` clean for ``lifespan.py``/``api/utils.py``.

## Pros and Cons of the Options

### httpx single stack with shared helpers (chosen)

* Good, because single retry / pool / http2 behaviour everywhere
* Good, because ``aiohttp`` connector lifecycle races disappear
* Good, because test seam is one protocol (``SessionLike``)
* Bad, because httpx brings ``h2`` deps (already transitive)

### Keep both stacks

* Good, because zero migration cost
* Bad, because two pools, two retries, two fakes, and lifespan leaks

### Migrate everything to aiohttp

* Good, because MCP server's original client
* Bad, because no ``http2=True`` negotiation; loses ``httpx`` API parity

## More Information

* Code: ``utils_pkg/klea_utils/mcp/lifespan.py:26`` (session),
  ``utils_pkg/klea_utils/api/utils.py:33`` (retryer), ``utils_pkg/klea_utils/mcp/tool_impls/web_fetch.py`` /
  ``download_file.py`` (tools reading ``http_session``), ``utils_pkg/klea_utils/mcp/server/bundled.py:34``
  (lifespan composition), ``AGENTS.md`` HTTP conventions.
* Related: ``devdocs/system/mcp-permissions.md:92`` SSRF guard
  (``check_ssrf`` + ``is_private_or_reserved`` + redirect limitation);
  ``devdocs/adr/0004-bundled-stdio-server.md`` (bundled server that
  reuses this lifespan).
* Commits: ``6dccebe`` (``api/utils._make_retryer_httpx``),
  ``e38a6ad`` (lifespan consolidation), ``1c2e5e5`` (remove ``aiohttp``),
  ``3e9e22c`` (``web_fetch`` httpx robustness), ``0a52959``/``e580eac``
  (complete move + ``AGENTS.md`` codification), codified ``2026-08-28``
  (original decisions ``2026-08-13..14`` per ``git log --grep=httpx``).
