# devdocs -- internal development notes

Internal development notes for the Klea team.  This folder is *for us*:
architecture research, design discussions, and decisions that guide future
implementation.  It is intentionally kept separate from `docs/`, which is
the public, user-facing documentation site.

Rules:

- One file per topic, `kebab-case.md` (e.g. `mcp-permissions.md`).
- Keep notes high-level: decisions, trade-offs, pointers to source and
  commits.  Omit routine work (git log has the step-by-step edits).
- ASCII-only text, matching the repo file conventions in `AGENTS.md`.
- When an idea here gets implemented, update the public `docs/` and
  `CHANGELOG.md` at that point -- not before.

Structure:

- `system/` -- architecture and component contracts.  Mermaid diagrams
  and data-flow notes that explain how a subsystem works.  See
  `system/store-create.md` for ingestion.
- `adr/` -- Architecture Decision Records in MADR format, numbered as
  `NNNN-<slug>.md` (e.g. `0001-chunk-workers.md`) so they can be
  referenced by number.  Each file records one decision: context, options
  considered, outcome, and consequences.  See `adr/0001-chunk-workers.md`
  for the chunking/mass-ingestion decisions.  The template is
  `adr-template.md` at the `devdocs/` root.
- `.agents/` -- session logs (see `AGENTS.md`).

## Index

| File | Topic |
|------|-------|
| `system/c4-system-context.md` | C4 model Level 1: system context diagram (whole Klea product as the system in scope) |
| `system/c4-container.md` | C4 model Level 2: container diagram (Klea packages/services, datastores, shared lib, and their interactions + external systems) |
| `system/store-create.md` | Store creation pipeline: chunk, store, build, worker isolation, and cache layout |
| `system/mcp-permissions.md` | Filesystem permissions for MCP tools: the current in-tool check, why it cannot cover third-party servers, and what opencode does instead |
| `adr/0001-chunk-workers.md` | ADR-0001: Subprocess chunk workers and DOI-cache batching for large-corpus ingestion |
| `adr/0002-worker-retry.md` | ADR-0002: Retry worker batches that die with no results instead of marking them failed |
| `adr/0003-mcp-iserror-compliance.md` | ADR-0003: Strictly require MCP isError for tool execution failures |
| `adr/0004-bundled-stdio-server.md` | ADR-0004: Bundled stdio MCP server and tag-filterable tool filtering |
| `adr/0005-httpx-single-stack.md` | ADR-0005: Single HTTP stack on httpx with shared retry and lifespan session |
| `adr/0006-monorepo.md` | ADR-0006: Monorepo for all Klea packages |
| `adr/0007-mcp-permissions.md` | ADR-0007: Declarative path permissions with dual-layer check and deferred interactive policy |
| `adr/0008-always-retrieve.md` | ADR-0008: Always retrieve for RAG queries |
| `adr/0009-no-answer-fallback.md` | ADR-0009: Configurable fallback when no grounded answer can be generated |
| `adr/0010-guard-node.md` | ADR-0010: Cheap guard node for production deployments |
| `adr/0011-multiple-query-domains.md` | ADR-0011: Multiple query domains per RAG query |
| `adr/0012-bm25-hybrid.md` | ADR-0012: BM25 hybrid retrieval for exact string matches |
| `adr/0013-inspection-features.md` | ADR-0013: Inspection features for validating RAG output |
| `adr/0014-runtime-model-switching.md` | ADR-0014: Runtime per-request model switching with user-supplied API keys |
| `adr/0015-profile-env-config.md` | ADR-0015: Layered config via env file and profile-resolved JSON |
| `adr/0016-baselanggraph-orchestrator.md` | ADR-0016: BaseLangGraph as single model/MCP/VS orchestrator (Template Method) |
| `adr/0019-shared-abstract-nodes.md` | ADR-0019: Shared abstract node hierarchy (Template Method for nodes) |
