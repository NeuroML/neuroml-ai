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
| `system/store-create.md` | Store creation pipeline: chunk, store, build, worker isolation, and cache layout |
| `system/mcp-permissions.md` | Filesystem permissions for MCP tools: the current in-tool check, why it cannot cover third-party servers, and what opencode does instead |
| `adr/0001-chunk-workers.md` | ADR-0001: Subprocess chunk workers and DOI-cache batching for large-corpus ingestion |
