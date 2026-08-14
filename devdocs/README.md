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

## Index

| File | Topic |
|------|-------|
| `mcp-permissions.md` | Filesystem permissions for MCP tools: the current in-tool check, why it cannot cover third-party servers, and what opencode does instead |
