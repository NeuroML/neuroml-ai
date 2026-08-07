# AGENTS.md -- Klea monorepo

**NOTE:** `CLAUDE.md` is a symlink to this file (`CLAUDE.md -> AGENTS.md`).
Editing or writing to `CLAUDE.md` will overwrite `AGENTS.md` -- always edit
`AGENTS.md` directly.

**IMPORTANT: Read this entire file at the start of every session.** It contains
workflow, git conventions, session-log guidelines, and CLI patterns that must be
followed. Do not proceed until you have read every section below.

Multi-package Python project (setuptools + `setup.cfg`). Each `*_pkg/` is a
separate installable; `code_pkg` and `rag_pkg` depend on `utils_pkg`.

## Workflow

This repo uses an incremental, review-driven workflow.  After each step of a
plan:
1. Apply the change.
2. Run the relevant verification (lint, typecheck, test, --help).
3. Stop and present the diff for review.
4. Wait for feedback before proceeding to the next step.

If the review requires changes, address them and loop back to step 2.  Only
move to the next step after explicit approval.

Verification in step 2 covers:
- Lint + format: `ruff check . --fix` and `ruff format .` (in the affected package)
- Type check: `ty` (from repo root)
- If a CLI entry point was modified: `<cli-name> --help` to confirm it starts
- If tests exist for the changed code: `pytest -v <test-path>`

## Packages at a glance

| Dir | Package name | CLI entry |
|-----|-------------|-----------|
| `utils_pkg/` | `klea_utils` | -- (shared lib) |
| `code_pkg/` | `klea_code` | `klea-code` |
| `rag_pkg/` | `klea_rag` | `klea-rag`, `klea-rag-serve` |
| `mcp_pkg/` | `neuroml_mcp` | `nml-mcp` |

Each has its own `AGENTS.md` with architecture details -- refer to those for
package-specific commands, node layout, and conventions.

## Commands

```bash
# Dev install (editable, all packages)
uv pip install -r requirements-dev.txt

# Run all tests across packages (from repo root)
bash scripts/run_tests.sh        # pytest -v -n auto in each *_pkg/tests

# Single package test
# NOTE: pytest must be run from within a package directory, never from the
# repo root. Each package's pyproject.toml sets asyncio_mode="auto" (and
# mcp_pkg adds -n 1); a root-level run skips that config, so async tests and
# fixtures error out. Only bash scripts/run_tests.sh is meant to be run from
# the repo root.
cd mcp_pkg && pytest -v          # uses -n 1 (mcp tools are asyncio)
cd utils_pkg && pytest -v

# Run only tests that do NOT need an LLM (from within a package)
cd utils_pkg && pytest -m "not localonly"

# Lint + format
ruff check . --fix
ruff format .
ruff check . --select I --fix    # import sorting

# Type check
ty

# Pre-commit (CI gate)
pre-commit run --all-files
```

## CI flow

`.github/workflows/ci.yml` (pushes/PRs to main/development/*test*/**feat*/**fix*):
`uv pip install -r ./requirements.txt` -> `ollama pull qwen3:0.6b bge-m3` ->
`bash scripts/run_tests.sh` -> `ruff check . --exit-zero`

`.github/workflows/ruff.yml`: changed-files lint on PRs.

## Config & env loading

Both `KleaCode` and `RAG` orchestrators load configuration via:
1. An env file (`k=v` format, path from `KLEA_CODE_ENV_FILE` / `KLEA_RAG_ENV_FILE` or default `klea_code.env` / `rag.env`)
2. A JSON config file referenced inside the env file

`ty.toml` adds `extra-paths` for all four packages so type-checking resolves
cross-package imports.

## Testing quirks

- Tests marked `localonly` require an LLM -- skipped in CI.
- `utils_pkg/tests/test_stores_retrieval.py` reads `VS_TEST_CONFIG` env var (default `vector-stores-tests.json`).
- MCP tests are asyncio + single-process; do **not** run with `-n auto` (uses `addopts = -n 1` in `pyproject.toml`).
- All packages ignore `F403` and `F405` in ruff.

## Key references

Copyright format: `# Copyright 2026 Ankur Sinha <sanjay DOT ankur AT gmail DOT com>`
(`mcp_pkg` additionally requires `#!/usr/bin/env python3` shebang on every `.py` file.)

MCP tool auto-discovery: any function ending `_tool` is registered.

MCP server support + tool description guidance (docstring-first convention,
length/style, reusable template): `docs/concepts/mcp.rst`.

`BaseLangGraph` lives at `utils_pkg/klea_utils/graph/base.py` -- shared
setup -> MCP client -> vector stores -> compile graph template method.

Vector stores use URI-style paths: `chroma:/path/to/dir`, `qdrant:http://...`,
`pgvector:postgresql://...`.

## Session continuity

`.agents/YYYY-MM-DD.md` logs previous work (see `.agents/Readme.md` for template).
Read previous logs at session start; write one at session end.

Keep logs high-level -- decisions, architecture changes, outcomes only.
Git log has the step-by-step edits. Omit routine work.

## Git conventions

- `git add --intent-to-add <new-file>` so new files appear in `git diff`.
- Never stage/commit without explicit user approval.
- Show `git diff --stat` first, then full diff before committing so scope is clear at a glance.
- Conventional commit messages with issue numbers when applicable.

## Versioning

- Version is tracked in each package's ``setup.cfg`` (``version`` field).
- ``klea_utils`` and ``klea_rag`` are published to PyPI; ``klea_code`` and
  ``neuroml_mcp`` are not yet published.
- Pre-1.0 (0.x.y) releases: bump minor for new features, patch for bug fixes.
- When cutting a release:
  1. ``git tag v<version>`` and ``git push --tags``
  2. The OIDC trusted publisher workflow builds and publishes to PyPI
- After a release, bump to the next dev version in ``setup.cfg``.
- ``CHANGELOG.md`` is kept at the repo root, covering all packages.

## File conventions

- Use ASCII-only text. No unicode dashes, arrows, ellipsis, or emoticons.
- Preserve existing comments (TODOs, FIXMEs, notes, etc.) -- never remove or
  edit comments that are unrelated to the immediate change being made.

## CLI conventions

- Heavy imports (orchestrators, vector store backends, LLM libraries) must be
  deferred inside the function body of Typer commands, not at module level.
  Otherwise `--help` forces eager import of the entire dependency chain.
- Every deferred import must have a comment explaining *why* it is lazy, so the
  pattern is self-documenting for future maintainers.

## Logging conventions

- Use `f"{variable = }"` (Python 3.8+ f-string debug syntax) when logging
  variable values, one variable per line:
  ```python
  self.logger.debug(
      f"{current_chat = }\n"
      f"{model_info = }"
  )
  ```
- This avoids manual label strings and keeps the variable name in the log output.

## DO NOT SPECULATE OR GUESS

When the behavior of code is unclear, **do not propose speculative fixes or guess
at root causes**. Instead:

1. **Read the source** — look at the actual library code (LangGraph, LangChain,
   etc.) to understand how it works.
2. **Write a minimal test** — a 30-line script can confirm or rule out a
   hypothesis in minutes. Do this before proposing multi-file changes.
3. **Use debug logging** — add targeted log statements at key points instead of
   guessing what data looks like.
4. **Collect empirical evidence** — run the test, read the output, then reason
   from evidence.

Speculation wastes time. Every incorrect guess compounds into more guesswork.
If you cannot determine the answer from the source or a test, say so and ask.
