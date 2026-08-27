---
status: "accepted"
date: 2026-08-27
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Strictly require MCP `isError: true` for tool execution failures

## Context and Problem Statement

Klea's MCP tool implementations (`klea_utils/mcp/tool_impls/*.py: sqlite_query, read_file, list_files, web_fetch`, `klea_utils/mcp/tool_impls/repositories/*`, `osb_mcp/tools/sqlite_tools.py`) signal failure via a non-empty `error` (or legacy `Error`) string in their returned `dict` (`{"error": "Not a sqlite database file: ..."}`). Wrappers (`klea_utils/mcp/server/bundled_tools.py`, `mcp_pkg/neuroml_mcp/tools/*`, `osb_mcp/tools/*`) returned that `dict` directly. FastMCP's `Tool.convert_result` (`fastmcp/tools/base.py:359`) maps a plain `dict` to `ToolResult(content=[TextContent(json)], structured_content=dict, is_error=False)` — never `True` unless the tool returns `ToolResult(is_error=True)` or raises `ToolError` (`mcp/server/lowlevel/server.py:589` → `CallToolResult(isError: true)`).

Consequently the wire `CallToolResult` (`mcp/types.py:1363 isError: bool=False`) was `False` for every failure. Downstream consumers that correctly respect the spec — `klea_utils/tools.py:54 textualize_tool_results` (`if result.is_error: **Error:**`) and `klea_utils/nodes/tools_caller.py:112 success_count = sum(not r.is_error)` — treated failures as successes. The initial report was `sqlite_query` against a missing `osb-data-repos.sqlite` returning `is_error=False` with `error: "Not a sqlite database file: ..."` in `structured_content`.

The MCP spec (`modelcontextprotocol.io/specification/2025-06-18/server/tools#Error Handling`) distinguishes *protocol errors* (`error: {code, message}`) from *tool execution errors* (`result: {content: [TextContent(error)], isError: true}` for `API failure / Invalid input / Business logic`). Our `sqlite` "not a file" is the second category and `MUST` be `isError: true`.

We needed to decide how strictly Klea should enforce this boundary, especially for third-party servers (e.g. the external OSB MCP at `OSB/OSB-AI/mcp`) that had the same bug.

## Decision Drivers

* Spec compliance must be the source of truth — clients (`tools.py`, `ToolsCallerNode`, `dispatch_tool_calls`) should not second-guess the wire `isError`.
* Tool return values already carry full structured context (`columns`, `rows`, `tables`) that should be preserved for LLM remediation — raising `ToolError` strips `structured_content`.
* Permission gating (`klea_utils/mcp/dispatch.py:23 _denied_result is_error=True`) is already spec-compliant; other tools should be consistent.
* The fix must be testable via `Client(bundle_server).call_tool(..., raise_on_error=False)` and not require changes to framework-agnostic impls (`sqlite_query` etc. still return `dict` for direct Python use and `pytest`).

## Considered Options

* **A. Fix wrappers only, keep client strict (chosen)** — Each FastMCP wrapper returns `ToolResult(is_error=bool(error))` via a shared helper `klea_utils/mcp/tool_result.py:to_result` (`utils_pkg/klea_utils/mcp/tool_result.py:1`). The helper preserves `structured_content` and sets `is_error` from `error`/`Error`. Clients (`dispatch.py`, `tools.py`, `ToolsCallerNode`) strictly respect `is_error` per `mcp/types.py`. Non-compliant third-party servers will appear as success until they are fixed — their bug is visible and must be fixed at the server.

* **B. Fix wrappers + add client-side normalization** — Wrappers as in A, plus `klea_utils/mcp/dispatch.py:91 _normalize_result` and `klea_utils/tools.py:54 _is_error_result` that treat a non-empty `error` field in `structured_content`/`data` as `is_error=True` even when the wire says `False`. Masks third-party non-compliance so LLM sees `**Error:**` immediately.

* **C. Fix impls to raise `ToolError`** — Change every `tool_impls/*.py` to `raise ToolError(msg)` on failure instead of `return {"error": msg}`. Rejected: strips `structured_content`, changes `pytest` expectations (`test_tools_sqlite.py:13` expects `dict`), and couples impls to the MCP transport.

## Decision Outcome

Chosen option: **A. Fix wrappers only, keep client strict**, because it enforces the spec at the producer (where `error` is known) without hiding non-compliance in the consumer. The alternative (B) would silently correct broken servers and make regressions invisible; the failing `osb-data-repos.sqlite: is_error=False` case is now fixed at its OSB server (`osb_mcp/tools/sqlite_tools.py:60`) rather than papered over in Klea's client.

* Added `klea_utils/mcp/tool_result.py:to_result` (`utils_pkg/klea_utils/mcp/tool_result.py:1`) — checks `error`/`Error` (trimmed) and returns `ToolResult(content=[TextContent(json)], structured_content=dict, is_error=bool(error))`.
* Updated 9 wrappers: `klea_utils/mcp/server/bundled_tools.py:18` (`web_fetch`, `list_files`, `read_file`, `download_file`, new `sqlite_query`/`sqlite_schema`), `mcp_pkg/neuroml_mcp/tools/code_tools.py:59` (`list_files`, `run_python_code` with `returncode` → `error`), `mcp_pkg/neuroml_mcp/tools/neuroml_tools.py:178` (`run_lems_simulation`), `get_models_from_neuromldb`/`get_repositories...` normalising `Error` → `error`, and all `osb_mcp/tools/*` (sqlite, biblio, github, biomodels, figshare, dandi, download, osb_tools).
* Fixed impl inconsistency `klea_utils/mcp/tool_impls/list_files.py:65` `truncated: "False"` → `bool` to match `read_file.py`/`web_fetch.py` and spec-like booleans.
* Kept `klea_utils/mcp/dispatch.py` and `klea_utils/tools.py` strict: `dispatch` inserts `res` unchanged (`dispatch.py:91`), `textualize_tool_results` checks `result.is_error` only (`tools.py:54`). A non-compliant server's `{"error": "..."}` with `is_error=False` will now surface as a code block, not `**Error:**`, making the server bug visible.

## Consequences

### Positive

* All Klea and OSB servers are now spec-compliant: `sqlite_query` missing file → `isError:true` (`osb-data-repos.sqlite` repro fixed), `read_file`/`web_fetch`/`list_files`/`download` likewise; `Client(bundle_server).call_tool(..., raise_on_error=False)` shows correct `is_error`.
* Structured context is preserved for LLM repair (`columns`, `known tables` hint from `sqlite_query.py:147`).
* The `tool_result` helper is a single, testable bridge (`FastMCP` preserves `ToolResult.is_error` via `tools/base.py:318`).

### Negative

* An external, still-buggy MCP server will until fixed appear as success (` ```json {"error": ...}``` `) rather than `**Error:**`. This is intentional — the server must be updated to use `to_result` or `ToolError`.
* `download_files` (`klea_utils/mcp/tool_impls/download_file.py:167`) and repository source helpers remain `dict`-returning; if ever exposed as tools they must also use `to_result`.

### Confirmation

* Updated `utils_pkg/tests/test_bundled_server.py`: expects 6 bundled tools (added `sqlite_query`, `sqlite_schema` with `checkpaths=["db_path"]`, `readOnlyHint:true`).
* Updated `mcp_pkg/tests/test_neuroml_tools.py` and `osb_mcp/tests/test_sqlite_tools.py`, `test_repository_tools.py`, `test_biblio_tools.py` to expect `ToolResult` (`assert not result.is_error; data = result.structured_content; assert data["error"]==""`) and `assert result.is_error` for failure cases.
* Manual verification: `python -c Client(bundle_server).call_tool("sqlite_query", ... SELECT * FROM t) is_error False` vs `SELECT * FROM missing is_error True` (`error: no such table`).
* Lint/type: `ruff check` (fix), `ruff format`, `ty` (pre-existing errors only), `pytest -v` 100 `utils_pkg` + 14 `mcp_pkg` + OSB `mcp/tests` pass.

## Pros and Cons of the Options

### A. Fix wrappers only, keep client strict (chosen)

* Good, because spec is enforced at the producer where `error` is authoritative
* Good, because structured context is preserved for LLM remediation
* Good, because non-compliant servers are visibly failing (no silent masking)
* Bad, because a still-buggy external server temporarily surfaces as success until patched

### B. Fix wrappers + client normalization

* Good, because external OSB bug would have been hidden immediately for LLM UX
* Bad, because it masks server non-compliance and adds per-call branching + warning noise (`dispatch.py:_normalize_result`)
* Bad, because two sources of truth for `is_error` (wire vs `structured_content.error`)

### C. Raise ToolError in impls

* Good, because FastMCP automatically yields `isError:true` via `lowlevel/server.py:589`
* Bad, because it strips `structured_content` (only `TextContent(str(exc))` remains)
* Bad, because framework-agnostic impls would become MCP-coupled and break direct `pytest` usage

## More Information

* MCP spec: `modelcontextprotocol.io/specification/2025-06-18/server/tools#Error%20Handling`, `mcp/types.py:CallToolResult`
* FastMCP mapping: `fastmcp/tools/base.py:ToolResult`, `fastmcp/tools/base.py:convert_result`, `mcp/server/lowlevel/server.py:_make_error_result`, `fastmcp/client/mixins/tools.py:_parse_call_tool_result`
* Related: `devdocs/system/mcp-permissions.md`, `klea_utils/mcp/dispatch.py:23 _denied_result`, `klea_utils/mcp/tool_impls/list_files.py:65` truncated type fix.
* Code: `utils_pkg/klea_utils/mcp/tool_result.py`, `utils_pkg/klea_utils/mcp/server/bundled_tools.py:215`, `osb_mcp/tools/sqlite_tools.py:60`, `mcp_pkg/neuroml_mcp/tools/code_tools.py:59`, `mcp_pkg/neuroml_mcp/tools/neuroml_tools.py:219`.
