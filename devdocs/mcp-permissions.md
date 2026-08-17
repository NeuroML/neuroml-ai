# MCP tool permissions: current state, limits, and options

Status: research/design note.  Updates to this note should be reflected in
the permission layer as it evolves.

## Current state

`klea_utils.mcp.tool_impls.permission` provides `check_path_access(path,
project_root=None)` and `PermissionDeniedError`.  It is an author-side
defense layer:

- `path` is allowed only when it resolves inside `project_root` (default:
  the current working directory).  Both sides are fully resolved first, so
  `..` traversal and symlink escapes outside the boundary are caught.
- Every Klea-authored tool that reads or writes the filesystem must gate
  its path arguments through `check_path_access` and return a clear,
  non-halting error on denial:
  - `list_files` (`klea_utils/mcp/tool_impls/list_files.py`) -- takes
    `project_root`.
  - `download_file` (`klea_utils/mcp/tool_impls/download_file.py`) -- takes
    `project_root`.
  - `download_file_to_cache` scopes its boundary to its own cache
    directory, so per-app cache helpers keep working unmodified.

`check_tool_arguments_permissions(tool_meta, arguments, project_root)`
(`klea_utils/mcp/tool_impls/permission.py`) is the client-side counterpart: it
reads the `checkpaths` key from a tool's MCP `meta` dict and checks each
declared path argument without raising.  It never touches a server it does
not control, so the gate is:

- **Author-side (in-tool):** the checks above, run inside the tool
  implementation.
- **Client-side (pre-dispatch):** `klea_utils.mcp.dispatch.dispatch_tool_calls`
  runs `check_tool_arguments_permissions` on every call before it reaches
  the MCP server.  Denied calls never reach the server; they become a
  synthetic, non-halting error result so the LLM can adapt.  The gate runs
  in the shared `ToolsCallerNode` (`klea_utils/nodes/tools_caller.py`) used
  by both Klea Agent and Klea RAG.

Both agents/RAG are expected to run from the directory the user is working
in, so the client-side gate uses `project_root=None` (the current working
directory) by default -- the same boundary the in-tool checks default to,
so the two layers agree.

## Declaring which arguments are paths

Tool authors mark path arguments declaratively on `ToolInfo`:

```python
@tool_meta(ToolInfo(..., checkpaths=["path"]))
async def list_files(path: str, ...): ...
```

`register_tools` folds `checkpaths` into the tool's `meta` dict, which
travels to clients on the MCP Tool's `_meta` field.  Tools that read or
write the filesystem should also call `check_path_access` inside their
implementation (the author-side layer).  Self-contained helpers with their
own containment (e.g. `download_file_to_cache`, the sandboxed code
execution tools) are not marked: their boundary is their own cache/sandbox,
not the project root.

## Standardised tool call state

Both Klea Agent and Klea RAG use the shared `ToolCallSchema` /
`ToolCallsSchema` from `klea_utils.mcp.schemas` and the same state fields:

- `tool_calls: list[ToolCallSchema]` -- selected calls (written by the
  shared `ToolsPicker`, `klea_utils/nodes/tools_picker.py`).
- `tool_results: list[CallToolResult]` -- results (written by the shared
  `ToolsCallerNode`).

The shared picker/caller nodes are configured per app (prompt directory,
`model_type`); the agent additionally passes a `post_dispatch` callback to
mark its per-plan-step status.

## Network safety (SSRF) for outbound tools

Outbound HTTP tools (`web_fetch`, `download_file`) share an SSRF guard in
`klea_utils.mcp.tool_impls.ssrf` (`check_ssrf`, `is_private_or_reserved`):
requests to loopback, private, link-local, reserved, or multicast
addresses are refused unless the caller passes `allow_internal_hosts=True`.

Known best-effort limitation (accepted for now): the guard checks only the
*initial* URL.  An httpx client that follows redirects
(`follow_redirects=True`) could still be redirected onto an internal host
after the check.  If this ever needs hardening, follow redirects manually
and re-check each hop.

## The author-side limit

The in-tool check only protects tools we write.  A third-party MCP server
runs its own code with its own privileges; Klea cannot inspect or bound
the paths it touches.  For servers we do not author there is no way to
enforce path-level permissions from inside the tool.

## What opencode does (external MCP tools)

Reference: the opencode repository
(`/home/asinha/Documents/02_Code/01_others/opencode`, checked out around
2026-08).

opencode does *not* inspect tool arguments.  Instead it applies a
client-side, per-tool-call permission policy that works for any server:

- Every external MCP tool is wrapped so its execution first runs a
  permission request keyed on the tool name (`server_tool`), see
  `packages/opencode/src/session/tools.ts` (around the
  `ctx.ask({ permission: key, ... })` call).
- The ruleset matches permission rules (allow / deny / ask) against the
  tool name, with wildcards (`tool-server_*`).  The default action when no
  rule matches is `ask`: an interactive user prompt with "once" and
  "always" replies.  "always" is remembered as an allow-rule for the
  session.  See `packages/opencode/src/permission/index.ts`.
- MCP tools denied by config are hidden from the model entirely
  (visibility filtering), so the system prompt only advertises allowed
  tools.

Key consequence: opencode's gate is *"may this tool be invoked at all"*,
never *"may it touch path X"*.  The user's approval is only as informed as
the tool description and their own understanding of the server.  A user
who approves a filesystem tool has granted it access to whatever paths the
server process can reach, and "always" turns one careless approval into a
session-wide pass.  The prompt is a consent mechanism, not a path
confinement guarantee.

## Options for Klea

1. **In-tool path checks (implemented)** -- path-aware and stricter than
   opencode's name-based gate, but only for tools we author.  Keep this as
   the author-side layer.
2. **Client-side tool-call policy layer (partially implemented)** --
   opencode-style allow / deny / ask per tool (and, where the tool declares
   it, per path), evaluated at the call site before dispatching to the MCP
   server.  The *per-path* half is done: the shared `ToolsCallerNode` gates
   calls through `dispatch_tool_calls` + `check_tool_arguments_permissions`
   using each tool's `checkpaths` declaration, denying out-of-boundary paths
   before they reach the server.  The *allow / deny / ask* ruleset and the
   interactive user-approval loop (graph pause + TUI/web input, opencode
   style) are still deferred -- see the TODO in `permission.py` and the
   kanban board.  The client-side gate only applies to tools that declare
   `checkpaths`; third-party servers that do not are not path-gated.
3. **OS-level sandboxing** -- run third-party MCP servers (or the whole
   agent) in a container / bubblewrap / chroot with only the project
   directory mounted.  This is the only hard boundary for servers we do
   not author, and it is orthogonal to options 1 and 2.

Recommended posture: 1 + 2 together, document the trust model, and advise
sandboxing for third-party servers.  In all cases, connecting to a server
means trusting its author: never connect to a server you do not trust.
