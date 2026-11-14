# Surface parity for built-in skills and tools

Agentica has one built-in tool catalogue and one bundled skill catalogue per
process. CLI, Web, Desktop, IM and headless runs are **outlets**: they may expose
a different command/menu/card, but they must not change which skills exist or
omit a tool the shared skill text promises. A per-surface skill list would also
change the system prompt byte-for-byte and defeat prefix-cache reuse.

## Product tool construction

`agentica.tools.product.build_product_tools()` is the single construction point
for product-level tools. Interactive CLI, gateway (Web/Desktop/IM) and cron/CLI
runners call it with their surface-specific context:

| Surface | Outlet | Shared construction |
|---|---|---|
| Interactive CLI | `/worktree`, `--worktree`, model `worktree` tool | `build_product_tools(surface="cli", ...)` |
| Web / Desktop | `/worktree` panel, model `worktree` tool | `build_product_tools(surface="web", ...)` via gateway |
| IM | gateway-bound session and model `worktree` tool | `build_product_tools(surface="gateway", ...)` via gateway |
| Headless / cron | model `worktree` tool; no human slash outlet | `build_product_tools(surface="headless"/"cron", ...)` |

Every product surface gets `WorktreeTool`. Surface-specific callbacks (for
example entering a CLI worktree, or binding a gateway session's `work_dir`) are
passed into the shared binder; they are not a reason to register a different
tool set.

## Worktree

`agentica.git.binding.WorktreeBinder` is core-layer code. The old
`agentica.cli.worktree_binding` module re-exports it so existing CLI imports and
tests keep working, but new code imports from `agentica.git.binding`.

The `worktree` model tool has the same `status` / `new` / `merge` / `remove`
semantics everywhere: create or reuse a per-task checkout, return its path, do
not move the session's cwd, and pass that path as `work_dir` / `path` on later
tool calls. Gateway surfaces additionally expose `/worktree` and
`GET|POST /api/worktrees`; **Bind** is the outlet that changes a Web/Desktop/IM
session's `work_dir` and invalidates that session's cached agent. CLI has its
own human slash command and `agentica --worktree <task>` startup flag.

The bundled `worktree` skill is shared by all surfaces. It says what every
surface has, names the differences only as outlets, and forbids writing Python
to call `agentica.git.worktree.ensure` by hand.

## What not to do

- Do not load a different bundled skill set by surface.
- Do not teach a skill a tool that only one surface registers.
- Do not add an import from `agentica.cli` into core/gateway tool construction.
- Do not solve a missing product tool with an `execute` Python heredoc.
