---
name: worktree
description: >-
  Isolate parallel work in a git worktree of the current repository on every
  Agentica surface (worktree tool: status / new / merge / remove). Use when
  another live session is dirty in the same directory, when two agents would
  fight over index.lock, or when starting a parallel task checkout.
metadata:
  version: "2.3"
---

# Isolate work in a worktree

Several sessions in one checkout overwrite each other and contend for git's
index. A worktree is one directory + one branch per *task*, sharing the
repository. CLI, Web, Desktop, IM and headless runs all load this skill and
all have the `worktree` tool; the surface is only the outlet, not a different
capability set.

Use `worktree(action="new")` to create or reuse a checkout. It returns a path.
Pass that path as `work_dir` on `read_file` / `write_file` / `apply_patch` /
`execute`; on `glob` / `grep`, pass it as `path`. Omit those arguments and the
call stays in this session's directory. The agent's own cwd does not move.

In Web / Desktop / IM, `/worktree` opens the worktree picker. It calls the
same backend operation as the model tool; selecting **Bind** also switches the
session's working directory and rebuilds that session's agent. In the CLI,
`/worktree status|new|merge|remove` is the human-facing command; a new CLI
process that should start inside a tree is `agentica --worktree <task>`.

Do not `cd` and expect later tool calls to follow — each call is independent.
Do not call `git worktree add` through `execute`.
Do not write Python inside a product session to import
`agentica.git.worktree.ensure`: that only makes a
directory, bypasses session binding and leaves later calls in the wrong place.
Use `worktree` or the matching `/worktree` outlet.

Python SDK callers may use `agentica.git.worktree.ensure` directly when their
host process owns the working directory, or mount `WorktreeTool` explicitly on
`Agent` / `DeepAgent`. Bundled product skills are not loaded by SDK callers;
this paragraph is a documented escape hatch, not an instruction for an agent
running in CLI / Web / Desktop / IM / headless / cron.

## When

- Another live session is working in this directory (dirty files, same branch)
  and you are about to edit.
- A change should not land on the shared working tree.
- Web / Desktop / IM development needs a branch-isolated session.

Not this: a read-only `task` subagent (it does not write).

## How

Name the *task*, not yourself. Reused while the checkout is healthy; gone
after merge. A registration that is no longer a checkout is refused — remove
it first, or pick another name. The tool does not delete leftover files to
make room.

- `worktree(action="new", name="<task>")` — create or reuse; returns the path
- `worktree(action="status")` — every worktree of this repo
- `worktree(action="merge", name="<task>")` — land on local main (conflicts
  stay in that worktree, then fast-forward on main), delete the checkout
- `worktree(action="remove", name="<task>")` — drop a `wt/*` checkout git
  will allow (dirty is refused; an unmerged branch is left in place)

`new(name="main")` is refused: `main` is the repository root, not a task.

If a directory goes missing, `status` labels it `(directory gone)` or
`(not a checkout)`. Say so; do not rebuild it by destroying what sits there.
`remove` the name, then `new` if you still want it.
