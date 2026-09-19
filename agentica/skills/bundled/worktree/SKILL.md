---
name: worktree
description: >-
  Isolate parallel work in a git worktree of the current repository
  (worktree tool: status / new / merge / remove). Use when another live
  session is dirty in the same directory (list_agents), when two agents would
  fight over index.lock, or when starting a parallel task checkout.
metadata:
  version: "2.1"
---

# Isolate work in a worktree

Several sessions in one checkout overwrite each other and contend for git's
index. A worktree is one directory + one branch per *task*, sharing the
repository.

This session does **not** move. `worktree(action="new")` returns a path.
Pass that path as `work_dir` on `read_file` / `write_file` / `apply_patch` /
`execute`. On `glob` / `grep`, pass it as `path`. Omit those arguments and
the call stays in this session's directory.

Do not `cd` and expect later calls to follow — each call is independent.
Do not `execute` `git worktree add`: that only makes a directory, and later
calls still use this session unless you pass `work_dir` / `path`. Use
`worktree(action="new")`.

## When

- `list_agents` shows another session in this directory (dirty files, same
  branch) and you are about to edit.
- A change should not land on the shared working tree.

Not this: a read-only `task` subagent (it does not write). A second CLI
that should start already inside a tree is `agentica --worktree <task>`.

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

For a chunk of work big enough for its own process, `delegate(task=...,
work_dir=<path>)` already takes the same path.

If a directory goes missing, `status` labels it `(directory gone)` or
`(not a checkout)`. Say so; do not rebuild it by destroying what sits there.
`remove` the name, then `new` if you still want it.
