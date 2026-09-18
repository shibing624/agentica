---
name: worktree
description: >-
  Isolate this CLI session in its own git worktree of the current repository
  (worktree tool: status / use / main / merge / remove). Use when another live
  session is dirty in the same directory (list_agents), when two agents would
  fight over index.lock, when starting a parallel task checkout, or when asked
  to switch into a named worktree.
metadata:
  version: "1.0"
---

# Isolate this session

Several sessions in one checkout overwrite each other and contend for git's
index. A worktree is one directory + one branch per *task*, sharing the
repository.

Do not create one with `execute` (`git worktree add`, `cd`). That only adds a
directory; this session's file tools stay in the old checkout. Call the
`worktree` tool — it moves process cwd, sandbox, peer record and the status bar
together. The transcript stays where it already was.

## When

- `list_agents` shows another session in this directory (dirty files, same
  branch) and you are about to edit.
- The user or a peer names a task checkout ("切到 gateway-peers").
- A change should not land on the shared working tree.

Not this: a read-only `task` subagent (it does not write). A second CLI the
human should watch is `multi-agent` — start *that* process with `--worktree` or
its own `-c`.

## How

Name the *task*, not yourself. Reused while the task is in progress; gone after
merge. The tool schema lists the actions; do not invent paths.

- `worktree(action="use", name="<task>")` — create or reuse, then move here
- `worktree(action="status")` — where you are, every worktree of this repo
- `worktree(action="main")` — return to the main checkout; the worktree stays
- `worktree(action="merge")` — land on local main (conflicts stay in this
  worktree, then fast-forward on main), delete the checkout, return
- `worktree(action="remove")` — drop a `wt/*` checkout git will allow (dirty
  or someone else's lock is refused; an unmerged branch is left in place)

Not `use(name="main")` when you mean "take me back to the main checkout":
`use` reads its name as a *task*, so that creates a second checkout on a
`wt/main` branch instead. `main` is the way to leave one unfinished. `merge`
and `remove` dispose of the worktree. A `cd` is not it: that moves the shell
only, not the file tools.

Do not delete a worktree with `execute` (`git worktree remove`, `rm -rf`) while
you are standing in it: your working directory disappears under you and every
later command fails on it, absolute paths included. The tool's `remove` and
`merge` move the session out before deleting, which is the reason to use them.

The human's surfaces: `agentica --worktree <task>` at start (see
`agentica --help`), `/worktree` mid-session. You cannot type slash commands;
use the tool. Tell the user which slash to type when they want to drive.
