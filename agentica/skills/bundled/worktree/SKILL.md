---
name: worktree
description: >-
  Isolate this CLI session in its own git worktree of the current repository
  (worktree tool: status / use / merge / remove). Use when another live
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
- `worktree(action="merge")` — land on local main, delete the checkout, return
- `worktree(action="remove")` — drop one with no unique work

The human's surfaces: `agentica --worktree <task>` at start (see
`agentica --help`), `/worktree` mid-session. You cannot type slash commands;
use the tool. Tell the user which slash to type when they want to drive.
