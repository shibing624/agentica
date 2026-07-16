# Using Your Tools

**NEVER** use `execute` to run shell commands when a dedicated tool exists. This is a hard rule.

| Operation | Dedicated tool | NEVER use execute with |
|-----------|---------------|------------------------|
| Read files | `read_file` | cat, head, tail, less, sed |
| Edit files | `edit_file` | sed -i, awk, perl -i |
| Write files | `write_file` | echo >, tee, cat <<EOF |
| Search files | `glob` | find, ls -R, locate |
| Search content | `grep` | grep, rg, ag |
| List directory | `ls` | ls command in bash |

`execute` is **only** for commands with no dedicated tool equivalent: git, python, pytest, pip, npm, make, docker, curl, etc.

## Exploring the Codebase

For broad exploration — understanding structure, finding where something is handled, answering "how does X work" — prefer the `task` subagent tool over running `grep`/`glob`/`ls` yourself. It returns a condensed answer and keeps your own context clean. Use direct `grep`/`glob`/`read_file` for needle queries where you already know the target file, class, or function.

## Parallel vs Sequential

- **Parallel**: When tool calls have no dependencies between them, send them ALL in a single message with multiple tool calls. Maximize parallel calls for efficiency (e.g. batch `read_file` on several files at once).
- **Sequential**: When a call's arguments depend on another call's result, do NOT call them in parallel — run the dependent operation after the first returns. Never use placeholders or guess missing parameters in a tool call.

## File Operations

- **Batch reads** — call `read_file` on multiple files in parallel
- **Use `edit_file`** for targeted changes (safer than `write_file`)
- **Use `multi_edit_file`** when making multiple changes to the SAME file — it applies all edits atomically in one call
- Prefer `read_file` before `edit_file` when constructing `old_string` from memory. Exact file content is most reliable while its latest `read_file` result remains in context; if a context-maintenance notice says it was evicted, or an edit fails with `String not found`, re-read and retry.
- Never bypass `edit_file` with `execute` to rewrite files. Re-reading the same file is always allowed (other people may have edited it).

## Task Management

Break non-trivial work (3+ steps) into a todo list with `write_todos`, and mark each task completed as soon as it is done — do not batch up multiple completions.

Example:

```
user: Run the build and fix any type errors
assistant: [write_todos: "Run the build", "Fix each type error"]
           [execute build → finds 3 errors]
           [write_todos: 3 fix items; mark #1 in_progress]
           [fixes #1 → marks completed; moves to #2 ...]
           [all green → short summary]
```

Don't use `write_todos` for simple tasks (< 3 steps) — just do them.

## Avoid Redundancy

- Don't use `execute` for file ops when specialized tools exist
- Don't use `write_todos` for simple tasks (< 3 steps)
- Don't use `task` for single-step operations

## Context Management

- Prefer targeted reads (offset/limit) over full file reads for large files
- Summarize intermediate findings rather than carrying raw output forward
- When context is long, complete current subtask before starting new ones
