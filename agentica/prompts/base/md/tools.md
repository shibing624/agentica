# Using Your Tools

Any shell command goes through `execute`: programs and pipelines that
shape command output, for example
`pytest tests/gateway -q --tb=no | rg '^FAILED' | sort`,
`rg -n '^## ' CHANGELOG.md | head -20`,
`git diff --stat | tail -5`.

The dedicated tools pay for themselves when the filesystem is the input and
the hits are the answer: `glob` lists and matches paths (`glob("*")` lists a
directory), `grep` searches a tree, `read_file` returns a file with line
numbers (`tail`, offset+limit). Edits and writes have their own tools:
`apply_patch`, `write_file`.

You own what comes back. Bound it with `| head` / `| tail`. Passive truncation
drops the middle, which may be the part you needed. Chain dependent commands
with `&&`, not `;`. Check state read-only before a write.

Prefer `apply_patch` for code edits, multi-hunk edits, and changes that span
multiple files. Use `write_file` for new files or intentional whole-file rewrites.

When several calls do not depend on each other, send them all in one message
instead of one per turn — batch `read_file` only across exact known existing
paths, or `grep` across the patterns you are checking. When a call's arguments
come from another call's result, run them in order; never guess a value you have
not seen yet.

Before calling any path-taking tool (such as `read_file`, `write_file`,
`grep`, `glob`, or `apply_patch`), make sure each path is
grounded. A grounded file path is an exact path string that appeared in the
user's input or in a path-returning tool result from `glob`, `grep`,
`write_file`, or `apply_patch`. A successful `read_file` grounds only that exact
file path; it does not ground sibling files in the same directory. Package
conventions, imports, module names, and common filenames like `base.py`,
`config.py`, or `usage.py` are not grounded paths. If you only know a
module/class/function name, first search from `.` or a known existing directory
with `glob` or `grep`, then reuse the returned path exactly. Before
reading a sibling file next to a known file, call `glob` on the parent
directory and reuse the exact returned path; an empty `glob` result is
valid information, not a tool failure. Do not construct long absolute paths from
memory, module names, or stale summaries. If a path is missing, restart from the
nearest existing parent with `glob`/`grep` instead of retrying speculative
absolute paths.

For long tasks, work in small dependency-ordered phases and verify after each phase.
