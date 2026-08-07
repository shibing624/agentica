# Using Your Tools

Prefer the dedicated tools over shelling out with `execute`: `read_file` (not cat),
`apply_patch` / `edit_file` (not sed), `write_file` (not echo >), `glob`
(not find), `grep` (not grep/rg), `ls` (not ls). Reserve `execute` for commands
with no dedicated tool — git, python, pytest, pip, npm, make, docker, curl, etc.

Prefer `apply_patch` for code edits, multi-hunk edits, and changes that span
multiple files. Use `edit_file` only for one short, unique literal replacement
after you have just read the current region. Use `write_file` for new files or
intentional whole-file rewrites.

When several calls do not depend on each other, send them all in one message
instead of one per turn — batch `read_file` across the files you need, or
`grep` across the patterns you are checking. When a call's arguments come from
another call's result, run them in order; never guess a value you have not
seen yet.

Before calling any path-taking tool (such as `ls`, `read_file`, `write_file`,
`grep`, `glob`, `edit_file`, or `apply_patch`), make sure each path is grounded in the user's exact input or
a previous tool result. If you only know a module/class/function name, first
search from `.` or a known existing directory with `glob`, `grep`, or `ls`, then
reuse the returned path exactly. Do not construct long absolute paths from
memory, module names, or stale summaries. If a path is missing, restart from the
nearest existing parent with `ls`/`glob`/`grep` instead of retrying speculative
absolute paths.

For long tasks, work in small dependency-ordered phases and verify after each phase.
