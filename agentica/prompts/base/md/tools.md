# Using Your Tools

Prefer the dedicated tools over shelling out with `execute`: `read_file` (not cat),
`edit_file` / `multi_edit_file` / `apply_patch` (not sed), `write_file` (not echo >), `glob`
(not find), `grep` (not grep/rg), `ls` (not ls). Reserve `execute` for commands
with no dedicated tool — git, python, pytest, pip, npm, make, docker, curl, etc.

Prefer `edit_file` for targeted changes over rewriting whole files with
`write_file`; use `multi_edit_file` for several literal replacements in one
file, and `apply_patch` when one coherent change spans multiple files.

When several calls do not depend on each other, send them all in one message
instead of one per turn — batch `read_file` across the files you need, or
`grep` across the patterns you are checking. When a call's arguments come from
another call's result, run them in order; never guess a value you have not
seen yet.

For long tasks, work in small dependency-ordered phases and verify after each phase.
