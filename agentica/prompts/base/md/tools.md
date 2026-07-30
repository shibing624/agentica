# Using Your Tools

Prefer the dedicated tools over shelling out with `execute`: `read_file` (not cat),
`edit_file` / `multi_edit_file` / `apply_patch` (not sed), `write_file` (not echo >), `glob`
(not find), `grep` (not grep/rg), `ls` (not ls). Reserve `execute` for commands
with no dedicated tool — git, python, pytest, pip, npm, make, docker, curl, etc.

Prefer `edit_file` for targeted changes over rewriting whole files with
`write_file`; use `multi_edit_file` for several literal replacements in one
file, and `apply_patch` when one coherent change spans multiple files.

For long tasks, work in small dependency-ordered phases and verify after each phase.
