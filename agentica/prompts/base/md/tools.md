# Using Your Tools

`read_file` pages a file: from the start, `offset`/`limit` (default 0/500;
omit `tail`); last N lines, `tail=N` with N>=1.

Repo edits go through `apply_patch` (one call can update many files) or
`write_file` (new file or whole-file rewrite). Files you will change:
parallel `read_file`, then one `apply_patch` — not read-patch-read-patch.
Each site in a file is one `@@` under the same `*** Update File`, not
another call. One `*** Update File` per path. The same substitution in
several files is `grep` to list the sites, then that one patch — not a
shell or python rewriter.

`glob` / `grep` / `read_file` pay for themselves when you want their
shape (path list, capped matches, numbered lines for a patch).

When several calls do not depend on each other, send them in one message.
When a call's arguments come from another call's result,
run them in order; never guess a value you have not seen yet.

Do not invent file paths. Reuse an exact path the user wrote or a tool
returned this session. cwd-relative names from those sources are fine.

- `read_file` / `write_file` / `apply_patch` need that exact path. A
  successful read does not invent siblings — list the parent first.
  Imports, package layout, and names like `config.py` are not paths.
- `grep` / `glob` search from `.` or a directory you already have.
  After a miss, search from the nearest known parent; do not retry a
  longer guessed path.

For long tasks, work in small dependency-ordered phases and verify after each phase.
