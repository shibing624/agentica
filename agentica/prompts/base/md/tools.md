# Using Your Tools

Any shell command goes through `execute`. Split or combine as needed —
do not force every probe into one script. Independent calls in one
message can run together (`execute(parallel_safe=True)`). Dependent
steps can share one command (pipes and `&&`). A miss that must not
stop the rest: `;` and `2>/dev/null`.
`cd /abs/project && pytest -q --tb=no | rg '^FAILED' | sort && python -m build 2>&1 | tail -8`
`rg -n Foo -A 12 src/a.py | head -40; echo ===; rg -n Bar docs/note.md 2>/dev/null | head`

Search with `rg`; if `rg` is missing, `grep`
(`rg -n PAT -- path || grep -n PAT path`). Bound noisy output with
`| head` / `| tail`. Newlines stay, so a `python3 - <<'EOF'` … `EOF`
heredoc works — print or analyze only; do not write the tree from a
script.

Do not dump a source file through the shell (`cd … && cat f.py`). That
is `read_file`: from the start, `offset`/`limit` (default 0/500; omit
`tail`); last N lines, `tail=N` with N>=1. A persisted dump still spent
a turn filling a pipe.

Repo edits go through `apply_patch` (one call can update many files) or
`write_file` (new file or whole-file rewrite). The same substitution in
several files is `rg` to list the sites, then one `apply_patch` with
several `*** Update File` hunks — not a shell or python rewriter.
`execute` is for explore, analyze, verify, build, and git.

`glob` / `grep` / `read_file` pay for themselves when you want their
shape (path list, capped rg, numbered lines for a patch).

When several dedicated-tool calls do not depend on each other, send them
in one message. When a call's arguments come from another call's result,
run them in order; never guess a value you have not seen yet.

Do not invent file paths. Reuse an exact path the user wrote or a tool
returned this session. cwd-relative names from those sources are fine.

- `read_file` / `write_file` / `apply_patch` need that exact path. A
  successful read does not invent siblings — list the parent first.
  Imports, package layout, and names like `config.py` are not paths.
- `grep` / `glob` search from `.` or a directory you already have.
- `execute` may reuse a known path, search from `.` (`rg -n PAT`), or
  hedge-probe a candidate with `2>/dev/null`. Do not assemble a long
  absolute path from memory or a stale summary. After a miss, search
  from the nearest known parent; do not retry a longer guessed path.

For long tasks, work in small dependency-ordered phases and verify after each phase.
