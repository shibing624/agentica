# -*- coding: utf-8 -*-
"""Commit (and push) a turn's changes from the review tab.

Only the files the caller names are committed: the turn's edits, not whatever
else is dirty in the tree (another session, the user's own work). Hooks run as
usual; nothing here force-pushes or skips verification.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List


class GitError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


_TIMEOUT_S = 120
# The diff fed to the message writer: enough to describe a turn, small enough
# for a cheap model.
_DIFF_CAP = 24_000


def _git(cwd: Path, *args: str, timeout: int = _TIMEOUT_S, check: bool = True) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError as e:
        raise GitError(500, "git is not installed") from e
    except subprocess.TimeoutExpired as e:
        raise GitError(504, f"git {args[0]} timed out") from e
    if check and proc.returncode != 0:
        raise GitError(409, (proc.stderr or proc.stdout).strip() or f"git {args[0]} failed")
    return proc


def repo_root(cwd: str) -> Path:
    path = Path(cwd).expanduser()
    if not path.is_dir():
        raise GitError(404, "directory not found")
    proc = _git(path, "rev-parse", "--show-toplevel", check=False)
    if proc.returncode != 0:
        raise GitError(404, "not a git repository")
    return Path(proc.stdout.strip())


def _repo_paths(root: Path, files: List[str]) -> List[str]:
    """Absolute or repo-relative paths -> repo-relative, all inside the repo."""
    out: List[str] = []
    for raw in files:
        p = Path(raw).expanduser()
        p = p if p.is_absolute() else root / p
        try:
            rel = p.resolve().relative_to(root.resolve())
        except ValueError as e:
            raise GitError(400, f"{raw} is outside the repository") from e
        out.append(rel.as_posix())
    return out


def status(cwd: str, files: List[str]) -> Dict:
    """Branch, upstream and which of ``files`` have something to commit."""
    root = repo_root(cwd)
    rels = _repo_paths(root, files) if files else []
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD", check=False).stdout.strip()
    upstream_proc = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False)
    upstream = upstream_proc.stdout.strip() if upstream_proc.returncode == 0 else ""
    ahead = behind = 0
    if upstream:
        counts = _git(root, "rev-list", "--left-right", "--count", f"{upstream}...HEAD", check=False).stdout.split()
        if len(counts) == 2:
            behind, ahead = int(counts[0]), int(counts[1])
    changed: List[str] = []
    ignored: List[str] = []
    if rels:
        porcelain = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *rels).stdout
        for item in porcelain.split("\0"):
            if len(item) > 3:
                changed.append(item[3:])
        # A .gitignore'd file can never be committed from here; say so rather
        # than showing it as merely unchanged.
        ignored = _git(root, "check-ignore", "--", *rels, check=False).stdout.split("\n")
        ignored = [p for p in ignored if p]
    return {
        "root": str(root),
        "branch": "" if branch == "HEAD" else branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "changed": changed,
        "ignored": ignored,
    }


def diff_for_message(cwd: str, files: List[str]) -> str:
    """The diff a commit message is written from: tracked changes plus new files."""
    root = repo_root(cwd)
    rels = _repo_paths(root, files)
    diff = _git(root, "diff", "HEAD", "--", *rels, check=False).stdout
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *rels).stdout.split()
    for rel in untracked:
        try:
            body = (root / rel).read_text(encoding="utf-8", errors="replace")[:4000]
        except OSError:
            continue
        diff += f"\nnew file {rel}\n{body}\n"
    return diff[:_DIFF_CAP]


def commit(cwd: str, files: List[str], message: str, push: bool = False) -> Dict:
    message = message.strip()
    if not message:
        raise GitError(400, "commit message is empty")
    if not files:
        raise GitError(400, "no files selected")
    root = repo_root(cwd)
    rels = _repo_paths(root, files)
    # -A stages deletions too; the pathspec keeps it to the chosen files.
    _git(root, "add", "-A", "--", *rels)
    staged = _git(root, "diff", "--cached", "--name-only", "--", *rels).stdout.split()
    if not staged:
        raise GitError(409, "nothing to commit in the selected files")
    # Commit only these paths, even if something else was already staged.
    _git(root, "commit", "-m", message, "--", *rels)
    sha = _git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    result: Dict = {"sha": sha, "files": staged, "pushed": False}
    if push:
        result.update(_push(root))
    return result


def _push(root: Path) -> Dict:
    upstream = _git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False)
    if upstream.returncode == 0:
        proc = _git(root, "push", check=False, timeout=300)
    else:
        branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if branch == "HEAD":
            raise GitError(409, "detached HEAD: nothing to push")
        proc = _git(root, "push", "-u", "origin", branch, check=False, timeout=300)
    if proc.returncode != 0:
        # The commit stands; only the push failed. Report it as such.
        return {"pushed": False, "pushError": (proc.stderr or proc.stdout).strip()}
    return {"pushed": True}


COMMIT_MESSAGE_PROMPT = """Write a git commit message for the diff below.
- One subject line under 72 characters, imperative mood, no trailing period.
- If the change has more than one part, add a blank line and a few short bullet lines.
- Match the language of the existing history samples when given; otherwise English.
Reply with ONLY the message: no quotes, no code fences, no preamble."""


def recent_subjects(cwd: str, n: int = 8) -> List[str]:
    try:
        root = repo_root(cwd)
    except GitError:
        return []
    out = _git(root, "log", f"-{n}", "--pretty=%s", check=False).stdout
    return [line for line in out.splitlines() if line.strip()]

