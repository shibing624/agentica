# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Per-task git worktrees, so several sessions can work one repo at once.

Two agentica sessions in the same checkout edit the same files, fight over
``index.lock``, and have to *ask each other* what changed. A worktree per task
removes the cause instead of coordinating around it: separate directory,
separate branch, one shared ``.git``.

This module is the git part only — resolve, create, reuse, list, merge back.
Binding a session to one is the CLI's business (``cli/commands/worktree_cmd.py``
and the ``worktree`` tool), because that is where a work_dir lives.

Decisions that are not obvious:

**Never delete anything.** ``ensure()`` creates or *reuses*; merging back leaves
the worktree in place with its branch reset onto the base. A worktree costs a
directory and buys a warm IDE index, an installed venv and a shell history; the
user asked for them to persist across weeks, so nothing here removes one. That
is also why an existing directory that is *not* a registered worktree is an
error rather than something to clear out of the way.

**Paths hang off the main worktree, never the current one.** ``git worktree
add`` works from inside any worktree, so a session that is already in
``agentica-gateway`` would otherwise create ``agentica-gateway-docs``. The main
checkout is found via ``--git-common-dir`` and every path is derived from it, so
the layout stays flat no matter where the command runs.

**gitignored files do not travel.** A fresh worktree has no ``.env`` — the
symptom is a session that starts and then cannot reach any model. Those files
are symlinked from the main checkout on creation, so one edit stays one edit.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Branch names for worktrees this module creates. Prefixed so `git branch`
# separates "a task someone is working in" from long-lived branches.
BRANCH_PREFIX = "wt/"

# Files git ignores but a working session needs. Symlinked (not copied) so a
# rotated key or a new variable reaches every worktree at once.
LINKED_PATHS: Tuple[str, ...] = (".env",)

DEFAULT_BRANCHES = ("main", "master")

TIMEOUT = 60.0

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


class WorktreeError(RuntimeError):
    """A worktree operation failed for a reason the caller should show a human."""


@dataclass(frozen=True)
class Worktree:
    """One worktree of the repository, as git reports it."""

    name: str
    path: str
    branch: str
    head: str = ""
    is_main: bool = False
    # True when this record describes something already on disk.
    exists: bool = True
    # Absolute paths that were symlinked in at creation time.
    linked: Tuple[str, ...] = ()

    @property
    def branch_short(self) -> str:
        return self.branch.removeprefix("refs/heads/")

    def describe(self) -> str:
        role = " (main)" if self.is_main else ""
        return f"{self.name}{role} — {self.branch_short} — {self.path}"


def slug(name: str) -> str:
    """Normalise a task name into something usable as a directory and a branch."""
    cleaned = _SLUG_RE.sub("-", (name or "").strip().casefold()).strip("-._")
    if not cleaned:
        raise WorktreeError("a worktree name must contain at least one letter or digit")
    return cleaned


def _git(args: Sequence[str], cwd: str, *, check: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except FileNotFoundError as e:
        raise WorktreeError("git is not installed or not on PATH") from e
    except subprocess.SubprocessError as e:
        raise WorktreeError(f"git {' '.join(args)} did not finish: {e}") from e
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise WorktreeError(
            f"git {' '.join(args)} failed: {detail[-1] if detail else 'unknown error'}"
        )
    return result.stdout


def is_git_repo(cwd: str) -> bool:
    try:
        return _git(["rev-parse", "--is-inside-work-tree"], cwd).strip() == "true"
    except WorktreeError:
        return False


def main_root(cwd: str) -> str:
    """The main checkout's root, even when called from inside a worktree.

    ``--git-common-dir`` is the shared ``.git`` of the repository; its parent is
    the main worktree. Deriving paths from here is what keeps a worktree of a
    worktree from ever happening.
    """
    common = _git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd).strip()
    if not common:
        raise WorktreeError(f"{cwd} is not inside a git repository")
    return str(Path(common).parent)


def current_root(cwd: str) -> str:
    """The root of the worktree ``cwd`` is in."""
    return _git(["rev-parse", "--show-toplevel"], cwd).strip()


def default_base(cwd: str) -> str:
    """The branch new worktrees fork from: local ``main``, else ``master``."""
    for candidate in DEFAULT_BRANCHES:
        out = _git(["rev-parse", "--verify", "--quiet", candidate], cwd, check=False)
        if out.strip():
            return candidate
    raise WorktreeError(
        "no local 'main' or 'master' branch to fork from — pass an explicit base"
    )


def worktree_path(cwd: str, name: str) -> str:
    """Where the worktree for ``name`` lives: a sibling of the main checkout."""
    root = Path(main_root(cwd))
    return str(root.parent / f"{root.name}-{slug(name)}")


def branch_for(name: str) -> str:
    return f"{BRANCH_PREFIX}{slug(name)}"


def list_worktrees(cwd: str) -> List[Worktree]:
    """Every worktree of this repository, main first."""
    out = _git(["worktree", "list", "--porcelain"], cwd)
    main = main_root(cwd)
    entries: List[Worktree] = []
    path = head = branch = ""
    detached = False

    def flush() -> None:
        nonlocal path, head, branch, detached
        if path:
            resolved = os.path.realpath(path)
            is_main = resolved == os.path.realpath(main)
            entries.append(Worktree(
                name=Path(path).name,
                path=path,
                branch="" if detached else branch,
                head=head,
                is_main=is_main,
            ))
        path = head = branch = ""
        detached = False

    for line in out.splitlines():
        if line.startswith("worktree "):
            flush()
            path = line[len("worktree "):].strip()
        elif line.startswith("HEAD "):
            head = line[len("HEAD "):].strip()[:9]
        elif line.startswith("branch "):
            branch = line[len("branch "):].strip()
        elif line.strip() == "detached":
            detached = True
    flush()
    entries.sort(key=lambda w: (not w.is_main, w.name))
    return entries


def find(cwd: str, name: str) -> Optional[Worktree]:
    """The existing worktree for ``name``, matched by path or by branch."""
    want_path = os.path.realpath(worktree_path(cwd, name))
    want_branch = branch_for(name)
    for entry in list_worktrees(cwd):
        if os.path.realpath(entry.path) == want_path:
            return entry
        if entry.branch_short == want_branch:
            return entry
    return None


def link_ignored(src_root: str, dst_root: str, names: Sequence[str] = LINKED_PATHS) -> List[str]:
    """Symlink gitignored-but-needed files from the main checkout.

    Skips what is absent in the source or already present in the target, so it
    is safe to run again on an existing worktree.
    """
    linked: List[str] = []
    for name in names:
        src = Path(src_root) / name
        dst = Path(dst_root) / name
        if not src.exists() or dst.exists() or dst.is_symlink():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)
        except OSError:
            # A filesystem without symlinks is a reason to skip this file, not
            # to fail the whole worktree.
            continue
        linked.append(str(dst))
    return linked


def ensure(
    cwd: str,
    name: str,
    *,
    base: Optional[str] = None,
    link: Sequence[str] = LINKED_PATHS,
) -> Worktree:
    """Return the worktree for ``name``, creating it only if it does not exist.

    Idempotent on purpose: "bind me to <task>" is a thing a long-running session
    may say more than once, and the second time must land in the same directory
    with the same branch and the same history.
    """
    if not is_git_repo(cwd):
        raise WorktreeError(f"{cwd} is not inside a git repository")

    existing = find(cwd, name)
    if existing is not None:
        # Reuse: also re-link, so a file added to LINKED_PATHS since creation
        # (or removed by hand) reappears.
        linked = link_ignored(main_root(cwd), existing.path, link)
        return Worktree(**{**existing.__dict__, "linked": tuple(linked)})

    path = worktree_path(cwd, name)
    branch = branch_for(name)
    if Path(path).exists():
        raise WorktreeError(
            f"{path} already exists but is not a worktree of this repository; "
            "move it aside or pick another name"
        )

    base_ref = base or default_base(cwd)
    if _git(["rev-parse", "--verify", "--quiet", branch], cwd, check=False).strip():
        # The branch outlived its worktree (someone pruned the directory).
        # Check it out again rather than refusing or renaming.
        _git(["worktree", "add", path, branch], cwd)
    else:
        _git(["worktree", "add", "-b", branch, path, base_ref], cwd)

    linked = link_ignored(main_root(cwd), path, link)
    created = find(cwd, name)
    if created is None:
        raise WorktreeError(f"git created {path} but does not list it as a worktree")
    return Worktree(**{**created.__dict__, "linked": tuple(linked)})


def status_lines(cwd: str) -> List[str]:
    """Human-readable listing, main checkout first."""
    return [entry.describe() for entry in list_worktrees(cwd)]
