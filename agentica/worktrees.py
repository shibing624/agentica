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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agentica.utils.log import logger

# Branch names for worktrees this module creates. Prefixed so `git branch`
# separates "a task someone is working in" from long-lived branches.
BRANCH_PREFIX = "wt/"

# Files git ignores but a working session needs. Symlinked (not copied) so a
# rotated key or a new variable reaches every worktree at once, and so a secret
# exists in one place on disk. Override with `worktree.link` in config.yaml.
LINKED_PATHS: Tuple[str, ...] = (".env",)

# Where worktrees are created. Unset means "next to the main checkout"
# (``../<repo>-<task>``), which is what a human types by hand and what they can
# cd into. Two situations need something else, and both are the user's to
# decide rather than ours to guess:
#
#   * a parent directory holding twenty repositories, where five worktrees each
#     is clutter — point this at one directory and get
#     ``<root>/<repo>/<task>``;
#   * a shared mount whose parent is not writable.
#
# A *relative* value resolves inside the main checkout (e.g.
# ``.agentica/worktrees``, which is what Claude Code does). That works, but know
# what it costs before choosing it: a worktree under an ignored directory is in
# range of ``git clean -xdff`` run in the main checkout, which deletes it —
# including whatever another session had not committed yet. Verified, not
# theorised: git's own dry-run reports the whole tree as removable. These
# worktrees are meant to outlive tasks, so that trade is off by default.
ROOT_SETTING = "worktree.root"
LINK_SETTING = "worktree.link"

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
    """Where the worktree for ``name`` lives.

    Default: a sibling of the main checkout, ``../<repo>-<task>``. With
    ``worktree.root`` set, ``<root>/<repo>/<task>`` — absolute anywhere on the
    machine, relative inside the main checkout (see ``ROOT_SETTING``).
    """
    root = Path(main_root(cwd))
    configured = _configured_root()
    if not configured:
        return str(root.parent / f"{root.name}-{slug(name)}")
    base = Path(os.path.expanduser(configured))
    if not base.is_absolute():
        # In-repo (``.agentica/worktrees``): the repository is already implied by
        # where the root lives, so ``<root>/<task>`` — inserting the repo name
        # again would read ``proj/.agentica/worktrees/proj/docs``.
        return str(root / base / slug(name))
    # One directory serving several repositories has to say which one.
    return str(base / root.name / slug(name))


def _configured_root() -> str:
    """``worktree.root`` from config.yaml, or "" when unset."""
    try:
        from agentica.global_config import get_setting

        return str(get_setting(ROOT_SETTING, "") or "").strip()
    except Exception:
        # Worktrees must keep working when config.yaml is missing or broken.
        return ""


# Nested-worktree lookups happen on every glob/grep, so they are cached for a
# few seconds per repository — creating a worktree is rare, searching is not.
NESTED_CACHE_TTL = 10.0
_nested_cache: dict = {}


def nested_worktrees(cwd: str, *, ttl: float = NESTED_CACHE_TTL) -> Tuple[str, ...]:
    """Absolute paths of this repository's worktrees that live *inside* it.

    Search tools exclude these (``tools/builtin/file_tool.py``). A worktree under
    the checkout is a second full copy of the project, so ``glob("**/*.py")``
    otherwise returns every file once per worktree — and the hazard is not the
    noise, it is an edit landing in the copy. Observed live in this repository:
    ``glob("**/peers.py")`` returned ``.worktrees/wechat-media/agentica/peers.py``
    next to the real one.

    Asked of git rather than derived from ``worktree.root``, because a nested
    worktree does not have to be one of ours: a person or another agent typing
    ``git worktree add .worktrees/x`` creates exactly the same duplication, and a
    list of names we happen to know would silently miss it.
    """
    now = time.time()
    key = os.path.realpath(cwd)
    cached = _nested_cache.get(key)
    if cached is not None and now - cached[0] < ttl:
        return cached[1]

    try:
        main = os.path.realpath(main_root(cwd))
        inside = tuple(
            os.path.realpath(entry.path)
            for entry in list_worktrees(cwd)
            if not entry.is_main
            and os.path.realpath(entry.path).startswith(main + os.sep)
        )
    except (WorktreeError, OSError):
        # Not a repository, no git, a locked index: nothing to exclude.
        inside = ()
    _nested_cache[key] = (now, inside)
    return inside


def configured_links() -> Tuple[str, ...]:
    """``worktree.link`` from config.yaml, or the default (``.env``)."""
    try:
        from agentica.global_config import get_setting

        configured = get_setting(LINK_SETTING, None)
    except Exception:
        configured = None
    if isinstance(configured, str):
        configured = [part.strip() for part in configured.split(",")]
    if isinstance(configured, (list, tuple)):
        names = tuple(str(item).strip() for item in configured if str(item).strip())
        if names:
            return names
    return LINKED_PATHS


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


def link_ignored(src_root: str, dst_root: str, names: Optional[Sequence[str]] = None) -> List[str]:
    """Symlink gitignored-but-needed files from the main checkout.

    Skips what is absent in the source or already present in the target, so it
    is safe to run again on an existing worktree.
    """
    linked: List[str] = []
    for name in names if names is not None else configured_links():
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

def _self_ignore(parent: Path, repo_root: str) -> None:
    """Make a worktree root inside the repository ignore itself.

    Only when it *is* inside: a root under the checkout (``.agentica/worktrees``,
    the shape Claude Code uses) would otherwise show up as untracked in every
    ``git status`` until someone edits the repository's ``.gitignore`` — and that
    edit is a change to a tracked, shared file, made by a tool, in someone
    else's repository. A ``.gitignore`` containing ``*`` *inside* the root ignores
    the tree and itself, touches nothing tracked, and needs no per-repo setup
    (the same trick pip uses for its caches).

    Note what this does not fix: an ignored tree is in range of
    ``git clean -xdff`` run in the main checkout. Single ``-f`` skips nested
    checkouts ("Skipping repository"), double ``-ff`` removes them along with
    whatever another session had not committed. That is the price of the
    in-repo layout, and it is why it is not the default.
    """
    try:
        # realpath both sides: /tmp is a symlink to /private/tmp on macOS, and a
        # string comparison would then decide the root is outside the repo.
        resolved_parent = Path(os.path.realpath(parent))
        resolved_repo = os.path.realpath(repo_root)
        inside = resolved_repo in (
            str(resolved_parent), *(str(p) for p in resolved_parent.parents)
        )
        if not inside:
            return
        marker = parent / ".gitignore"
        if not marker.exists():
            marker.write_text("# Created by agentica: worktrees live here, git ignores them.\n*\n")
    except OSError:
        # Not worth failing a worktree over; the user can add the ignore line.
        logger.debug("could not write %s/.gitignore", parent, exc_info=True)



def ensure(
    cwd: str,
    name: str,
    *,
    base: Optional[str] = None,
    link: Optional[Sequence[str]] = None,
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
    parent = Path(path).parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise WorktreeError(
            f"cannot create {parent} ({e}); set `{ROOT_SETTING}` in "
            "~/.agentica/config.yaml to a directory you can write to"
        ) from e
    if not os.access(parent, os.W_OK):
        raise WorktreeError(
            f"{parent} is not writable; set `{ROOT_SETTING}` in "
            "~/.agentica/config.yaml to a directory you can write to"
        )
    _self_ignore(parent, main_root(cwd))

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


# How long to keep retrying an operation in the main checkout that lost a race
# for git's own index.lock. Two sessions merging at the same time is the case;
# git's lock is already the mutex, so nothing here invents a second one.
LOCK_RETRIES = 5
LOCK_WAIT = 1.0


def _git_patient(args: Sequence[str], cwd: str, *, check: bool = True) -> str:
    """Run a git command, waiting out another process holding the index lock."""
    last: Optional[WorktreeError] = None
    for attempt in range(LOCK_RETRIES):
        try:
            return _git(args, cwd, check=check)
        except WorktreeError as e:
            if "index.lock" not in str(e) and "another git process" not in str(e).lower():
                raise
            last = e
            time.sleep(LOCK_WAIT * (attempt + 1))
    raise WorktreeError(
        f"{last} — another session has been holding git's lock for "
        f"{int(LOCK_RETRIES * LOCK_WAIT)}s; try again in a moment"
    )


@dataclass(frozen=True)
class MergeResult:
    """What ``merge_back`` did, in the caller's words."""

    branch: str
    base: str
    commits: int
    merged_sha: str
    conflicted_files: Tuple[str, ...] = ()

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicted_files)


def merge_back(cwd: str, *, base: Optional[str] = None) -> MergeResult:
    """Land this worktree's branch on the base branch, keeping the worktree.

    The order is what makes it safe, and it is not the obvious one:

    1. **Base into the branch, inside the worktree.** A conflict then belongs to
       the session that wrote the code, in the directory it was written in, with
       its tests one command away — instead of stranding a half-merged index in
       the main checkout that every other session shares.
    2. **Branch into base, in the main checkout.** After step 1 this is a
       fast-forward, so the shared checkout is touched for as little as possible.
       Git's own index lock is the mutex against another session doing the same
       thing; ``_git_patient`` waits it out rather than adding a second lock.

    A worktree left behind at an old base is the thing that makes these go stale,
    so step 1 is not an optimisation: it is why the worktree is still usable
    afterwards (it ends level with the base). Nothing is ever deleted.

    Refuses — rather than guessing — when there is uncommitted work on either
    side, or when the main checkout is not on the base branch.
    """
    root = current_root(cwd)
    main = main_root(cwd)
    base_ref = base or default_base(cwd)

    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).strip()
    if branch == "HEAD":
        raise WorktreeError("this worktree is on a detached HEAD; check out a branch first")
    if branch == base_ref:
        raise WorktreeError(
            f"this worktree is already on {base_ref}; there is nothing to merge back"
        )
    if os.path.realpath(root) == os.path.realpath(main):
        raise WorktreeError(
            "this is the main checkout, not a worktree — switch to a worktree first"
        )
    if _git(["status", "--porcelain"], root).strip():
        raise WorktreeError(
            "commit (or stash) this worktree's changes first — merging would "
            "otherwise land a half-finished state on " + base_ref
        )

    ahead = _git(["rev-list", "--count", f"{base_ref}..{branch}"], root).strip()
    commits = int(ahead) if ahead.isdigit() else 0
    if commits == 0:
        raise WorktreeError(
            f"{branch} has no commits that {base_ref} does not already have"
        )

    # 1. Base into the branch, here, where a conflict can be resolved.
    merge = _git(["merge", "--no-edit", base_ref], root, check=False)
    conflicted = tuple(
        line.strip()
        for line in _git(["diff", "--name-only", "--diff-filter=U"], root).splitlines()
        if line.strip()
    )
    if conflicted:
        return MergeResult(
            branch=branch, base=base_ref, commits=commits, merged_sha="",
            conflicted_files=conflicted,
        )
    if "CONFLICT" in merge:
        raise WorktreeError(f"merging {base_ref} into {branch} failed: {merge.strip()[-200:]}")

    # 2. Branch into base, in the main checkout — a fast-forward after step 1.
    main_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], main).strip()
    if main_branch != base_ref:
        raise WorktreeError(
            f"the main checkout is on '{main_branch}', not '{base_ref}'; "
            f"leave it on {base_ref} so merges land where everyone reads them"
        )
    if _git(["status", "--porcelain"], main).strip():
        raise WorktreeError(
            f"the main checkout has uncommitted changes; {base_ref} must be clean "
            "before anything is merged into it"
        )
    _git_patient(["merge", "--ff-only", "--no-edit", branch], main)
    merged_sha = _git(["rev-parse", "--short", "HEAD"], main).strip()

    return MergeResult(branch=branch, base=base_ref, commits=commits, merged_sha=merged_sha)
