# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Per-task git worktrees, so several sessions can work one repo at once.

Two agentica sessions in the same checkout edit the same files, fight over
``index.lock``, and have to *ask each other* what changed. A worktree per task
removes the cause instead of coordinating around it: separate directory,
separate branch, one shared ``.git``.

This module is the git part only — resolve, create, reuse, list, merge back.
A worktree is a directory other tools can be pointed at (``work_dir=`` /
``path=``). The session that created it stays where it is.

Decisions that are not obvious:

**A worktree is a feature checkout, not the session's home.** ``ensure()``
creates or reuses an *in-progress* directory. ``merge_back()`` lands the
branch on the local base; ``remove()`` deletes the checkout. We refuse only
what we own (``wt/*``, not the main tree, not Claude Code / detached). Dirty
trees are git's to refuse; unique commits stay on the branch because
``git branch -d`` will not delete it.

A registration that is no longer a checkout is reported, not destroyed.
``remove`` the name first, or pick another.

**Default path is inside the repository.** ``<repo>/.agentica/worktrees/<task>``,
the Claude Code shape. Sibling ``../<repo>-<task>`` is ``worktree.root: sibling``
for machines that want it. An existing directory that is *not* a registered
worktree is still an error rather than something to clear out of the way.

**Paths hang off the main worktree, never the current one.** ``git worktree
add`` works from inside any worktree, so a session that is already in
``.agentica/worktrees/gateway`` would otherwise nest another copy. The main
checkout is found via ``--git-common-dir`` and every path is derived from it.

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

# Where worktrees are created. Default is Claude Code's shape, inside the
# repository: ``<repo>/.agentica/worktrees/<task>``. ``git clean -xdf`` (one
# ``-f``) skips the nested checkout; ``git clean -xdff`` will remove it.
# Opt out with:
#
#   * ``worktree.root: sibling`` → ``../<repo>-<task>`` (the old default);
#   * an absolute path → ``<root>/<repo>/<task>`` (one farm, many repos);
#   * any other relative path → ``<repo>/<that>/<task>``.
DEFAULT_ROOT = ".agentica/worktrees"
SIBLING_ROOT = "sibling"
ROOT_SETTING = "worktree.root"
LINK_SETTING = "worktree.link"

DEFAULT_BRANCHES = ("main", "master")

# The base branch's own name, which ``use`` would otherwise read as a task and
# give ``wt/main`` — a second checkout on a branch named after the base. Nobody
# means that; "take me to the main checkout" is the ``main`` action. Refused
# rather than translated, because a silent translation would move a caller who
# believed they were creating a task. Only these two: the collision is with
# *branch* names, and every other word is a legitimate task name.
RESERVED_NAMES = tuple(DEFAULT_BRANCHES)

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
    # False when the path is not the root of a checkout: gone (``rm -rf``, a
    # foreign ``git worktree remove``), or resurrected as a bare directory by a
    # relative write. ``isdir`` is not enough — a mkdir inside the default
    # in-repo layout still sits in the main checkout. Named in ``describe`` so
    # a listing never offers a path that cannot be entered.
    exists: bool = True
    # Absolute paths that were symlinked in at creation time.
    linked: Tuple[str, ...] = ()
    locked: bool = False
    lock_reason: str = ""

    @property
    def branch_short(self) -> str:
        return self.branch.removeprefix("refs/heads/")

    def describe(self) -> str:
        role = " (main)" if self.is_main else ""
        if self.exists:
            missing = ""
        elif os.path.isdir(self.path):
            missing = " (not a checkout)"
        else:
            missing = " (directory gone)"
        return f"{self.name}{role} — {self.branch_short} — {self.path}{missing}"


def slug(name: str) -> str:
    """Normalise a task name into something usable as a directory and a branch."""
    cleaned = _SLUG_RE.sub("-", (name or "").strip().casefold()).strip("-._")
    if not cleaned:
        raise WorktreeError("a worktree name must contain at least one letter or digit")
    if cleaned in RESERVED_NAMES:
        raise WorktreeError(
            f"'{cleaned}' names the main checkout, not a task. Stay in the "
            "repository root to work there. Pick a name for the *task* otherwise."
        )
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
        # Two different absences reach here as ENOENT: the git binary, and
        # ``cwd`` itself. Blaming git for a deleted directory sends the reader
        # to install a binary they already have, and hides the one fact that
        # matters — this session is standing in a directory that is gone, which
        # is exactly the state ``status`` is asked about after another session
        # removes the worktree.
        if not Path(cwd).is_dir():
            raise WorktreeError(
                f"the directory this session works in no longer exists: {cwd} "
                "— git cannot run anywhere. Move to a directory that exists "
                "(the main checkout, or a new worktree) and retry."
            ) from e
        raise WorktreeError("git is not installed or not on PATH") from e
    except subprocess.SubprocessError as e:
        raise WorktreeError(f"git {' '.join(args)} did not finish: {e}") from e
    if check and result.returncode != 0:
        # Keep every line git wrote: taking only the last one dropped the
        # diagnosis and kept the hint. Git's wording is the explanation the
        # caller gets, verbatim — including the ``--force`` it advertises.
        # Rewriting that reads as censorship and only works in English; the
        # model can run the same command through ``execute`` anyway.
        raise WorktreeError(
            f"git {' '.join(args)} failed: {_failure_detail(result) or 'unknown error'}"
        )
    return result.stdout


def _failure_detail(result: subprocess.CompletedProcess) -> str:
    """Everything git said, on one line, in the order it said it."""
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    return " / ".join(line.strip() for line in lines if line.strip())


def is_git_repo(cwd: str) -> bool:
    try:
        return _git(["rev-parse", "--is-inside-work-tree"], cwd).strip() == "true"
    except WorktreeError:
        return False


def _nearest_existing_dir(cwd: str) -> str:
    """``cwd`` if it exists, else the closest ancestor that does.

    A worktree removed by another session leaves this one pointing at a path
    with nothing behind it. Its parent (``.agentica/worktrees``, or the
    repository itself) is still there and still inside the same repository, so
    that is where a git question about "this session's repo" can be answered.
    """
    path = Path(cwd)
    if path.is_dir():
        return cwd
    for parent in path.parents:
        if parent.is_dir():
            return str(parent)
    return cwd


def _directory_is_worktree_root(path: str) -> bool:
    """True when ``path`` is the root of a git checkout, not merely a directory.

    A relative write (or ``mkdir``) can resurrect a deleted worktree's path as
    a bare directory. Under the default in-repo layout that directory still
    sits inside the main checkout, so ``rev-parse --show-toplevel`` walks up
    and names main — reuse then looks successful and the lie sticks. The
    directory is a checkout of this path only when the toplevel *is* the path.
    """
    if not os.path.isdir(path):
        return False
    try:
        top = _git(["rev-parse", "--show-toplevel"], path).strip()
    except WorktreeError:
        return False
    return bool(top) and os.path.realpath(top) == os.path.realpath(path)


def main_root(cwd: str) -> str:
    """The main checkout's root, even when called from inside a worktree.

    ``--git-common-dir`` is the shared ``.git`` of the repository; its parent is
    the main worktree. Deriving paths from here is what keeps a worktree of a
    worktree from ever happening.

    Answered from the nearest ancestor that still exists, because a directory
    removed from elsewhere must still be *nameable*: a stale registration
    can only be ``remove``d if we can still name the repository it belongs to.
    """
    common = _git(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"],
        _nearest_existing_dir(cwd),
    ).strip()
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

    Default: ``<repo>/.agentica/worktrees/<task>``. ``sibling`` restores
    ``../<repo>-<task>``. An absolute ``worktree.root`` namespaces by repository
    (``<root>/<repo>/<task>``); any other relative value is inside the checkout.
    """
    root = Path(main_root(cwd))
    configured = _configured_root()
    if configured == SIBLING_ROOT:
        return str(root.parent / f"{root.name}-{slug(name)}")
    base = Path(os.path.expanduser(configured))
    if not base.is_absolute():
        # In-repo (``.agentica/worktrees``): the repository is already implied by
        # where the root lives, so ``<root>/<task>`` — inserting the repo name
        # again would read ``proj/.agentica/worktrees/proj/docs``.
        return str(root / base / slug(name))
    # One directory serving several repositories has to say which one.
    return str(base / root.name / slug(name))


def _worktree_setting(name: str, default: Optional[str] = None):
    """``settings.worktree.<name>``, also accepting the flat ``worktree.<name>`` key."""
    try:
        from agentica.global_config import get_setting, load_global_config

        flat = get_setting(f"worktree.{name}", None)
        if flat is not None:
            return flat
        data = load_global_config()
        settings = data.get("settings")
        if isinstance(settings, dict):
            block = settings.get("worktree")
            if isinstance(block, dict) and name in block:
                return block[name]
    except Exception:
        # Worktrees must keep working when config.yaml is missing or broken.
        return default
    return default


def _configured_root() -> str:
    """``worktree.root`` from config.yaml, or ``DEFAULT_ROOT`` when unset."""
    configured = _worktree_setting("root", None)
    if configured is None or not str(configured).strip():
        return DEFAULT_ROOT
    return str(configured).strip()


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
        configured = _worktree_setting("link", None)
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
    locked = False
    lock_reason = ""

    def flush() -> None:
        nonlocal path, head, branch, detached, locked, lock_reason
        if path:
            resolved = os.path.realpath(path)
            is_main = resolved == os.path.realpath(main)
            entries.append(Worktree(
                name=Path(path).name,
                path=path,
                branch="" if detached else branch,
                head=head,
                is_main=is_main,
                exists=_directory_is_worktree_root(path),
                locked=locked,
                lock_reason=lock_reason,
            ))
        path = head = branch = ""
        detached = False
        locked = False
        lock_reason = ""

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
        elif line == "locked" or line.startswith("locked "):
            locked = True
            lock_reason = line[len("locked "):].strip() if line.startswith("locked ") else ""
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


def _entry_for_path(path: str) -> Optional[Worktree]:
    want = os.path.realpath(path)
    probe = path if is_git_repo(path) else str(Path(path).parent)
    if not is_git_repo(probe):
        return None
    for entry in list_worktrees(probe):
        if os.path.realpath(entry.path) == want:
            return entry
    return None


def _invalidate_nested(cwd: str) -> None:
    _nested_cache.pop(os.path.realpath(cwd), None)
    try:
        _nested_cache.pop(os.path.realpath(main_root(cwd)), None)
    except (WorktreeError, OSError):
        pass


def is_managed(entry: Worktree) -> bool:
    """True when this checkout is one agentica created (branch ``wt/<name>``)."""
    return entry.branch_short.startswith(BRANCH_PREFIX)


def resolve_entry(cwd: str) -> Worktree:
    """The worktree record for ``cwd`` (a worktree root or a path inside one)."""
    entry = _entry_for_path(cwd)
    if entry is None and is_git_repo(cwd):
        entry = _entry_for_path(current_root(cwd))
    if entry is None:
        raise WorktreeError(f"{cwd} is not a registered worktree")
    if not entry.exists:
        raise WorktreeError(
            f"{entry.path} is registered as a worktree but is not a checkout "
            f"— its branch is {entry.branch_short or 'detached'}. "
            f"Remove it with worktree(action=\"remove\", name=\"{entry.name}\") "
            "first, or pick another name."
        )
    return entry


def _registration_at(cwd: str, path: str) -> Optional[Worktree]:
    """The listed worktree at ``path``, asked of the repository ``cwd`` is in.

    The probe is the session's checkout, not the path's parent.
    ``_entry_for_path`` walks to ``Path(path).parent`` when the tree itself
    is not a repo — that works for the default in-repo layout (parent is
    still inside the checkout) and fails for ``sibling`` / an absolute
    ``worktree.root`` (parent is an ordinary directory). ``git worktree
    list`` still shows the registration as prunable; the helper just
    cannot see it. Callers that already have a repo path must use this.
    """
    want = os.path.realpath(path)
    try:
        listed = list_worktrees(_nearest_existing_dir(cwd))
    except WorktreeError:
        return None
    for entry in listed:
        if os.path.realpath(entry.path) == want:
            return entry
    return None


def check_removable(entry: Worktree) -> Worktree:
    """Raise if deleting this worktree would delete something we do not own.

    Ownership only. Dirty trees are git's to refuse. Unique commits are not:
    ``git worktree remove`` deletes the checkout and leaves the branch, and
    ``git branch -d`` then refuses to drop unmerged commits.

    A registration that is not a checkout is still removable: ``remove()``
    then clears the bookkeeping (``unlock`` + ``prune``) and does not delete
    whatever sits at the path. ``has_unique_work`` is the separate question
    session teardown asks.

    Takes the already-resolved listing, not a path. Reverse-lookup from the
    tree path cannot see a sibling / absolute-root registration whose
    parent is outside the repository.

    Does not delete anything. ``remove()`` calls this, then acts.
    """
    if entry.is_main:
        raise WorktreeError("this is the main checkout, not a worktree")
    if not is_managed(entry):
        label = entry.branch_short or "detached"
        raise WorktreeError(
            f"{entry.path} is not an agentica worktree (branch {label}); "
            "only wt/* checkouts are removed — use git worktree remove yourself"
        )
    return entry


def has_unique_work(entry: Worktree) -> bool:
    """Whether this worktree holds anything the local base does not.

    Uncommitted files, or commits the base lacks. Session teardown uses this to
    decide whether to clean the checkout up or leave it on disk
    for the next session. It is deliberately not part of ``check_removable``:
    an explicit "remove this" is an instruction, while teardown is a guess
    about work nobody asked to throw away.
    """
    if _git(["status", "--porcelain"], entry.path).strip():
        return True
    if not entry.branch_short:
        return False
    try:
        base = default_base(entry.path)
    except WorktreeError:
        # No local base to compare against: cannot prove the work is landed,
        # so treat it as unique rather than delete it.
        return True
    ahead = _git(
        ["rev-list", "--count", f"{base}..{entry.branch_short}"], entry.path,
        check=False,
    ).strip()
    if not ahead.isdigit():
        # Cannot tell — teardown must not delete.
        return True
    return int(ahead) > 0


def _clear_stale_registration(entry: Worktree, main: str) -> None:
    """Drop a registration that is not a checkout. Does not delete the path.

    ``git worktree remove`` on these is a dead end: a missing ``.git`` is
    rc 128, and a leftover lock on a gone directory is the same refusal.
    ``unlock`` then ``prune`` clears the bookkeeping. Prune only expires
    registrations whose checkout is gone or unverifiable — it does not
    delete leftover files at the path.
    """
    _git(["worktree", "unlock", entry.path], main, check=False)
    # prune is repo-wide: every other prunable registration goes too.
    # It does not delete files or branches. A per-path ``remove -f -f``
    # would be narrower, but on a bare / ``.git``-less directory it
    # deletes leftover files — the case prune exists to leave alone.
    _git(["worktree", "prune", "--expire", "now"], main)
    if _registration_at(main, entry.path) is not None:
        raise WorktreeError(
            f"{entry.path} is still registered after prune; "
            "the name cannot be reused"
        )


def remove(entry: Worktree, cwd: str) -> Worktree:
    """Delete this worktree and its branch.

    ``entry`` is the listing already resolved from ``cwd``'s repository
    (``find`` / ``list_worktrees``). ``cwd`` is a path inside that
    repository — the session directory, not the tree. Ownership
    (``check_removable``: a ``wt/`` checkout, not the main tree) then
    git: a dirty tree is refused; a branch the base has not absorbed
    stays after ``git branch -d`` fails.

    A registration that is not a checkout (directory gone, leftover lock,
    or a bare directory at the path) is cleared with ``unlock`` + ``prune``
    so ``new`` can reuse the name. That path is not deleted.

    Refuses when this process's cwd *is* a live checkout (``--worktree``
    at start): removing it would delete the directory the process is in.
    """
    check_removable(entry)
    main = main_root(_nearest_existing_dir(cwd))
    current = _registration_at(main, entry.path)
    if current is None:
        raise WorktreeError(f"{entry.path} is not a registered worktree")
    check_removable(current)
    _invalidate_nested(main)
    if not current.exists:
        _clear_stale_registration(current, main)
        if current.branch_short:
            _git(["branch", "-d", current.branch_short], main, check=False)
        return current
    try:
        here = os.path.realpath(os.getcwd())
    except OSError:
        here = ""
    if here and os.path.realpath(current.path) == here:
        raise WorktreeError(
            f"{current.path} is this process's working directory; "
            "remove it from another session, or exit first"
        )
    _git(["worktree", "remove", current.path], main)
    if current.branch_short:
        _git(["branch", "-d", current.branch_short], main, check=False)
    return current


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
    checkouts ("Skipping repository"), double ``-ff`` removes them.
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

    Idempotent on a healthy checkout: the second call returns the same
    directory, branch and history. A registration that is no longer a
    checkout is refused — ``remove`` the name first, or pick another.
    """
    cwd = _nearest_existing_dir(cwd)
    if not is_git_repo(cwd):
        raise WorktreeError(f"{cwd} is not inside a git repository")

    existing = find(cwd, name)
    if existing is not None and not existing.exists:
        raise WorktreeError(
            f"{existing.path} is registered as a worktree but is not a checkout "
            f"(branch {existing.branch_short or 'detached'}). "
            f"Remove it with worktree(action=\"remove\", name=\"{name}\") first, "
            "or pick another name."
        )
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
    _invalidate_nested(cwd)
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
    # The base already contained every commit of the branch, so there was
    # nothing to land. Not a failure: it is what a finished (or never started)
    # task looks like, and the caller's next step — remove the checkout — is
    # the same as after a successful merge.
    already_merged: bool = False

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicted_files)


def merge_back(cwd: str, *, base: Optional[str] = None) -> MergeResult:
    """Land this worktree's branch on the local base branch.

    Does not delete the checkout — ``remove()`` does that, and the binder
    calls it after this returns clean. The order is what makes it safe:

    1. **Base into the branch, inside the worktree.** A conflict then belongs to
       the session that wrote the code, in the directory it was written in, with
       its tests one command away — instead of stranding a half-merged index in
       the main checkout that every other session shares.
    2. **Branch into base, in the main checkout.** After step 1 this is a
       fast-forward, so the shared checkout is touched for as little as possible.
       Git's own index lock is the mutex against another session doing the same
       thing; ``_git_patient`` waits it out rather than adding a second lock.

    After step 1 the worktree is level with the base, which is the safety
    standard ``remove()`` uses ("no commits the local base does not have").

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
        # Nothing to land — the base already has it all. This is the state a
        # finished task ends in (merged earlier, or committed straight onto the
        # base), so report it and let the caller clean up. Raising here made the
        # tool look like a dead end for its own success case, and a session that
        # believed that went around it with a raw ``git worktree remove`` of the
        # directory it was standing in.
        return MergeResult(
            branch=branch, base=base_ref, commits=0,
            merged_sha=_git(["rev-parse", "--short", "HEAD"], root).strip(),
            already_merged=True,
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
