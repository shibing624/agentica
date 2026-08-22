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

**A worktree is a feature checkout, not a standing room.** ``ensure()`` creates
or reuses an *in-progress* directory (so a long-running session can still be
told "切到 gateway-peers 再改"). ``merge_back()`` lands the branch on the local
base; ``remove()`` deletes the checkout and the ``wt/<name>`` branch. Remove refuses anything that would lose work: a checkout agentica did not
create (branch is not ``wt/<name>``, including detached), uncommitted files,
commits not on the local base, no local ``main``/``master`` to compare
against, or a lock held by a live process. That last one is why we take
``git worktree lock`` while a session is bound — without it, another process
(or a sweep) can ``git worktree remove`` a tree the agent is mid-edit in.
The lock stays on a dirty tree when the session exits; a later ``claim_lock``
steals it once the pid is dead.

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
# ``-f``) skips the nested checkout; ``git clean -xdff`` will remove it — that
# is acceptable now that a finished task is merged and deleted, and an
# in-progress one is ``git worktree lock``'d for the life of the session
# (left locked if the session exits with unique work still in it). Opt out with:
#
#   * ``worktree.root: sibling`` → ``../<repo>-<task>`` (the old default);
#   * an absolute path → ``<root>/<repo>/<task>`` (one farm, many repos);
#   * any other relative path → ``<repo>/<that>/<task>``.
DEFAULT_ROOT = ".agentica/worktrees"
SIBLING_ROOT = "sibling"
ROOT_SETTING = "worktree.root"
LINK_SETTING = "worktree.link"
LOCK_REASON_PREFIX = "agentica pid="

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
    locked: bool = False
    lock_reason: str = ""

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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # pid 1 (and other processes we do not own): exists, just not ours.
        return True
    return True


def lock_reason_for_pid(pid: Optional[int] = None) -> str:
    return f"{LOCK_REASON_PREFIX}{pid if pid is not None else os.getpid()}"


def _pid_from_reason(reason: str) -> Optional[int]:
    if not (reason or "").startswith(LOCK_REASON_PREFIX):
        return None
    token = reason[len(LOCK_REASON_PREFIX):].strip().split()[0]
    try:
        return int(token)
    except ValueError:
        return None


def _is_our_lock(reason: str) -> bool:
    return _pid_from_reason(reason) == os.getpid()


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


def lock(path: str, *, reason: Optional[str] = None) -> None:
    """``git worktree lock`` this checkout so prune/remove cannot take it."""
    target = os.path.realpath(path)
    why = reason if reason is not None else lock_reason_for_pid()
    _git(["worktree", "lock", "--reason", why, target], main_root(target))


def unlock(path: str) -> None:
    """``git worktree unlock``. Idempotent: already-unlocked is not an error."""
    target = os.path.realpath(path)
    _git(["worktree", "unlock", target], main_root(target), check=False)


def claim_lock(path: str) -> bool:
    """Hold the agentica lock for this process.

    True if we hold it. False if a live foreign holder (another agentica pid,
    or a lock the user set by hand) owns it — sharing the directory is still
    allowed; ``remove`` will refuse. A lock whose agentica pid is dead is stolen.
    """
    entry = _entry_for_path(path)
    if entry is None:
        raise WorktreeError(f"{path} is not a registered worktree")
    if not entry.locked:
        lock(path)
        return True
    if _is_our_lock(entry.lock_reason):
        return True
    foreign_pid = _pid_from_reason(entry.lock_reason)
    if foreign_pid is not None and not _pid_alive(foreign_pid):
        unlock(path)
        lock(path)
        return True
    return False


def release_lock(path: str) -> None:
    """Drop the lock only if we hold it, or if the holder pid is dead."""
    entry = _entry_for_path(path)
    if entry is None or not entry.locked:
        return
    if _is_our_lock(entry.lock_reason):
        unlock(path)
        return
    pid = _pid_from_reason(entry.lock_reason)
    if pid is not None and not _pid_alive(pid):
        unlock(path)


def is_managed(entry: Worktree) -> bool:
    """True when this checkout is one agentica created (branch ``wt/<name>``)."""
    return entry.branch_short.startswith(BRANCH_PREFIX)


def lock_holder_description(path: str) -> str:
    """How to name whoever holds the lock, for a human-facing warning."""
    entry = _entry_for_path(path)
    if entry is None or not entry.locked:
        return "unknown"
    pid = _pid_from_reason(entry.lock_reason)
    if pid is not None:
        return f"pid {pid}"
    return entry.lock_reason or "locked"


def resolve_entry(cwd: str) -> Worktree:
    """The worktree record for ``cwd`` (a worktree root or a path inside one)."""
    entry = _entry_for_path(cwd)
    if entry is None and is_git_repo(cwd):
        entry = _entry_for_path(current_root(cwd))
    if entry is None:
        raise WorktreeError(f"{cwd} is not a registered worktree")
    return entry


def check_removable(cwd: str) -> Worktree:
    """Raise if deleting this worktree would lose work or delete someone else's.

    Does not unlock or delete anything. ``remove()`` calls this, then acts.
    """
    entry = resolve_entry(cwd)
    if entry.is_main:
        raise WorktreeError("this is the main checkout, not a worktree")
    if not is_managed(entry):
        label = entry.branch_short or "detached"
        raise WorktreeError(
            f"{entry.path} is not an agentica worktree (branch {label}); "
            "only wt/* checkouts are removed — use git worktree remove yourself"
        )
    if _git(["status", "--porcelain"], entry.path).strip():
        raise WorktreeError(
            "this worktree has uncommitted changes; commit, stash, or discard "
            "them before removing it"
        )
    base = default_base(entry.path)
    ahead = _git(
        ["rev-list", "--count", f"{base}..{entry.branch_short}"], entry.path
    ).strip()
    commits = int(ahead) if ahead.isdigit() else 0
    if commits:
        raise WorktreeError(
            f"{entry.branch_short} has {commits} commit(s) not merged into the "
            f"local base ({base}); merge them first (worktree action=merge)"
        )
    if entry.locked and not _is_our_lock(entry.lock_reason):
        pid = _pid_from_reason(entry.lock_reason)
        if pid is None or _pid_alive(pid):
            why = entry.lock_reason or "no reason"
            raise WorktreeError(
                f"{entry.path} is locked ({why}); another session is using "
                "it, or unlock it by hand"
            )
    return entry


def remove(cwd: str) -> Worktree:
    """Delete this worktree and its branch, if that would not lose work.

    Safe means: an agentica ``wt/`` checkout, not the main tree, no uncommitted
    changes, no commits the local base does not already have, a local base
    exists to compare against, and not locked by a live foreign holder.
    A lock we hold (or a dead agentica pid) is released first.
    After a successful ``merge_back`` this is always safe. Must be called
    from outside the worktree (typically after moving back to the main
    checkout) so the process cwd is not deleted out from under it.
    """
    entry = check_removable(cwd)
    main = main_root(entry.path)
    if entry.locked:
        unlock(entry.path)

    _invalidate_nested(main)
    _git(["worktree", "remove", entry.path], main)
    if entry.branch_short:
        _git(["branch", "-d", entry.branch_short], main, check=False)
    return entry


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
    checkouts ("Skipping repository"), double ``-ff`` removes them. That is
    why an in-progress worktree is ``git worktree lock``'d while the session
    is alive, and left locked if the session exits with unique work still in
    it (a later claim steals the lock once the pid is dead).
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
