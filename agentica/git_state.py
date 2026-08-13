# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: One session's git position, collected cheaply enough to publish.

Two agentica sessions working the same repository spend real effort answering
"did you already change that file?" and "am I behind main?" by *asking each
other* — a message, a turn on both sides, and an answer that is stale the
moment a third session commits. Git already knows all of it. So presence
carries it: every session publishes its branch, head, distance from the base
branch and its dirty files (``agentica/peers.py``), and ``list_agents`` answers
the question with no round trip.

Two constraints shape this module:

**It runs on a presence tick, so it must be cheap and never raise.** Three
short-lived git calls behind a TTL cache (``CACHE_TTL``): the CLI's peer loop
wakes every second, and shelling out three times a second per session would be
a background load nobody asked for. Anything unexpected — not a repository, git
missing, a locked index, a timeout — yields ``GitState()``, which publishes as
"nothing to say" rather than breaking the tick.

**"Behind" has to mean behind what the team merges into.** Upstream tracking is
the wrong yardstick alone: a peer that commits to local ``main`` without
pushing is invisible to ``origin/main``, and that peer is exactly who you are
about to conflict with. So the base is the local default branch when there is
one (``main``, else ``master``), and the upstream only when there is not.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# A presence tick may ask far more often than the answer can change in a way
# anyone acts on. One collection per 10s per directory is plenty, and it also
# bounds how often a changed dirty list can force a presence write.
CACHE_TTL = 10.0

# Git should answer in milliseconds; a hang (network remote, stale lock) must
# not stall the peer loop.
TIMEOUT = 3.0

# How many paths travel in a live record. The point is recognising an overlap,
# not reproducing `git status` — and every listing pays for this.
MAX_DIRTY_FILES = 12

DEFAULT_BRANCHES = ("main", "master")


@dataclass(frozen=True)
class GitState:
    """Where a session sits in its repository. Empty means "no answer"."""

    branch: str = ""
    head_sha: str = ""
    # What ahead/behind was measured against ("main", "origin/main", ...).
    base_ref: str = ""
    ahead: int = 0
    behind: int = 0
    # Modified/added/deleted/untracked paths, repo-relative, capped.
    dirty_files: Tuple[str, ...] = ()
    # Total before truncation, so a listing can say "12 of 40". ``None`` means
    # *not collected* — a record from an older agentica, or a directory git
    # would not answer for. It must not read as "clean": another session
    # deciding whether it is safe to rebase would act on a claim nobody made.
    dirty_count: Optional[int] = None

    @property
    def known(self) -> bool:
        return bool(self.branch or self.head_sha)

    def summary(self) -> str:
        """One line for a listing: branch, head, distance, dirty count."""
        if not self.known:
            return ""
        parts = []
        if self.branch:
            parts.append(self.branch)
        if self.head_sha:
            parts.append(f"@ {self.head_sha}")
        if self.base_ref and (self.ahead or self.behind):
            parts.append(f"+{self.ahead}/-{self.behind} vs {self.base_ref}")
        elif self.base_ref:
            parts.append(f"in sync with {self.base_ref}")
        if self.dirty_count is None:
            pass
        elif self.dirty_count:
            parts.append(f"{self.dirty_count} dirty")
        else:
            parts.append("clean")
        return " · ".join(parts)

    def dirty_line(self) -> str:
        """The paths themselves, with a count when the list was truncated."""
        if not self.dirty_files:
            return ""
        shown = ", ".join(self.dirty_files)
        hidden = (self.dirty_count or 0) - len(self.dirty_files)
        if hidden > 0:
            return f"{shown} (+{hidden} more)"
        return shown


@dataclass
class _CacheEntry:
    state: GitState
    at: float = field(default_factory=time.time)


_cache: Dict[str, _CacheEntry] = {}


def _git(args: List[str], cwd: str) -> Optional[str]:
    """Run one git command, or None when it did not answer cleanly."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _base_ref(cwd: str, branch: str, upstream: str) -> str:
    """The ref this session's work is measured against.

    Local ``main`` first (a peer's unpushed commit is the one about to collide
    with yours), the upstream second, and nothing when neither exists — a
    detached or orphan checkout has no meaningful distance to report.
    """
    for candidate in DEFAULT_BRANCHES:
        if candidate == branch:
            continue
        if _git(["rev-parse", "--verify", "--quiet", candidate], cwd) is not None:
            return candidate
    return upstream


def _parse_status(output: str) -> Tuple[str, str, List[str]]:
    """Split ``git status -sb --porcelain`` into branch, upstream and paths."""
    branch = ""
    upstream = ""
    paths: List[str] = []
    for line in output.splitlines():
        if line.startswith("## "):
            head = line[3:].strip()
            # "branch...upstream [ahead 1, behind 2]" — the bracket counts are
            # dropped on purpose: they are measured against the upstream, and
            # the base we report may be a local branch instead.
            head = head.split(" [", 1)[0]
            branch, _, upstream = head.partition("...")
            branch = branch.strip()
            upstream = upstream.strip()
            if branch.startswith("HEAD (no branch)"):
                branch = ""
            continue
        if len(line) < 4:
            continue
        path = line[3:].strip()
        # A rename reads "old -> new"; the new path is what someone else would
        # collide with.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip('"'))
    return branch, upstream, paths


def collect(work_dir: str, *, ttl: float = CACHE_TTL) -> GitState:
    """This directory's git position, cached for ``ttl`` seconds.

    Never raises: a non-repository, a missing git, or anything unexpected is
    reported as an empty ``GitState``.
    """
    now = time.time()
    cached = _cache.get(work_dir)
    if cached is not None and now - cached.at < ttl:
        return cached.state

    state = _collect_uncached(work_dir)
    _cache[work_dir] = _CacheEntry(state=state, at=now)
    return state


def invalidate(work_dir: Optional[str] = None) -> None:
    """Drop cached state — used right after this process changes the repo."""
    if work_dir is None:
        _cache.clear()
    else:
        _cache.pop(work_dir, None)


def _collect_uncached(work_dir: str) -> GitState:
    status = _git(["status", "-sb", "--porcelain"], work_dir)
    if status is None:
        return GitState()
    branch, upstream, paths = _parse_status(status)
    head = (_git(["rev-parse", "--short", "HEAD"], work_dir) or "").strip()

    base = _base_ref(work_dir, branch, upstream)
    ahead = behind = 0
    if base:
        counts = _git(["rev-list", "--left-right", "--count", f"{base}...HEAD"], work_dir)
        if counts:
            fields = counts.split()
            if len(fields) == 2 and all(f.isdigit() for f in fields):
                behind, ahead = int(fields[0]), int(fields[1])
        else:
            # An unrelated or missing base is not a distance of zero.
            base = ""

    return GitState(
        branch=branch,
        head_sha=head,
        base_ref=base,
        ahead=ahead,
        behind=behind,
        dirty_files=tuple(sorted(paths)[:MAX_DIRTY_FILES]),
        dirty_count=len(paths),
    )
