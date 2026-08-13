# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Warn when another live session already has this file dirty.

The expensive part of several sessions on one repository is not the merge — it
is finding out, an hour later, that someone else had the file open. Presence
already carries each session's dirty paths (``agentica/git_state.py`` published
through ``agentica/peers.py``), so the answer is a dictionary lookup at the
moment it matters: the write.

Three decisions, all in the direction of "say it, do not stop it":

**It never blocks.** A peer's dirty file is information, not a lock. Two sessions
editing one file is sometimes exactly right (one writes the fix, the other the
test) and no heuristic here can tell that from a collision. Refusing the write
would put an agent in a hole it cannot dig out of; a line in the tool result puts
the choice where the judgement is.

**Only the same repository counts.** Comparing bare relative paths across
unrelated checkouts would warn about ``README.md`` forever. Peers publish the
repository their directory belongs to (``repo_root``), so the comparison is
exact, and two *worktrees* of one repo still match — which is the case worth
warning about.

**Once per file and peer.** The same warning on every edit of the same file is
noise that trains an agent to ignore it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Set, Tuple

from agentica.utils.log import logger


class PeerConflictChecker:
    """Answers "is another live session also editing this file?" for one session."""

    def __init__(self, peer_session) -> None:
        self._peers = peer_session
        self._warned: Set[Tuple[str, str]] = set()

    def check(self, abs_path: str) -> str:
        """A warning line for ``abs_path``, or "" when nobody else has it dirty."""
        try:
            return self._check(abs_path)
        except OSError:
            # Presence lives in a shared directory; failing to read it is never
            # worth failing a write over.
            logger.debug("peer conflict check failed", exc_info=True)
            return ""

    def _check(self, abs_path: str) -> str:
        from agentica import git_state

        mine = git_state.collect(str(Path(abs_path).parent))
        if not mine.repo_root:
            return ""
        try:
            relative = os.path.relpath(os.path.realpath(abs_path), os.path.realpath(mine.repo_root))
        except ValueError:
            return ""
        if relative.startswith(".."):
            return ""

        notes = []
        for peer in self._peers.list_peers():
            state = peer.git_state
            if not state.repo_root or os.path.realpath(state.repo_root) != os.path.realpath(mine.repo_root):
                continue
            if relative not in state.dirty_files:
                continue
            key = (peer.peer_id, relative)
            if key in self._warned:
                continue
            self._warned.add(key)
            where = f"{peer.name} ({state.branch or 'no branch'}, {peer.cwd})"
            notes.append(where)

        if not notes:
            return ""
        who = "; ".join(notes)
        return (
            f"Another live session has {relative} uncommitted: {who}. "
            "Your write went through — decide whether to coordinate "
            "(send_message) or to work in your own checkout "
            "(worktree(action=\"use\", name=...)). Said once per file and session."
        )


def build_checker(peer_session) -> Optional[PeerConflictChecker]:
    """A checker, or None when this session has no presence to compare against."""
    return PeerConflictChecker(peer_session) if peer_session is not None else None
