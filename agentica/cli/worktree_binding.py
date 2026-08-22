# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Bind a *running* CLI session to a per-task git worktree.

``agentica --worktree <task>`` covers a session that is about to start. This
module covers the one that has been running for weeks — which is the case that
matters, because those are the sessions a user drives from IM ("切到 gateway
那个 worktree 去改") and the ones whose context is too expensive to restart.

"The session's directory" is four things that must move together, and this class
is the only place that knows all four:

* the **process cwd** — git, ``@file`` completion and shell-outs read it;
* the **agent's execution** — prompt, sandbox and every file/shell tool
  (``Agent.rebind_work_dir``);
* the **live peer record** — other sessions decide who is where from it, and its
  addressable name must survive the move (``PeerSession.rebind``);
* the **status bar**, so the human can see which worktree they are typing into.

What deliberately does *not* move: the transcript. ``Agent._session_log`` keeps
writing where it was already writing, which is what ``session_base_dir`` exists
for — one conversation stays one file even though its work moved. The price is
that ``/resume`` later offers the original directory; that prompt already exists
and already remembers the answer.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from agentica import git_state, worktrees
from agentica.utils.log import logger


class WorktreeBinder:
    """The coordinated "move this session into that worktree" operation.

    Constructed with getters rather than values: the CLI rebuilds its agent on
    ``/model``, ``/resume`` and ``/newchat``, and the status bar dict is created
    after the first agent. A binder holding stale references would move a
    session that no longer exists.
    """

    def __init__(
        self,
        *,
        agent_config: Dict[str, Any],
        get_agent: Callable[[], Any],
        get_peers: Callable[[], Any] = lambda: None,
        get_tui_state: Callable[[], Optional[Dict[str, Any]]] = lambda: None,
    ) -> None:
        self._agent_config = agent_config
        self._get_agent = get_agent
        self._get_peers = get_peers
        self._get_tui_state = get_tui_state

    # -- state -------------------------------------------------------------

    def work_dir(self) -> str:
        return str(self._agent_config.get("work_dir") or os.getcwd())

    def status(self) -> str:
        """Where this session is, and every worktree of the repository."""
        cwd = self.work_dir()
        if not worktrees.is_git_repo(cwd):
            return f"{cwd} is not inside a git repository, so there are no worktrees."
        here = os.path.realpath(worktrees.current_root(cwd))
        lines = [f"This session works in {cwd}"]
        state = git_state.collect(cwd, ttl=0)
        if state.known:
            lines.append(f"  {state.summary()}")
        lines.append("")
        lines.append("Worktrees of this repository:")
        for entry in worktrees.list_worktrees(cwd):
            marker = " <- you are here" if os.path.realpath(entry.path) == here else ""
            lines.append(f"  {entry.describe()}{marker}")
        lines.append("")
        lines.append(
            "Switch with worktree(action=\"use\", name=\"<task>\") — created "
            "if missing, reused while the task is in progress. "
            "worktree(action=\"merge\") lands on the local base and removes "
            "the checkout; worktree(action=\"remove\") drops an unused one."
        )
        return "\n".join(lines)

    # -- the move ----------------------------------------------------------

    def switch(self, name: str, *, base: Optional[str] = None) -> str:
        """Enter the worktree for ``name``, creating it on first use."""
        agent = self._get_agent()
        if agent is None:
            raise worktrees.WorktreeError("no active agent to move")

        cwd = self.work_dir()
        worktree = worktrees.ensure(cwd, name, base=base)
        owned = worktrees.claim_lock(worktree.path)
        target = os.path.realpath(worktree.path)
        if target == os.path.realpath(cwd):
            extra = ""
            if not owned:
                holder = worktrees.lock_holder_description(worktree.path)
                extra = (
                    f" another live session ({holder}) is in this worktree — "
                    "sharing it; expect index.lock contention."
                )
            return (
                f"Already working in {worktree.path} "
                f"(branch {worktree.branch_short}); nothing to move.{extra}"
            )

        previous = os.path.realpath(cwd)
        state = self._relocate(worktree.path)
        if os.path.realpath(previous) != os.path.realpath(worktrees.main_root(worktree.path)):
            worktrees.release_lock(previous)

        logger.info(f"Session bound to worktree {worktree.path} ({worktree.branch_short})")

        lines = [
            f"Now working in {worktree.path}",
            f"  branch: {worktree.branch_short}",
        ]
        if state.known:
            lines.append(f"  {state.summary()}")
        if worktree.linked:
            lines.append(f"  linked from the main checkout: {', '.join(worktree.linked)}")
        if not owned:
            holder = worktrees.lock_holder_description(worktree.path)
            lines.append(
                f"  another live session ({holder}) is in this worktree — "
                "sharing it; expect index.lock contention"
            )
        lines.append(
            "  transcript: still written where this session started "
            "(one conversation stays one file)"
        )
        lines.append(
            "  other sessions see the new directory and branch in list_agents; "
            "this session's name did not change."
        )
        return "\n".join(lines)

    def merge(self, *, base: Optional[str] = None) -> str:
        """Land this worktree's branch on the local base, then delete the checkout."""
        cwd = self.work_dir()
        result = worktrees.merge_back(cwd, base=base)
        git_state.invalidate()

        if result.conflicted:
            files = "\n".join(f"    {path}" for path in result.conflicted_files)
            return (
                f"{result.base} was merged into {result.branch} and conflicted. "
                f"Nothing has been merged into {result.base}.\n"
                f"  resolve here ({cwd}), commit, then merge again:\n{files}"
            )

        main = worktrees.main_root(cwd)
        root = worktrees.current_root(cwd)
        self._relocate(main)
        leftover = ""
        try:
            worktrees.remove(root)
        except worktrees.WorktreeError as exc:
            leftover = str(exc)

        landed = (
            f"Merged {result.commits} commit(s) from {result.branch} into "
            f"{result.base} ({result.merged_sha})."
        )
        if leftover:
            return (
                f"{landed}\n"
                f"  now working in {main}.\n"
                f"  worktree not removed: {leftover}"
            )
        return (
            f"{landed}\n"
            f"  worktree removed; now working in {main} on {result.base}.\n"
            f"  nothing was pushed — do that explicitly if you want it on the remote."
        )

    def remove(self) -> str:
        """Drop this worktree if it holds no unique work, and return to main."""
        cwd = self.work_dir()
        root = worktrees.current_root(cwd)
        main = worktrees.main_root(cwd)
        if os.path.realpath(root) == os.path.realpath(main):
            raise worktrees.WorktreeError("this is the main checkout, not a worktree")
        worktrees.check_removable(root)
        self._relocate(main)
        worktrees.remove(root)
        return (
            f"Removed worktree {root} and returned to {main}. "
            "The branch is gone if it had no unique commits."
        )

    def release(self) -> Optional[str]:
        """Session teardown: delete a clean unused *agentica* worktree.

        Called from the CLI ``finally`` so a ``--worktree`` that never produced
        unique work does not leak a directory. Foreign checkouts (detached,
        Claude Code, a hand-made ``git worktree add``) are left alone. Unique
        work (dirty or unmerged) is left on disk *and stays locked*; the next
        ``use`` of the same name steals the lock once this pid is dead.
        """
        cwd = self.work_dir()
        if not worktrees.is_git_repo(cwd):
            return None
        try:
            root = worktrees.current_root(cwd)
            main = worktrees.main_root(cwd)
            entry = worktrees.resolve_entry(root)
        except worktrees.WorktreeError:
            return None
        if entry.is_main or not worktrees.is_managed(entry):
            return None
        from agentica.cli.session_resume import enter_work_dir
        enter_work_dir(main)
        try:
            worktrees.remove(root)
            self._agent_config["work_dir"] = main
            return root
        except worktrees.WorktreeError:
            return None

    def _relocate(self, target: str):
        """Move process cwd, agent execution, peer record and status bar together."""
        from agentica.cli.session_resume import enter_work_dir

        if not enter_work_dir(target):
            raise worktrees.WorktreeError(f"cannot enter {target}")
        agent = self._get_agent()
        if agent is not None:
            agent.rebind_work_dir(target)
        self._agent_config["work_dir"] = target
        peers = self._get_peers()
        if peers is not None:
            peers.rebind(target)
        git_state.invalidate()
        state = git_state.collect(target, ttl=0)
        tui_state = self._get_tui_state()
        if tui_state is not None:
            tui_state["work_dir"] = target
            tui_state["git_branch"] = state.branch
        return state
