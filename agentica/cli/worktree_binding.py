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
            "Switch with worktree(action=\"use\", name=\"<task>\") — it is created "
            "on first use and reused afterwards."
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
        target = os.path.realpath(worktree.path)
        if target == os.path.realpath(cwd):
            return (
                f"Already working in {worktree.path} "
                f"(branch {worktree.branch_short}); nothing to move."
            )

        # cwd first: everything below reads or reports it.
        from agentica.cli.session_resume import enter_work_dir

        if not enter_work_dir(worktree.path):
            raise worktrees.WorktreeError(f"cannot enter {worktree.path}")

        agent.rebind_work_dir(worktree.path)
        self._agent_config["work_dir"] = worktree.path

        peers = self._get_peers()
        if peers is not None:
            peers.rebind(worktree.path)

        git_state.invalidate()
        state = git_state.collect(worktree.path, ttl=0)

        tui_state = self._get_tui_state()
        if tui_state is not None:
            tui_state["work_dir"] = worktree.path
            tui_state["git_branch"] = state.branch

        logger.info(f"Session bound to worktree {worktree.path} ({worktree.branch_short})")

        lines = [
            f"Now working in {worktree.path}",
            f"  branch: {worktree.branch_short}",
        ]
        if state.known:
            lines.append(f"  {state.summary()}")
        if worktree.linked:
            lines.append(f"  linked from the main checkout: {', '.join(worktree.linked)}")
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
        """Land this worktree's branch on the base branch, keeping the worktree."""
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

        tui_state = self._get_tui_state()
        state = git_state.collect(cwd, ttl=0)
        if tui_state is not None:
            tui_state["git_branch"] = state.branch
        return (
            f"Merged {result.commits} commit(s) from {result.branch} into "
            f"{result.base} ({result.merged_sha}).\n"
            f"  this worktree is kept and is now level with {result.base}, "
            "so it stays usable for the next task on the same thing.\n"
            f"  nothing was pushed — do that explicitly if you want it on the remote."
        )
