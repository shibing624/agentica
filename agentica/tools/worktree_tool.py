# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The ``worktree`` tool — a session moves itself into its own checkout.

Isolation between sessions is worth nothing if arranging it requires a human at
that particular keyboard. The sessions that most need a worktree are the ones
running unattended for weeks, driven from IM through ``send_message``; so the
capability is a tool, and "把你自己切到 gateway 那个 worktree" is something a
peer message can actually carry out.

The tool is CLI-only, and not because of a policy: moving a session means moving
the process cwd, which a gateway serving many sessions in one process cannot do
for one of them.
"""
from __future__ import annotations

from agentica.tools.base import Tool
from agentica.worktrees import WorktreeError


class WorktreeTool(Tool):
    """Expose ``worktree`` so the agent can bind its own session to a checkout."""

    def __init__(self, binder):
        super().__init__(name="worktree_tool")
        self._binder = binder
        # Not concurrency-safe by any reading: it changes the directory every
        # other tool resolves paths against.
        self.register(self.worktree, is_destructive=False)

    async def worktree(self, action: str = "status", name: str = "", base: str = "") -> str:
        """Put this session in its own git worktree of the current repository.

        Isolation for parallel work: one directory and one branch per task,
        sharing the repository's history. Other sessions keep working in theirs,
        so neither overwrites the other's files and neither waits on git's index.

        Args:
            action: ``status`` (default) lists every worktree and says which one
                this session is in. ``use`` moves this session into the worktree
                for ``name``, creating it the first time and reusing it while
                the task is in progress. ``merge`` lands this worktree's branch
                on the local base and removes the checkout. ``remove`` drops an
                unused worktree (refused if it still has unique work).
            name: The task the worktree is for, e.g. "gateway-peers". Required
                for ``use``. Normalised to a directory under
                ``.agentica/worktrees/`` and a ``wt/<name>`` branch.
            base: Branch new worktrees fork from. Defaults to the repository's
                local ``main`` (or ``master``).

        Returns:
            What happened, including the directory and branch now in effect.
        """
        chosen = (action or "status").strip().casefold()
        try:
            if chosen in ("status", "list", "info", ""):
                return self._binder.status()
            if chosen in ("use", "switch", "bind", "create", "new"):
                if not name.strip():
                    return (
                        "A name is required: worktree(action=\"use\", name=\"<task>\"). "
                        "Call action=\"status\" to see the worktrees that already exist."
                    )
                return self._binder.switch(name, base=base.strip() or None)
            if chosen in ("merge", "merge-back", "land"):
                return self._binder.merge()
            if chosen in ("remove", "delete", "drop"):
                return self._binder.remove()
            return (
                f"Unknown action '{action}'. Use status, use (with name=...), "
                "merge, or remove."
            )
        except WorktreeError as e:
            # The message is written for a human ("move it aside or pick another
            # name"); paraphrasing it would only lose that.
            return f"Worktree operation refused: {e}"
