# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The ``worktree`` tool — create a checkout, then pass its path.

Isolation is a directory, not a session move. ``new`` returns a path; subsequent
``read_file`` / ``write_file`` / ``apply_patch`` / ``execute`` take ``work_dir=``
(glob / grep take ``path=``). This session stays where it is.
"""
from __future__ import annotations

from agentica.tools.base import Tool
from agentica.worktrees import WorktreeError


class WorktreeTool(Tool):
    """Expose ``worktree`` so the agent can create and dispose checkouts."""

    def __init__(self, binder):
        super().__init__(name="worktree_tool")
        self._binder = binder
        self.register(self.worktree, is_destructive=False)

    async def worktree(self, action: str = "status", name: str = "", base: str = "") -> str:
        """Create a git worktree of this repository, or dispose of one.

        Isolation for parallel work: one directory and one branch per task,
        sharing the repository's history. This session does not move. Pass the
        returned path as ``work_dir`` on file / execute calls (``path`` on
        glob / grep).

        Args:
            action: ``status`` (default) lists every worktree. ``new`` creates
                or reuses the worktree for ``name`` and returns its path.
                ``merge`` lands that branch on the local base and removes the
                checkout. ``remove`` drops a ``wt/*`` checkout; git refuses if
                it is dirty, and an unmerged branch is left in place.
            name: The task the worktree is for, e.g. "gateway-peers". Required
                for ``new``, ``merge``, and ``remove``.
            base: Branch new worktrees fork from. Defaults to the repository's
                local ``main`` (or ``master``).

        Returns:
            What happened, including the directory to pass as ``work_dir``.
        """
        chosen = (action or "status").strip().casefold()
        try:
            if chosen in ("status", "list", "info", ""):
                return self._binder.status()
            if chosen in ("new", "use", "switch", "bind", "create"):
                if not name.strip():
                    return (
                        "A name is required: worktree(action=\"new\", name=\"<task>\"). "
                        "Call action=\"status\" to see the worktrees that already exist."
                    )
                return self._binder.create(name, base=base.strip() or None)
            if chosen in ("main", "home"):
                return (
                    "There is no main action — this session does not move. "
                    "Stay in the current directory. Pass work_dir= only on "
                    "the calls that should run in a worktree."
                )
            if chosen in ("merge", "merge-back", "land"):
                if not name.strip():
                    return (
                        "A name is required: worktree(action=\"merge\", name=\"<task>\")."
                    )
                return self._binder.merge(name, base=base.strip() or None)
            if chosen in ("remove", "delete", "drop"):
                if not name.strip():
                    return (
                        "A name is required: worktree(action=\"remove\", name=\"<task>\")."
                    )
                return self._binder.remove(name)
            return (
                f"Unknown action '{action}'. Use status, new (with name=...), "
                "merge (with name=...), or remove (with name=...)."
            )
        except WorktreeError as e:
            return f"Worktree operation refused: {e}"
