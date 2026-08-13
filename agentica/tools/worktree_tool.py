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

from typing import Optional

from agentica.tools.base import Tool
from agentica.worktrees import WorktreeError

WORKTREE_POLICY = """<worktrees>
Several sessions sharing one checkout overwrite each other's edits and fight
over git's index. A worktree is one directory + one branch per *task*, sharing
the repository — created on first use, reused forever after, never deleted.

Use `worktree(action="use", name="<task>")` when you are about to make changes
and another live session is working in the same directory (`list_agents` shows
each session's directory, branch and dirty files). The switch is immediate and
does not restart anything: this conversation, its todo list and its goal
continue, and the transcript keeps being written where it already was.

Name the *task*, not yourself: a name is a place two sessions can hand work over
in ("gateway-peers", "wechat-fix"), and reusing it is the point.

`worktree(action="merge")` puts the work on the base branch and leaves the
worktree in place, caught up with the base so it stays usable.
</worktrees>"""


class WorktreeTool(Tool):
    """Expose ``worktree`` so the agent can bind its own session to a checkout."""

    def __init__(self, binder):
        super().__init__(name="worktree_tool")
        self._binder = binder
        # Not concurrency-safe by any reading: it changes the directory every
        # other tool resolves paths against.
        self.register(self.worktree, is_destructive=False)

    def get_system_prompt(self) -> Optional[str]:
        return WORKTREE_POLICY

    async def worktree(self, action: str = "status", name: str = "", base: str = "") -> str:
        """Put this session in its own git worktree of the current repository.

        Isolation for parallel work: one directory and one branch per task,
        sharing the repository's history. Other sessions keep working in theirs,
        so neither overwrites the other's files and neither waits on git's index.

        Args:
            action: ``status`` (default) lists every worktree and says which one
                this session is in. ``use`` moves this session into the worktree
                for ``name``, creating it the first time and reusing it after.
                ``merge`` merges this worktree's branch into the base branch
                (see the worktree tool's instructions) and keeps the worktree.
            name: The task the worktree is for, e.g. "gateway-peers". Required
                for ``use``. Normalised to a directory name next to the main
                checkout and a ``wt/<name>`` branch.
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
            return (
                f"Unknown action '{action}'. Use status, use (with name=...), or merge."
            )
        except WorktreeError as e:
            # The message is written for a human ("move it aside or pick another
            # name"); paraphrasing it would only lose that.
            return f"Worktree operation refused: {e}"
