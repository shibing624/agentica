# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Create and dispose per-task git worktrees without moving the session.

A worktree is a directory. File / execute tools take it as ``work_dir`` (glob
and grep as ``path``). This binder does not chdir, does not rebind the agent,
and does not lock. ``--worktree`` at process start is the one case the process
is born inside a tree; ``release()`` then may delete a clean unused one on exit.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from agentica.peers import git_state, worktrees


class WorktreeBinder:
    """Create / list / merge / remove worktrees. The session stays put."""

    def __init__(self, *, agent_config: Dict[str, Any]) -> None:
        self._agent_config = agent_config
        # True only after ``--worktree``. Plain CLI sessions must not run git
        # teardown on every exit.
        self._entered = False

    def work_dir(self) -> str:
        return str(self._agent_config.get("work_dir") or os.getcwd())

    def status(self) -> str:
        """Where this session is, and every worktree of the repository."""
        cwd = self.work_dir()
        if not os.path.isdir(cwd):
            return (
                f"This session's working directory no longer exists: {cwd}\n"
                "Nothing listed — git cannot run here. Restart from the main "
                "checkout (`agentica --resume`)."
            )
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
            "Create or reuse with worktree(action=\"new\", name=\"<task>\"). "
            "This session stays here. Pass the returned path as work_dir= on "
            "read_file / write_file / apply_patch / execute (glob/grep: path=). "
            "worktree(action=\"merge\", name=\"<task>\") lands on the local "
            "base and removes the checkout; worktree(action=\"remove\", "
            "name=\"<task>\") drops an unused one."
        )
        return "\n".join(lines)

    def create(self, name: str, *, base: Optional[str] = None) -> str:
        """Create or reuse the worktree for ``name``. Does not move this session."""
        cwd = self.work_dir()
        worktree = worktrees.ensure(cwd, name, base=base)
        lines = [
            f"Worktree ready at {worktree.path}",
            f"  branch: {worktree.branch_short}",
            f"  this session stayed in {cwd}",
            "  pass work_dir=" + worktree.path
            + " on read_file / write_file / apply_patch / execute "
            "(glob/grep: path=). Omit it and those calls stay in this session.",
        ]
        if worktree.linked:
            lines.append(f"  linked from the main checkout: {', '.join(worktree.linked)}")
        return "\n".join(lines)

    def merge(self, name: str, *, base: Optional[str] = None) -> str:
        """Land ``name``'s branch on the local base, then delete the checkout."""
        cwd = self.work_dir()
        entry = worktrees.find(cwd, name)
        if entry is None:
            raise worktrees.WorktreeError(f"no worktree named {name!r}")
        if not entry.exists:
            raise worktrees.WorktreeError(
                f"{entry.path} is registered as a worktree but is not a checkout. "
                f"Remove it with worktree(action=\"remove\", name=\"{name}\") first."
            )
        result = worktrees.merge_back(entry.path, base=base)
        git_state.invalidate()

        if result.conflicted:
            files = "\n".join(f"    {path}" for path in result.conflicted_files)
            return (
                f"{result.base} was merged into {result.branch} and conflicted. "
                f"Nothing has been merged into {result.base}.\n"
                f"  resolve in {entry.path}, commit, then merge again:\n{files}"
            )

        leftover = ""
        try:
            worktrees.remove(entry, cwd)
        except worktrees.WorktreeError as exc:
            leftover = str(exc)

        landed = (
            f"Merged {result.commits} commit(s) from {result.branch} into "
            f"{result.base} ({result.merged_sha})."
        )
        if result.already_merged:
            landed = (
                f"{result.base} already had every commit of {result.branch}; "
                "nothing to merge."
            )
        if leftover:
            return (
                f"{landed}\n"
                f"  this session stayed in {cwd}.\n"
                f"  worktree not removed: {leftover}"
            )
        return (
            f"{landed}\n"
            f"  worktree removed. this session stayed in {cwd} on {result.base}.\n"
            f"  nothing was pushed — do that explicitly if you want it on the remote."
        )

    def remove(self, name: str) -> str:
        """Drop the worktree for ``name``. This session does not move."""
        cwd = self.work_dir()
        entry = worktrees.find(cwd, name)
        if entry is None:
            raise worktrees.WorktreeError(f"no worktree named {name!r}")
        worktrees.remove(entry, cwd)
        return (
            f"Removed worktree {entry.path}. "
            f"This session stayed in {cwd}. "
            "The branch remains if git would not delete it (not fully merged)."
        )

    def mark_entered(self) -> None:
        """Record that this process was started with ``--worktree``."""
        self._entered = True

    def release(self) -> Optional[str]:
        """Session teardown: delete a clean unused *agentica* worktree.

        Only runs if this process was started with ``--worktree``. Foreign
        checkouts are left alone. Unique work (dirty or unmerged) is left
        on disk.
        """
        if not self._entered:
            return None
        try:
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
            if worktrees.has_unique_work(entry):
                return None
            from agentica.cli.session_resume import enter_work_dir
            enter_work_dir(main)
            try:
                worktrees.remove(entry, main)
                self._agent_config["work_dir"] = main
                return root
            except worktrees.WorktreeError:
                return None
        except (KeyboardInterrupt, InterruptedError):
            return None
