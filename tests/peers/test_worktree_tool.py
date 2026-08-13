# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the `worktree` tool's dispatch (agentica/tools/worktree_tool.py).

This is the surface a peer message drives ("切到 gateway-peers 再改"), so what
matters is that every spelling an agent might reach for lands on the right
action, and that a refusal keeps the reason it was refused for.
"""
import asyncio

import pytest

from agentica.tools.worktree_tool import WorktreeTool
from agentica.worktrees import WorktreeError


class FakeBinder:
    def __init__(self, *, fail=None):
        self.calls = []
        self._fail = fail

    def status(self):
        self.calls.append(("status", None))
        return "STATUS"

    def switch(self, name, *, base=None):
        self.calls.append(("switch", (name, base)))
        if self._fail:
            raise WorktreeError(self._fail)
        return f"SWITCHED {name}"

    def merge(self):
        self.calls.append(("merge", None))
        return "MERGED"


def _run(tool, **kwargs):
    return asyncio.run(tool.worktree(**kwargs))


class TestDispatch:
    def test_the_default_action_is_status(self):
        binder = FakeBinder()
        assert _run(WorktreeTool(binder)) == "STATUS"
        assert binder.calls == [("status", None)]

    @pytest.mark.parametrize("action", ["status", "list", "info", ""])
    def test_listing_spellings(self, action):
        assert _run(WorktreeTool(FakeBinder()), action=action) == "STATUS"

    @pytest.mark.parametrize("action", ["use", "switch", "bind", "create", "new"])
    def test_switching_spellings(self, action):
        binder = FakeBinder()
        assert _run(WorktreeTool(binder), action=action, name="docs") == "SWITCHED docs"
        assert binder.calls == [("switch", ("docs", None))]

    def test_a_base_branch_is_passed_through(self):
        binder = FakeBinder()
        _run(WorktreeTool(binder), action="use", name="docs", base="release")
        assert binder.calls == [("switch", ("docs", "release"))]

    def test_merge_spellings(self):
        for action in ("merge", "merge-back", "land"):
            binder = FakeBinder()
            assert _run(WorktreeTool(binder), action=action) == "MERGED"

    def test_an_unknown_action_lists_the_real_ones(self):
        out = _run(WorktreeTool(FakeBinder()), action="delete")
        assert "status" in out and "use" in out and "merge" in out


class TestRefusals:
    def test_use_without_a_name_says_what_to_do_instead(self):
        binder = FakeBinder()
        out = _run(WorktreeTool(binder), action="use")
        assert "name" in out
        assert binder.calls == []

    def test_a_refusal_keeps_the_reason(self):
        """The git-level message tells the user what to do ("move it aside or
        pick another name"); paraphrasing it would lose that."""
        out = _run(
            WorktreeTool(FakeBinder(fail="/tmp/x already exists but is not a worktree")),
            action="use",
            name="docs",
        )
        assert "already exists but is not a worktree" in out


class TestInstructions:
    def test_the_policy_reaches_the_system_prompt(self):
        prompt = WorktreeTool(FakeBinder()).get_system_prompt()
        assert "worktree" in prompt
        assert "list_agents" in prompt
