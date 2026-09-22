# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the `worktree` tool's dispatch.
"""
import asyncio

import pytest

from agentica.tools.worktree_tool import WorktreeTool
from agentica.peers.worktrees import WorktreeError


class FakeBinder:
    def __init__(self, *, fail=None):
        self.calls = []
        self._fail = fail

    def status(self):
        self.calls.append(("status", None))
        return "STATUS"

    def create(self, name, *, base=None):
        self.calls.append(("create", (name, base)))
        if self._fail:
            raise WorktreeError(self._fail)
        return f"CREATED {name}"

    def merge(self, name, *, base=None):
        self.calls.append(("merge", (name, base)))
        if self._fail:
            raise WorktreeError(self._fail)
        return f"MERGED {name}"

    def remove(self, name):
        self.calls.append(("remove", name))
        if self._fail:
            raise WorktreeError(self._fail)
        return f"REMOVED {name}"


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

    @pytest.mark.parametrize("action", ["new", "use", "switch", "bind", "create"])
    def test_create_spellings(self, action):
        binder = FakeBinder()
        assert _run(WorktreeTool(binder), action=action, name="docs") == "CREATED docs"
        assert binder.calls == [("create", ("docs", None))]

    def test_a_base_branch_is_passed_through(self, ):
        binder = FakeBinder()
        _run(WorktreeTool(binder), action="new", name="docs", base="release")
        assert binder.calls == [("create", ("docs", "release"))]

    def test_main_spellings_do_not_move(self):
        for action in ("main", "home"):
            binder = FakeBinder()
            out = _run(WorktreeTool(binder), action=action)
            assert "does not move" in out
            assert binder.calls == []

    def test_merge_requires_a_name(self):
        binder = FakeBinder()
        out = _run(WorktreeTool(binder), action="merge")
        assert "name" in out
        assert binder.calls == []

    def test_merge_spellings(self):
        for action in ("merge", "merge-back", "land"):
            binder = FakeBinder()
            assert _run(WorktreeTool(binder), action=action, name="docs") == "MERGED docs"
            assert binder.calls == [("merge", ("docs", None))]

    def test_remove_spellings(self):
        for action in ("remove", "delete", "drop"):
            binder = FakeBinder()
            assert _run(WorktreeTool(binder), action=action, name="docs") == "REMOVED docs"
            assert binder.calls == [("remove", "docs")]

    def test_an_unknown_action_lists_the_real_ones(self):
        out = _run(WorktreeTool(FakeBinder()), action="explode")
        assert "status" in out and "new" in out and "merge" in out
        assert "remove" in out
        assert "main" not in out or "Use status" in out


class TestRefusals:
    def test_new_without_a_name_says_what_to_do_instead(self):
        binder = FakeBinder()
        out = _run(WorktreeTool(binder), action="new")
        assert "name" in out
        assert binder.calls == []

    def test_a_refusal_keeps_the_reason(self):
        out = _run(
            WorktreeTool(FakeBinder(fail="/tmp/x already exists but is not a worktree")),
            action="new",
            name="docs",
        )
        assert "already exists but is not a worktree" in out


class TestInstructions:
    def test_the_tool_does_not_inject_a_system_prompt(self):
        assert WorktreeTool(FakeBinder()).get_system_prompt() is None
