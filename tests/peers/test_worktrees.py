# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Fast unit tests for worktree path / slug helpers.

Real ``git worktree add`` coverage used to live here and dominated the suite
(~12s). Layout and naming are what the rest of the feature keys on; creating
a second checkout is git's job and is not re-tested on every run.
"""
import pytest

from agentica import worktrees
from agentica.worktrees import WorktreeError, ensure, slug


class TestSlug:
    def test_spaces_and_case_are_normalised(self):
        assert slug("Gateway Peers") == "gateway-peers"

    def test_a_name_with_nothing_usable_is_refused(self):
        with pytest.raises(WorktreeError):
            slug("///")


class TestLayout:
    def test_a_worktree_is_a_sibling_of_the_main_checkout(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        assert worktrees.worktree_path(str(repo), "docs") == str(repo.parent / "proj-docs")

    def test_an_absolute_worktree_root_namespaces_by_repository(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        monkeypatch.setattr(worktrees, "_configured_root", lambda: str(tmp_path / "wt"))

        assert worktrees.worktree_path(str(repo), "docs") == str(tmp_path / "wt" / "proj" / "docs")

    def test_a_relative_worktree_root_resolves_inside_the_checkout(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        monkeypatch.setattr(worktrees, "_configured_root", lambda: ".agentica/worktrees")

        assert worktrees.worktree_path(str(repo), "docs") == str(
            repo / ".agentica/worktrees" / "docs"
        )

    def test_the_branch_is_prefixed(self):
        assert worktrees.branch_for("docs") == "wt/docs"


class TestEnsureRefuses:
    def test_outside_a_repository_it_refuses(self, tmp_path):
        with pytest.raises(WorktreeError, match="not inside a git repository"):
            ensure(str(tmp_path), "docs")
