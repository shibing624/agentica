# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Search tools skip nested checkout copies.

The skip list comes from ``nested_worktrees``; these tests stub that so they
do not pay for ``git worktree add`` on every run.
"""
import asyncio
import os
from pathlib import Path

from agentica.tools.builtin.file_tool import BuiltinFileTool


def _tool(root):
    return BuiltinFileTool(work_dir=str(root))


def _tree_with_nested_copy(tmp_path, nested_rel=".agentica/worktrees/docs"):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "code.py").write_text("MARKER_TOKEN = 1\n")
    nested = root / nested_rel
    nested.mkdir(parents=True)
    (nested / "code.py").write_text("MARKER_TOKEN = 1\n")
    return root, nested


class TestSearchSkipsNestedCheckouts:
    def test_glob_returns_each_file_once(self, tmp_path, monkeypatch):
        root, nested = _tree_with_nested_copy(tmp_path)
        monkeypatch.setattr(
            "agentica.worktrees.nested_worktrees",
            lambda *a, **k: (os.path.realpath(str(nested)),),
        )

        out = asyncio.run(_tool(root).glob("**/*.py", path="."))

        assert out.count("code.py") == 1
        assert ".agentica" not in out

    def test_grep_returns_each_file_once(self, tmp_path, monkeypatch):
        root, nested = _tree_with_nested_copy(tmp_path)
        monkeypatch.setattr(
            "agentica.worktrees.nested_worktrees",
            lambda *a, **k: (os.path.realpath(str(nested)),),
        )

        out = asyncio.run(
            _tool(root).grep("MARKER_TOKEN", path=".")
        )

        assert out.count("code.py") == 1
        assert ".agentica" not in out

    def test_the_nested_copy_is_still_readable_when_it_is_the_work_dir(self, tmp_path, monkeypatch):
        root, nested = _tree_with_nested_copy(tmp_path)
        monkeypatch.setattr(
            "agentica.worktrees.nested_worktrees",
            lambda *a, **k: (os.path.realpath(str(nested)),),
        )

        out = asyncio.run(_tool(nested).glob("**/*.py", path="."))

        assert "code.py" in out

    def test_a_custom_nested_name_is_excluded_the_same_way(self, tmp_path, monkeypatch):
        root, nested = _tree_with_nested_copy(tmp_path, nested_rel=".worktrees/docs")
        monkeypatch.setattr(
            "agentica.worktrees.nested_worktrees",
            lambda *a, **k: (os.path.realpath(str(nested)),),
        )

        out = asyncio.run(_tool(root).glob("**/*.py", path="."))

        assert out.count("code.py") == 1
        assert ".worktrees" not in out


class TestUnstubbedNestedSelfExclude:
    """The skip list comes from git, including this process's own checkout.

    Stubbing ``nested_worktrees`` hid the case that used to make glob from
    inside a worktree return nothing: the tree's own path was on the skip
    list.
    """

    def test_glob_from_main_skips_the_nested_tree(self, clone_git_repo, tmp_path, monkeypatch):
        from agentica import worktrees
        from agentica.worktrees import ensure

        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
        repo = clone_git_repo(tmp_path / "repo")
        wt = ensure(str(repo), "docs")
        (repo / "only_in_main.py").write_text("MAIN = 1\n")
        (Path(wt.path) / "only_in_tree.py").write_text("TREE = 1\n")

        out = asyncio.run(_tool(repo).glob("**/*.py", path="."))
        assert "only_in_main.py" in out
        assert "only_in_tree.py" not in out

    def test_glob_from_inside_the_worktree_still_sees_its_files(
        self, clone_git_repo, tmp_path, monkeypatch
    ):
        from agentica import worktrees
        from agentica.worktrees import ensure

        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
        repo = clone_git_repo(tmp_path / "repo")
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "only_in_tree.py").write_text("TREE = 1\n")

        out = asyncio.run(_tool(wt.path).glob("**/*.py", path="."))
        assert "only_in_tree.py" in out

    def test_glob_with_path_pointed_at_the_tree_sees_its_files(
        self, clone_git_repo, tmp_path, monkeypatch
    ):
        from agentica import worktrees
        from agentica.worktrees import ensure

        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
        repo = clone_git_repo(tmp_path / "repo")
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "only_in_tree.py").write_text("TREE = 1\n")

        out = asyncio.run(_tool(repo).glob("**/*.py", path=wt.path))
        assert "only_in_tree.py" in out
