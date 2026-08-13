# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Search tools must not return a worktree's copy of every file.

With ``settings.worktree.root`` inside the repository (``.agentica/worktrees``),
a second full checkout lives under the project. The noise here is the small
problem; the large one is an edit landing in the copy.
"""
import asyncio
import subprocess

import pytest

from agentica.tools.builtin.file_tool import BuiltinFileTool


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def repo_with_in_repo_worktree(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "code.py").write_text("MARKER_TOKEN = 1\n")
    _git(root, "add", "code.py")
    _git(root, "commit", "-q", "-m", "first")
    _git(root, "worktree", "add", "-q", ".agentica/worktrees/docs", "-b", "wt/docs")
    return root


@pytest.fixture
def repo_with_custom_root(tmp_path):
    """The same, under a name the user chose (`.worktrees`) — a second session in
    this very repo picked exactly that one, so it must not be a special case."""
    root = tmp_path / "proj2"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "code.py").write_text("MARKER_TOKEN = 1\n")
    _git(root, "add", "code.py")
    _git(root, "commit", "-q", "-m", "first")
    _git(root, "worktree", "add", "-q", ".worktrees/docs", "-b", "wt/docs")
    return root


def _tool(root):
    return BuiltinFileTool(work_dir=str(root))


class TestSearchSkipsInRepoWorktrees:
    def test_glob_returns_each_file_once(self, repo_with_in_repo_worktree):
        out = asyncio.run(_tool(repo_with_in_repo_worktree).glob("**/*.py", path="."))

        assert out.count("code.py") == 1
        assert ".agentica" not in out

    def test_grep_returns_each_file_once(self, repo_with_in_repo_worktree):
        out = asyncio.run(
            _tool(repo_with_in_repo_worktree).grep(
                "MARKER_TOKEN", path=".", output_mode="files_with_matches"
            )
        )

        assert out.count("code.py") == 1
        assert ".agentica" not in out

    def test_the_worktree_copy_is_still_readable_when_asked_for_directly(
        self, repo_with_in_repo_worktree
    ):
        """Excluded from *searching*, not from working: a session bound to that
        worktree passes it as its work_dir and must see everything."""
        inner = repo_with_in_repo_worktree / ".agentica/worktrees/docs"

        out = asyncio.run(_tool(inner).glob("**/*.py", path="."))

        assert "code.py" in out

    def test_a_configured_root_name_is_excluded_too(self, repo_with_custom_root, monkeypatch):
        from agentica import worktrees

        monkeypatch.setattr(worktrees, "_configured_root", lambda: ".worktrees")

        out = asyncio.run(_tool(repo_with_custom_root).glob("**/*.py", path="."))

        assert out.count("code.py") == 1
        assert ".worktrees" not in out

    def test_an_unconfigured_name_is_not_excluded(self, repo_with_custom_root, monkeypatch):
        """The set is derived, not a pile of guesses: with no in-repo root
        configured, `.worktrees` is just a directory and stays visible."""
        from agentica import worktrees

        monkeypatch.setattr(worktrees, "_configured_root", lambda: "")

        out = asyncio.run(_tool(repo_with_custom_root).glob("**/*.py", path="."))

        assert out.count("code.py") == 2
