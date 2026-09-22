# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Binder creates and disposes worktrees without moving the session.
"""
import shutil
from pathlib import Path

import pytest

from agentica.peers import worktrees
from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.worktree_cmd import _cmd_worktree
from agentica.cli.worktree_binding import WorktreeBinder
from agentica.peers.worktrees import WorktreeError, ensure


def _git(cwd, *args):
    import subprocess

    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(clone_git_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
    return clone_git_repo(tmp_path / "repo")


def _binder(work_dir):
    cfg = {"work_dir": str(work_dir)}
    return WorktreeBinder(agent_config=cfg), cfg


class TestCreateDoesNotMoveTheSession:
    def test_create_returns_the_path_and_leaves_work_dir(self, repo):
        binder, cfg = _binder(repo)
        out = binder.create("docs")
        assert str(repo / ".agentica/worktrees" / "docs") in out
        assert "stayed in" in out
        assert cfg["work_dir"] == str(repo)

    def test_create_refuses_the_base_branch_name(self, repo):
        binder, _ = _binder(repo)
        with pytest.raises(WorktreeError, match="main checkout"):
            binder.create("main")
        assert not Path(repo, ".agentica/worktrees/main").exists()


class TestMergeAndRemoveByName:
    def test_merge_lands_and_deletes_without_moving(self, repo):
        binder, cfg = _binder(repo)
        binder.create("docs")
        wt = worktrees.find(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "add feature")

        out = binder.merge("docs")
        assert "Merged" in out or "already had" in out
        assert cfg["work_dir"] == str(repo)
        assert (repo / "feature.py").is_file()
        assert not Path(wt.path).exists()

    def test_remove_by_name_leaves_the_session(self, repo):
        binder, cfg = _binder(repo)
        binder.create("docs")
        out = binder.remove("docs")
        assert "Removed" in out
        assert cfg["work_dir"] == str(repo)
        assert worktrees.find(str(repo), "docs") is None

    def test_remove_unknown_name_raises(self, repo):
        binder, _ = _binder(repo)
        with pytest.raises(WorktreeError, match="no worktree"):
            binder.remove("nope")

    def test_remove_clears_a_gone_sibling_so_new_can_reuse_the_name(
        self, repo, monkeypatch
    ):
        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.SIBLING_ROOT)
        binder, cfg = _binder(repo)
        binder.create("docs")
        entry = worktrees.find(str(repo), "docs")
        assert entry is not None
        shutil.rmtree(entry.path)

        with pytest.raises(WorktreeError, match="not a checkout"):
            binder.merge("docs")
        with pytest.raises(WorktreeError, match="not a checkout"):
            binder.create("docs")

        out = binder.remove("docs")
        assert "Removed" in out
        assert cfg["work_dir"] == str(repo)
        again = binder.create("docs")
        assert str(repo.parent / f"{repo.name}-docs") in again
        assert Path(repo.parent / f"{repo.name}-docs").is_dir()


class TestReleaseOnlyAfterEntering:
    def test_release_skips_git_when_session_never_entered_a_worktree(self, repo, monkeypatch):
        wt = ensure(str(repo), "docs")
        binder, _ = _binder(repo)
        probed = []
        monkeypatch.setattr(worktrees, "is_git_repo", lambda *_a, **_k: probed.append(True) or True)
        assert binder.release() is None
        assert probed == []
        assert Path(wt.path).is_dir()

    def test_release_leaves_dirty_worktree(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "dirty.py").write_text("nope\n")
        binder, _ = _binder(wt.path)
        binder.mark_entered()
        assert binder.release() is None
        assert Path(wt.path).is_dir()

    def test_release_swallows_getcwd_eintr_when_entered(self, monkeypatch):
        def boom():
            raise InterruptedError(4, "Interrupted system call")

        monkeypatch.setattr("os.getcwd", boom)
        binder, _ = _binder("/unused")
        binder.mark_entered()
        binder._agent_config.clear()
        assert binder.release() is None


class TestSlashDispatch:
    def test_new_without_a_name_prints_usage(self, capsys):
        class Binder:
            def create(self, name, *, base=None):
                raise AssertionError("must not create")

        ctx = CommandContext(agent_config={}, current_agent=None, worktree_binder=Binder())
        _cmd_worktree(ctx, "new")
        assert "Usage" in capsys.readouterr().out

    def test_main_is_refused(self, capsys):
        ctx = CommandContext(
            agent_config={}, current_agent=None, worktree_binder=object()
        )
        _cmd_worktree(ctx, "main")
        assert "does not move" in capsys.readouterr().out
