# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Binder must not move the session on a refused remove, and must
say so when it shares a locked worktree.
"""
from pathlib import Path

import pytest

from agentica import worktrees
from agentica.cli.worktree_binding import WorktreeBinder
from agentica.worktrees import WorktreeError, ensure


def _git(cwd, *args):
    import subprocess

    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )


class FakeAgent:
    def __init__(self, work_dir):
        self.work_dir = str(work_dir)

    def rebind_work_dir(self, work_dir):
        self.work_dir = str(work_dir)


@pytest.fixture
def repo(clone_git_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
    return clone_git_repo(tmp_path / "repo")


def _binder(work_dir, agent=None):
    agent = agent or FakeAgent(work_dir)
    cfg = {"work_dir": str(work_dir)}
    return WorktreeBinder(
        agent_config=cfg,
        get_agent=lambda: agent,
    ), agent, cfg


class TestReleaseOwnsOnlyWtBranches:
    def test_release_does_not_delete_a_detached_foreign_worktree(self, repo, tmp_path):
        inspect = tmp_path / "inspect"
        _git(repo, "worktree", "add", "--detach", str(inspect), "HEAD")
        binder, _, _ = _binder(inspect)
        assert binder.release() is None
        assert inspect.is_dir()


class TestReleaseKeepsLockOnUniqueWork:
    def test_release_leaves_a_dirty_worktree_locked(self, repo):
        wt = ensure(str(repo), "docs")
        assert worktrees.claim_lock(wt.path) is True
        (Path(wt.path) / "dirty.py").write_text("nope\n")
        binder, _, _ = _binder(wt.path)
        assert binder.release() is None
        entry = worktrees.resolve_entry(wt.path)
        assert entry.locked
        assert Path(wt.path).is_dir()


class TestRemoveValidatesBeforeMoving:
    def test_a_refused_remove_leaves_the_session_in_the_worktree(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "dirty.py").write_text("nope\n")
        agent = FakeAgent(wt.path)
        binder, agent, cfg = _binder(wt.path, agent=agent)

        with pytest.raises(WorktreeError, match="uncommitted"):
            binder.remove()

        assert Path(agent.work_dir).resolve() == Path(wt.path).resolve()
        assert Path(cfg["work_dir"]).resolve() == Path(wt.path).resolve()
        assert Path(wt.path).is_dir()


class TestSwitchReportsSharing:
    def test_a_live_foreign_lock_is_named_in_the_switch_result(self, repo):
        wt = ensure(str(repo), "docs")
        worktrees.lock(wt.path, reason="agentica pid=1")
        binder, _, _ = _binder(repo)

        out = binder.switch("docs")

        assert "pid 1" in out
        assert "sharing" in out.lower()
        assert "index.lock" in out
