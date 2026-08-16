# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for agentica/peer_conflicts.py — "someone else has this file dirty".

The warning has to be right about *which* repository and it has to be quiet
enough to keep being read, so that is what these check.
"""
import pytest

from agentica import git_state, peers
from agentica.peer_conflicts import PeerConflictChecker, build_checker


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path / "cache"))
    git_state.invalidate()
    yield
    git_state.invalidate()


@pytest.fixture
def repo(clone_git_repo, tmp_path):
    return clone_git_repo(tmp_path / "codes" / "proj")


def _publish(cwd, *, name):
    """A live peer that publishes its git state, as the CLI heartbeat does."""
    session = peers.PeerSession(cwd=str(cwd), name=name)
    state = git_state.collect(str(cwd), ttl=0)
    session.publish(
        git_branch=state.branch,
        head_sha=state.head_sha,
        repo_root=state.repo_root,
        base_ref=state.base_ref,
        ahead=state.ahead,
        behind=state.behind,
        dirty_files=list(state.dirty_files),
        dirty_count=state.dirty_count,
    )
    return session


class TestWarning:
    def test_a_peer_with_the_same_file_dirty_is_reported(self, repo):
        other = _publish(repo, name="other-a1")
        (repo / "a.py").write_text("their edit\n")
        other.publish(dirty_files=["a.py"], dirty_count=1)
        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()

        note = PeerConflictChecker(me).check(str(repo / "a.py"))

        assert "a.py" in note
        assert "other-a1" in note
        assert "send_message" in note

    def test_a_second_cwd_of_the_same_repo_counts(self, repo, tmp_path):
        """Two sessions, two directories, one repo_root — no real worktree needed."""
        other_cwd = tmp_path / "other-checkout"
        other_cwd.mkdir()
        other = peers.PeerSession(cwd=str(other_cwd), name="other-a1")
        other.publish(
            repo_root=str(repo.resolve()),
            dirty_files=["a.py"],
            dirty_count=1,
        )
        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()

        note = PeerConflictChecker(me).check(str(repo / "a.py"))

        assert "other-a1" in note

    def test_an_unrelated_repository_with_the_same_filename_is_not_reported(
        self, clone_git_repo, tmp_path
    ):
        """Otherwise every README.md on the machine warns forever."""
        mine = clone_git_repo(tmp_path / "codes" / "mine")
        theirs = clone_git_repo(tmp_path / "codes" / "theirs")
        other = _publish(theirs, name="other-a1")
        (theirs / "a.py").write_text("their edit\n")
        other.publish(dirty_files=["a.py"], dirty_count=1)
        me = peers.PeerSession(cwd=str(mine), name="me-b2")
        me.publish()

        assert PeerConflictChecker(me).check(str(mine / "a.py")) == ""

    def test_a_clean_peer_is_not_reported(self, repo):
        _publish(repo, name="other-a1")
        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()

        assert PeerConflictChecker(me).check(str(repo / "a.py")) == ""

    def test_a_different_file_is_not_reported(self, repo):
        other = _publish(repo, name="other-a1")
        other.publish(dirty_files=["b.py"], dirty_count=1)
        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()

        assert PeerConflictChecker(me).check(str(repo / "a.py")) == ""

    def test_it_is_said_once_per_file_and_peer(self, repo):
        other = _publish(repo, name="other-a1")
        other.publish(dirty_files=["a.py"], dirty_count=1)
        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()
        checker = PeerConflictChecker(me)

        first = checker.check(str(repo / "a.py"))
        second = checker.check(str(repo / "a.py"))

        assert first
        assert second == ""

    def test_outside_a_repository_there_is_nothing_to_compare(self, tmp_path):
        me = peers.PeerSession(cwd=str(tmp_path), name="me-b2")
        me.publish()

        assert PeerConflictChecker(me).check(str(tmp_path / "loose.txt")) == ""

    def test_no_presence_means_no_checker(self):
        assert build_checker(None) is None


class TestThroughTheFileTool:
    """The warning has to reach the model, on the write that caused it."""

    def _tool(self, repo, me):
        from agentica.tools.builtin.file_tool import BuiltinFileTool

        return BuiltinFileTool(
            work_dir=str(repo),
            peer_conflict_checker=PeerConflictChecker(me),
        )

    def test_write_file_appends_the_warning(self, repo):
        import asyncio

        other = _publish(repo, name="other-a1")
        other.publish(dirty_files=["a.py"], dirty_count=1)
        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()
        tool = self._tool(repo, me)

        result = asyncio.run(tool.write_file(str(repo / "a.py"), "my edit\n"))

        assert "other-a1" in str(result)
        # The write still happened: this is advice, not a lock.
        assert (repo / "a.py").read_text() == "my edit\n"

    def test_write_file_says_nothing_when_nobody_else_is_editing(self, repo):
        import asyncio

        me = peers.PeerSession(cwd=str(repo), name="me-b2")
        me.publish()
        tool = self._tool(repo, me)

        result = asyncio.run(tool.write_file(str(repo / "b.py"), "mine\n"))

        assert "Another live session" not in str(result)
