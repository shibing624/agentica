# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for agentica/git_state.py — the git position a session publishes.

Real repositories in tmp dirs, real git. The point of this module is that
another session can answer "are you behind main?" and "did you touch that
file?" without asking, so the tests are about what a peer would read.
"""
import subprocess

import pytest

from agentica import git_state
from agentica.git_state import GitState, collect


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture(autouse=True)
def no_cache_between_tests():
    git_state.invalidate()
    yield
    git_state.invalidate()


@pytest.fixture
def repo(tmp_path):
    """A repo on `main` with one commit."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "a.py").write_text("print(1)\n")
    _git(root, "add", "a.py")
    _git(root, "commit", "-q", "-m", "first")
    return root


class TestOutsideGit:
    def test_a_plain_directory_reports_nothing(self, tmp_path):
        state = collect(str(tmp_path))
        assert state == GitState()
        assert not state.known
        assert state.summary() == ""

    def test_a_missing_directory_does_not_raise(self, tmp_path):
        assert collect(str(tmp_path / "nope")) == GitState()


class TestBranchAndHead:
    def test_branch_and_short_head_are_reported(self, repo):
        state = collect(str(repo))
        assert state.branch == "main"
        assert len(state.head_sha) >= 7
        assert state.known

    def test_a_clean_checkout_says_clean(self, repo):
        assert collect(str(repo)).dirty_count == 0
        assert "clean" in collect(str(repo)).summary()


class TestDirtyFiles:
    def test_modified_and_untracked_paths_are_both_reported(self, repo):
        (repo / "a.py").write_text("print(2)\n")
        (repo / "b.py").write_text("new\n")

        state = collect(str(repo))

        assert state.dirty_files == ("a.py", "b.py")
        assert state.dirty_count == 2

    def test_the_list_is_capped_but_the_count_is_not(self, repo):
        for i in range(git_state.MAX_DIRTY_FILES + 5):
            (repo / f"f{i:02d}.py").write_text("x\n")

        state = collect(str(repo))

        assert len(state.dirty_files) == git_state.MAX_DIRTY_FILES
        assert state.dirty_count == git_state.MAX_DIRTY_FILES + 5
        assert "+5 more" in state.dirty_line()

    def test_a_rename_reports_the_new_path(self, repo):
        _git(repo, "mv", "a.py", "renamed.py")

        assert collect(str(repo)).dirty_files == ("renamed.py",)


class TestDistanceFromTheBaseBranch:
    def test_a_branch_reports_ahead_and_behind_local_main(self, repo):
        """The base is local `main`: a peer's unpushed commit is exactly the one
        about to collide with yours, and origin/main cannot see it."""
        _git(repo, "checkout", "-q", "-b", "feature")
        (repo / "c.py").write_text("c\n")
        _git(repo, "add", "c.py")
        _git(repo, "commit", "-q", "-m", "on feature")
        _git(repo, "checkout", "-q", "main")
        (repo / "d.py").write_text("d\n")
        _git(repo, "add", "d.py")
        _git(repo, "commit", "-q", "-m", "on main")
        _git(repo, "checkout", "-q", "feature")

        state = collect(str(repo))

        assert (state.base_ref, state.ahead, state.behind) == ("main", 1, 1)
        assert "+1/-1 vs main" in state.summary()

    def test_a_session_on_main_has_no_local_base_to_compare_against(self, repo):
        """Nothing to report rather than comparing main with itself."""
        state = collect(str(repo))
        assert state.base_ref == ""
        assert "vs" not in state.summary()

    def test_master_is_used_when_there_is_no_main(self, tmp_path):
        root = tmp_path / "old"
        root.mkdir()
        _git(root, "init", "-q", "-b", "master")
        _git(root, "config", "user.email", "t@example.com")
        _git(root, "config", "user.name", "T")
        (root / "a").write_text("a\n")
        _git(root, "add", "a")
        _git(root, "commit", "-q", "-m", "first")
        _git(root, "checkout", "-q", "-b", "wt/x")

        assert collect(str(root)).base_ref == "master"


class TestCaching:
    def test_a_second_call_within_the_ttl_does_not_re_read_the_repo(self, repo):
        first = collect(str(repo))
        (repo / "b.py").write_text("new\n")

        assert collect(str(repo)) == first

    def test_invalidate_makes_the_next_call_see_the_change(self, repo):
        collect(str(repo))
        (repo / "b.py").write_text("new\n")

        git_state.invalidate(str(repo))

        assert collect(str(repo)).dirty_files == ("b.py",)

    def test_a_zero_ttl_always_re_reads(self, repo):
        collect(str(repo), ttl=0)
        (repo / "b.py").write_text("new\n")

        assert collect(str(repo), ttl=0).dirty_files == ("b.py",)


class TestPublishedThroughPresence:
    """What another session actually reads in `list_agents`."""

    def test_a_peer_listing_shows_branch_distance_and_dirty_paths(self, repo, tmp_path, monkeypatch):
        from agentica import peers
        from agentica.peers import PeerSession, list_live_peers

        monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path / "cache"))
        _git(repo, "checkout", "-q", "-b", "wt/feature")
        (repo / "a.py").write_text("changed\n")
        state = collect(str(repo))

        session = PeerSession(cwd=str(repo))
        session.publish(
            git_branch=state.branch,
            head_sha=state.head_sha,
            base_ref=state.base_ref,
            ahead=state.ahead,
            behind=state.behind,
            dirty_files=list(state.dirty_files),
            dirty_count=state.dirty_count,
        )

        described = list_live_peers()[0].describe()
        assert "wt/feature" in described
        assert "1 dirty" in described
        assert "dirty: a.py" in described

    def test_a_record_from_an_older_agentica_still_lists(self, tmp_path, monkeypatch):
        """No dirty fields published: report the branch, and never claim clean.

        "clean" is what another session reads before deciding it is safe to
        rebase; an absent field must not be able to say it.
        """
        from agentica import peers
        from agentica.peers import PeerInfo

        monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path / "cache"))
        info = PeerInfo.from_dict({
            "peer_id": "abc", "name": "old-a1", "pid": 1,
            "cwd": str(tmp_path), "git_branch": "main",
        })

        described = info.describe()
        assert "git: main" in described
        assert "clean" not in described
        assert "dirty:" not in described
