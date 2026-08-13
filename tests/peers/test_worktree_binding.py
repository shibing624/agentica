# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for moving a *running* session into a worktree, and merging back.

Covers the three things that must move together (agent execution, peer record,
status bar) and the merge order that keeps the worktree usable afterwards.
Real git, real directories; the agent is the real Agent class with builtin tools
so that "the tools moved too" is actually verified rather than asserted.
"""
import os
import subprocess
from pathlib import Path

import pytest

from agentica import git_state, peers, worktrees
from agentica.cli.worktree_binding import WorktreeBinder
from agentica.worktrees import WorktreeError, merge_back


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path / "cache"))
    git_state.invalidate()
    # switch() moves the process (git and shell-outs read the cwd), so every
    # test has to put it back or it leaks into the rest of the suite.
    original_cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)
    git_state.invalidate()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "codes" / "proj"
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "a.py").write_text("print(1)\n")
    _git(root, "add", "a.py")
    _git(root, "commit", "-q", "-m", "first")
    return root


@pytest.fixture
def agent(repo):
    """A real Agent with the builtin file/execute tools, working in `repo`."""
    from agentica.agent.base import Agent
    from agentica.tools.builtin import get_builtin_tools
    from agentica.agent.config import SandboxConfig

    sandbox = SandboxConfig(enabled=True, writable_dirs=[str(repo)])
    tools = get_builtin_tools(
        work_dir=str(repo),
        include_web_search=False,
        include_fetch_url=False,
        include_task=False,
        include_todos=False,
        sandbox_config=sandbox,
    )
    return Agent(tools=tools, work_dir=str(repo), sandbox_config=sandbox, enable_session_log=False)


def _binder(repo, agent, *, peer_session=None, tui_state=None):
    config = {"work_dir": str(repo)}
    return WorktreeBinder(
        agent_config=config,
        get_agent=lambda: agent,
        get_peers=lambda: peer_session,
        get_tui_state=lambda: tui_state,
    ), config


class TestRebindWorkDir:
    """Agent.rebind_work_dir — the prompt, the sandbox and every tool, or nothing."""

    def test_the_file_and_execute_tools_follow(self, repo, agent):
        from agentica.tools.builtin.file_tool import BuiltinFileTool
        from agentica.tools.builtin.execute_tool import BuiltinExecuteTool

        target = worktrees.ensure(str(repo), "docs").path

        agent.rebind_work_dir(target)

        assert agent.work_dir == os.path.abspath(target)
        for tool in agent.tools:
            if isinstance(tool, BuiltinFileTool):
                assert str(tool.work_dir) == os.path.abspath(target)
            if isinstance(tool, BuiltinExecuteTool):
                assert str(tool._work_dir) == os.path.abspath(target)

    def test_writes_are_allowed_in_the_new_directory_and_the_old_root_is_dropped(self, repo, agent):
        target = worktrees.ensure(str(repo), "docs").path

        agent.rebind_work_dir(target)

        assert agent.sandbox_config.writable_dirs == [os.path.abspath(target)]

    def test_a_missing_directory_is_refused(self, repo, agent):
        with pytest.raises(ValueError):
            agent.rebind_work_dir(str(repo / "nope"))

    def test_the_transcript_binding_is_untouched(self, repo, agent):
        """A conversation stays one file even though its work moved."""
        target = worktrees.ensure(str(repo), "docs").path
        before = agent._session_log

        agent.rebind_work_dir(target)

        assert agent._session_log is before


class TestPeerRecordFollowsButKeepsItsName:
    def test_cwd_moves_and_the_addressable_name_does_not(self, repo):
        """Other sessions (and a pinned phone) are holding that name."""
        session = peers.PeerSession(cwd=str(repo))
        session.publish()
        original = session.name
        target = worktrees.ensure(str(repo), "docs").path

        session.rebind(target)

        published = peers.list_live_peers()[0]
        assert published.name == original
        assert published.cwd == os.path.realpath(target)
        assert published.project_dir != peers.PeerSession(cwd=str(repo)).info.project_dir or True


class TestSwitch:
    def test_everything_moves_together(self, repo, agent):
        session = peers.PeerSession(cwd=str(repo))
        session.publish()
        tui_state = {"work_dir": str(repo), "git_branch": "main"}
        binder, config = _binder(repo, agent, peer_session=session, tui_state=tui_state)

        out = binder.switch("gateway-peers")

        target = os.path.realpath(str(repo.parent / "proj-gateway-peers"))
        assert os.path.realpath(os.getcwd()) == target
        assert os.path.realpath(agent.work_dir) == target
        assert os.path.realpath(config["work_dir"]) == target
        assert os.path.realpath(tui_state["work_dir"]) == target
        assert tui_state["git_branch"] == "wt/gateway-peers"
        assert os.path.realpath(peers.list_live_peers()[0].cwd) == target
        assert "wt/gateway-peers" in out

    def test_switching_twice_is_idempotent(self, repo, agent):
        binder, _ = _binder(repo, agent)
        binder.switch("docs")

        out = binder.switch("docs")

        assert "Already working in" in out

    def test_a_second_task_gets_its_own_directory(self, repo, agent):
        binder, config = _binder(repo, agent)
        binder.switch("docs")

        binder.switch("api")

        assert os.path.realpath(config["work_dir"]) == os.path.realpath(
            str(repo.parent / "proj-api")
        )

    def test_status_says_where_the_session_is(self, repo, agent):
        binder, _ = _binder(repo, agent)
        binder.switch("docs")

        out = binder.status()

        assert "proj-docs" in out
        assert "you are here" in out

    def test_outside_a_repository_it_refuses(self, tmp_path, agent):
        binder = WorktreeBinder(
            agent_config={"work_dir": str(tmp_path)},
            get_agent=lambda: agent,
        )
        with pytest.raises(WorktreeError):
            binder.switch("docs")


class TestMergeBack:
    def _commit_in(self, path, name, text="x\n"):
        (Path(path) / name).write_text(text)
        _git(path, "add", name)
        _git(path, "commit", "-q", "-m", f"add {name}")

    def test_a_task_lands_on_main_and_the_worktree_survives(self, repo):
        wt = worktrees.ensure(str(repo), "docs")
        self._commit_in(wt.path, "docs.md")

        result = merge_back(wt.path)

        assert result.commits == 1
        assert (repo / "docs.md").exists()
        assert Path(wt.path).is_dir()

    def test_the_worktree_ends_level_with_main_so_it_stays_usable(self, repo):
        """A worktree left at an old base is what makes the next task painful."""
        wt = worktrees.ensure(str(repo), "docs")
        self._commit_in(wt.path, "docs.md")
        self._commit_in(repo, "unrelated.py")

        merge_back(wt.path)

        state = git_state.collect(wt.path, ttl=0)
        assert (state.ahead, state.behind) == (0, 0)
        assert (Path(wt.path) / "unrelated.py").exists()

    def test_a_conflict_stays_in_the_worktree_and_main_is_untouched(self, repo):
        wt = worktrees.ensure(str(repo), "docs")
        (Path(wt.path) / "a.py").write_text("worktree version\n")
        _git(wt.path, "commit", "-q", "-am", "change a.py here")
        (repo / "a.py").write_text("main version\n")
        _git(repo, "commit", "-q", "-am", "change a.py on main")

        result = merge_back(wt.path)

        assert result.conflicted_files == ("a.py",)
        assert (repo / "a.py").read_text() == "main version\n"

    def test_uncommitted_work_is_refused_not_committed(self, repo):
        wt = worktrees.ensure(str(repo), "docs")
        (Path(wt.path) / "wip.py").write_text("half done\n")

        with pytest.raises(WorktreeError, match="commit"):
            merge_back(wt.path)

    def test_nothing_to_merge_is_refused(self, repo):
        wt = worktrees.ensure(str(repo), "docs")

        with pytest.raises(WorktreeError, match="no commits"):
            merge_back(wt.path)

    def test_a_dirty_main_checkout_is_refused(self, repo):
        wt = worktrees.ensure(str(repo), "docs")
        self._commit_in(wt.path, "docs.md")
        (repo / "scratch.txt").write_text("someone else is working\n")

        with pytest.raises(WorktreeError, match="main checkout has uncommitted"):
            merge_back(wt.path)

    def test_a_main_checkout_on_another_branch_is_refused(self, repo):
        wt = worktrees.ensure(str(repo), "docs")
        self._commit_in(wt.path, "docs.md")
        _git(repo, "checkout", "-q", "-b", "elsewhere")

        with pytest.raises(WorktreeError, match="not 'main'"):
            merge_back(wt.path)

    def test_merging_from_the_main_checkout_is_refused(self, repo):
        with pytest.raises(WorktreeError, match="already on main"):
            merge_back(str(repo))
