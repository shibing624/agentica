"""SDK callers can use worktrees without product-level skill injection."""
import asyncio
import subprocess
from pathlib import Path

import pytest

from agentica.git import worktree as worktrees
from agentica.git.binding import WorktreeBinder
from agentica.tools.worktree_tool import WorktreeTool


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "sdk@example.com")
    _git(root, "config", "user.name", "SDK")
    (root / "app.py").write_text("print(1)\n")
    _git(root, "add", "app.py")
    _git(root, "commit", "-q", "-m", "first")
    monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
    return root


def test_sdk_can_create_a_worktree_with_the_core_api(repo):
    wt = worktrees.ensure(str(repo), "sdk-task", base="main")

    assert Path(wt.path).is_dir()
    assert worktrees.find(str(repo), "sdk-task").path == wt.path
    assert Path(wt.path, "app.py").read_text() == "print(1)\n"
    assert worktrees.is_git_repo(wt.path)
    worktrees.remove(wt, str(repo))


def test_sdk_can_mount_the_worktree_tool_explicitly(repo):
    binder = WorktreeBinder(agent_config={"work_dir": str(repo)})
    tool = WorktreeTool(binder)

    out = asyncio.run(tool.worktree(action="new", name="sdk-tool"))

    assert f"Worktree ready at {repo / '.worktrees' / 'sdk-tool'}" in out
    assert "this session stayed in" in out
    assert Path(repo / ".worktrees" / "sdk-tool").is_dir()