# -*- coding: utf-8 -*-
"""WorktreeBinder: message/state decisions, with git orchestration stubbed.

The binder's job is to translate "create / merge / remove this task worktree"
into the right `worktrees.*` call and the right sentence for the model, never
moving the session. Real git was tested here before; it is not the unit.
"""
import os
from types import SimpleNamespace

import pytest

from agentica.git import binding
from agentica.git.binding import WorktreeBinder
from agentica.git.worktree import MergeResult, Worktree, WorktreeError


@pytest.fixture
def binder(tmp_path):
    return WorktreeBinder(agent_config={"work_dir": str(tmp_path)})


def test_create_returns_the_path_and_keeps_the_session_put(monkeypatch, binder, tmp_path):
    entry = Worktree(name="docs", path=str(tmp_path / ".worktrees" / "docs"), branch="refs/heads/wt/docs",
                     linked=("AGENTS.md",))
    monkeypatch.setattr(binding.worktrees, "ensure", lambda *a, **k: entry)
    out = binder.create("docs")
    assert entry.path in out
    assert "stayed in" in out
    assert "AGENTS.md" in out
    assert binder.work_dir() == str(tmp_path)


def test_create_propagates_a_refusal(monkeypatch, binder):
    def boom(*a, **k):
        raise WorktreeError("main is the main checkout")

    monkeypatch.setattr(binding.worktrees, "ensure", boom)
    with pytest.raises(WorktreeError, match="main checkout"):
        binder.create("main")


def _entry(tmp_path, name="docs", *, exists=True):
    return Worktree(name=name, path=str(tmp_path / ".worktrees" / name),
                    branch=f"refs/heads/wt/{name}", exists=exists)


def test_merge_without_a_worktree_is_refused(monkeypatch, binder):
    monkeypatch.setattr(binding.worktrees, "find", lambda cwd, name: None)
    with pytest.raises(WorktreeError, match="no worktree named"):
        binder.merge("docs")


def test_merge_refuses_a_registration_that_is_not_a_checkout(monkeypatch, binder, tmp_path):
    monkeypatch.setattr(binding.worktrees, "find", lambda cwd, name: _entry(tmp_path, exists=False))
    with pytest.raises(WorktreeError, match="not a checkout"):
        binder.merge("docs")


def test_merge_conflict_reports_files_and_removes_nothing(monkeypatch, binder, tmp_path):
    removed = []
    monkeypatch.setattr(binding.worktrees, "find", lambda cwd, name: _entry(tmp_path))
    monkeypatch.setattr(
        binding.worktrees, "merge_back",
        lambda path, base=None: MergeResult(branch="wt/docs", base="main", commits=2,
                                            merged_sha="", conflicted_files=("a.py",)),
    )
    monkeypatch.setattr(binding.worktrees, "remove", lambda *a: removed.append(a))
    out = binder.merge("docs")
    assert "conflicted" in out and "a.py" in out
    assert removed == []


def test_merge_clean_lands_then_removes_the_checkout(monkeypatch, binder, tmp_path):
    removed = []
    monkeypatch.setattr(binding.worktrees, "find", lambda cwd, name: _entry(tmp_path))
    monkeypatch.setattr(
        binding.worktrees, "merge_back",
        lambda path, base=None: MergeResult(branch="wt/docs", base="main", commits=1, merged_sha="abc"),
    )
    monkeypatch.setattr(binding.worktrees, "remove", lambda entry, cwd: removed.append(entry))
    monkeypatch.setattr(binding.git_state, "invalidate", lambda: None)
    out = binder.merge("docs")
    assert "Merged 1 commit(s)" in out
    assert "worktree removed" in out
    assert removed


def test_merge_reports_already_merged(monkeypatch, binder, tmp_path):
    monkeypatch.setattr(binding.worktrees, "find", lambda cwd, name: _entry(tmp_path))
    monkeypatch.setattr(
        binding.worktrees, "merge_back",
        lambda path, base=None: MergeResult(branch="wt/docs", base="main", commits=0,
                                            merged_sha="abc", already_merged=True),
    )
    monkeypatch.setattr(binding.worktrees, "remove", lambda *a: None)
    monkeypatch.setattr(binding.git_state, "invalidate", lambda: None)
    assert "already had every commit" in binder.merge("docs")


def test_remove_drops_the_checkout_without_moving(monkeypatch, binder, tmp_path):
    entry = _entry(tmp_path)
    monkeypatch.setattr(binding.worktrees, "find", lambda cwd, name: entry)
    removed = []
    monkeypatch.setattr(binding.worktrees, "remove", lambda e, cwd: removed.append(e))
    out = binder.remove("docs")
    assert "Removed worktree" in out and "stayed in" in out
    assert removed == [entry]


def test_status_says_when_there_is_no_repository(monkeypatch, binder):
    monkeypatch.setattr(binding.worktrees, "is_git_repo", lambda cwd: False)
    assert "not inside a git repository" in binder.status()


def test_status_marks_the_current_tree(monkeypatch, binder, tmp_path):
    entry = Worktree(name="docs", path=str(tmp_path / "docs"), branch="refs/heads/wt/docs")
    monkeypatch.setattr(binding.worktrees, "is_git_repo", lambda cwd: True)
    monkeypatch.setattr(binding.worktrees, "current_root", lambda cwd: entry.path)
    monkeypatch.setattr(binding.worktrees, "list_worktrees", lambda cwd: [entry])
    monkeypatch.setattr(
        binding.git_state, "collect",
        lambda cwd, ttl=0: SimpleNamespace(known=True, summary=lambda: "clean"),
    )
    out = binder.status()
    assert "<- you are here" in out
    assert "clean" in out


def test_release_is_a_no_op_unless_started_with_worktree(binder):
    assert binder.release() is None


def test_release_removes_a_clean_managed_checkout(monkeypatch, binder, tmp_path):
    entry = _entry(tmp_path)
    monkeypatch.setattr(binding.worktrees, "is_git_repo", lambda cwd: True)
    monkeypatch.setattr(binding.worktrees, "current_root", lambda cwd: entry.path)
    monkeypatch.setattr(binding.worktrees, "main_root", lambda cwd: str(tmp_path))
    monkeypatch.setattr(binding.worktrees, "resolve_entry", lambda path: entry)
    monkeypatch.setattr(binding.worktrees, "is_managed", lambda e: True)
    monkeypatch.setattr(binding.worktrees, "has_unique_work", lambda e: False)
    monkeypatch.setattr(binding.worktrees, "remove", lambda e, cwd: None)
    monkeypatch.setattr(binding.os, "chdir", lambda path: None)
    binder.mark_entered()
    assert binder.release() == entry.path
    assert binder.work_dir() == str(tmp_path)
