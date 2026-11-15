# -*- coding: utf-8 -*-
"""git worktree orchestration as logic: argv decisions, porcelain parsing, guards.

The old suite created a real repository and real worktrees for every scenario.
That is environment risk (git, hooks, filesystem, stale registrations) for a
layer whose decisions are string handling plus "which argv do we run". ``_git``
is stubbed; real git is exercised by using the product, not by every test run.
"""
import os
from pathlib import Path

import pytest

from agentica.git import worktree as wt
from agentica.git.worktree import MergeResult, Worktree, WorktreeError

PORCELAIN = """worktree /repo
HEAD 1234567890abcdef
branch refs/heads/main

worktree /repo/.worktrees/docs
HEAD abcdef1234567890
branch refs/heads/wt/docs

worktree /repo/.worktrees/gone
HEAD deadbeefdeadbeef
detached
locked stale registration
"""


def _stub_git(monkeypatch, mapping, calls=None):
    def fake(args, cwd, *, check=True):
        if calls is not None:
            calls.append((tuple(args), cwd))
        value = mapping.get(tuple(args), "")
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(wt, "_git", fake)


def _entries(monkeypatch):
    _stub_git(monkeypatch, {("worktree", "list", "--porcelain"): PORCELAIN})
    monkeypatch.setattr(wt, "main_root", lambda cwd: "/repo")
    monkeypatch.setattr(wt, "_directory_is_worktree_root", lambda path: True)


# ---------------------------------------------------------------- naming


def test_slug_normalises_a_task_name():
    assert wt.slug("Fix Parser/Bug!") == "fix-parser-bug"
    assert wt.slug("  Docs  ") == "docs"


def test_slug_rejects_reserved_and_empty_names():
    with pytest.raises(WorktreeError, match="main checkout"):
        wt.slug("main")
    with pytest.raises(WorktreeError, match="at least one letter"):
        wt.slug("---")


def test_branch_for_uses_the_namespace():
    assert wt.branch_for("Docs") == f"{wt.BRANCH_PREFIX}docs"


def test_worktree_path_defaults_into_the_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "main_root", lambda cwd: str(tmp_path))
    monkeypatch.setattr(wt, "_configured_root", lambda: wt.DEFAULT_ROOT)
    assert wt.worktree_path(str(tmp_path), "docs") == str(tmp_path / wt.DEFAULT_ROOT / "docs")


def test_worktree_path_supports_sibling_and_absolute_roots(monkeypatch, tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    monkeypatch.setattr(wt, "main_root", lambda cwd: str(repo))
    monkeypatch.setattr(wt, "_configured_root", lambda: wt.SIBLING_ROOT)
    assert wt.worktree_path(str(repo), "docs") == str(tmp_path / "proj-docs")
    absolute = tmp_path / "trees"
    monkeypatch.setattr(wt, "_configured_root", lambda: str(absolute))
    assert wt.worktree_path(str(repo), "docs") == str(absolute / "proj" / "docs")


# ---------------------------------------------------------------- listing


def test_list_worktrees_parses_porcelain_main_first(monkeypatch):
    _entries(monkeypatch)
    entries = wt.list_worktrees("/repo")
    assert [entry.name for entry in entries] == ["repo", "docs", "gone"]
    assert entries[0].is_main is True
    assert entries[1].branch_short == "wt/docs"
    detached = entries[2]
    assert detached.branch == ""
    assert detached.locked is True
    assert detached.lock_reason == "stale registration"


def test_find_matches_by_branch_when_the_path_moved(monkeypatch):
    _entries(monkeypatch)
    monkeypatch.setattr(wt, "worktree_path", lambda cwd, name: "/elsewhere/docs")
    found = wt.find("/repo", "docs")
    assert found is not None and found.path == "/repo/.worktrees/docs"


def test_find_returns_none_for_an_unknown_name(monkeypatch):
    _entries(monkeypatch)
    monkeypatch.setattr(wt, "worktree_path", lambda cwd, name: "/repo/.worktrees/nope")
    assert wt.find("/repo", "nope") is None


# ---------------------------------------------------------------- ensure


def test_ensure_refuses_outside_a_repository(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "is_git_repo", lambda cwd: False)
    with pytest.raises(WorktreeError, match="not inside a git repository"):
        wt.ensure(str(tmp_path), "docs")


def test_ensure_refuses_a_registration_that_is_not_a_checkout(monkeypatch, tmp_path):
    monkeypatch.setattr(wt, "is_git_repo", lambda cwd: True)
    monkeypatch.setattr(
        wt, "find", lambda cwd, name: Worktree(name=name, path=str(tmp_path / "x"), branch="", exists=False)
    )
    with pytest.raises(WorktreeError, match="registered as a worktree"):
        wt.ensure(str(tmp_path), "docs")


def test_ensure_reuses_a_healthy_checkout(monkeypatch, tmp_path):
    entry = Worktree(name="docs", path=str(tmp_path / "docs"), branch="refs/heads/wt/docs")
    monkeypatch.setattr(wt, "is_git_repo", lambda cwd: True)
    monkeypatch.setattr(wt, "find", lambda cwd, name: entry)
    monkeypatch.setattr(wt, "main_root", lambda cwd: str(tmp_path))
    monkeypatch.setattr(wt, "_self_ignore", lambda *a: None)
    monkeypatch.setattr(wt, "link_ignored", lambda *a: ["AGENTS.md"])
    out = wt.ensure(str(tmp_path), "docs")
    assert out.path == entry.path
    assert out.linked == ("AGENTS.md",)


def test_ensure_creates_a_new_branch_when_the_branch_does_not_exist(monkeypatch, tmp_path):
    calls = []
    path = str(tmp_path / ".worktrees" / "docs")
    monkeypatch.setattr(wt, "is_git_repo", lambda cwd: True)
    created = Worktree(name="docs", path=path, branch="refs/heads/wt/docs")
    found = iter([None, created])
    monkeypatch.setattr(wt, "find", lambda cwd, name: next(found))
    monkeypatch.setattr(wt, "worktree_path", lambda cwd, name: path)
    monkeypatch.setattr(wt, "default_base", lambda cwd: "main")
    monkeypatch.setattr(wt, "main_root", lambda cwd: str(tmp_path))
    monkeypatch.setattr(wt, "_self_ignore", lambda *a: None)
    monkeypatch.setattr(wt, "link_ignored", lambda *a: [])
    monkeypatch.setattr(wt, "_invalidate_nested", lambda *a: None)
    _stub_git(
        monkeypatch,
        {
            ("rev-parse", "--verify", "--quiet", "wt/docs"): "",
            ("worktree", "add", "-b", "wt/docs", path, "main"): "",
        },
        calls=calls,
    )
    out = wt.ensure(str(tmp_path), "docs")
    assert out.path == path
    assert ("worktree", "add", "-b", "wt/docs", path, "main") in [c[0] for c in calls]


def test_ensure_rechecks_out_a_branch_that_outlived_its_checkout(monkeypatch, tmp_path):
    calls = []
    path = str(tmp_path / ".worktrees" / "docs")
    monkeypatch.setattr(wt, "is_git_repo", lambda cwd: True)
    created = Worktree(name="docs", path=path, branch="refs/heads/wt/docs")
    found = iter([None, created])
    monkeypatch.setattr(wt, "find", lambda cwd, name: next(found))
    monkeypatch.setattr(wt, "worktree_path", lambda cwd, name: path)
    monkeypatch.setattr(wt, "default_base", lambda cwd: "main")
    monkeypatch.setattr(wt, "main_root", lambda cwd: str(tmp_path))
    monkeypatch.setattr(wt, "_self_ignore", lambda *a: None)
    monkeypatch.setattr(wt, "link_ignored", lambda *a: [])
    monkeypatch.setattr(wt, "_invalidate_nested", lambda *a: None)
    _stub_git(
        monkeypatch,
        {
            ("rev-parse", "--verify", "--quiet", "wt/docs"): "abc123",
            ("worktree", "add", path, "wt/docs"): "",
        },
        calls=calls,
    )
    wt.ensure(str(tmp_path), "docs")
    assert ("worktree", "add", path, "wt/docs") in [c[0] for c in calls]


# ------------------------------------------------------------ merge guards


def _merge_stubs(monkeypatch, overrides=None):
    git = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "wt/docs",
        ("status", "--porcelain"): "",
        ("rev-list", "--count", "main..wt/docs"): "0",
        ("rev-parse", "--short", "HEAD"): "abc123",
    }
    git.update(overrides or {})
    _stub_git(monkeypatch, git)
    monkeypatch.setattr(wt, "current_root", lambda cwd: "/repo/.worktrees/docs")
    monkeypatch.setattr(wt, "main_root", lambda cwd: "/repo")
    monkeypatch.setattr(wt, "default_base", lambda cwd: "main")


def test_merge_back_refuses_a_detached_head(monkeypatch):
    _merge_stubs(monkeypatch, {("rev-parse", "--abbrev-ref", "HEAD"): "HEAD"})
    with pytest.raises(WorktreeError, match="detached HEAD"):
        wt.merge_back("/repo/.worktrees/docs")


def test_merge_back_refuses_the_main_checkout(monkeypatch):
    _merge_stubs(monkeypatch)
    monkeypatch.setattr(wt, "main_root", lambda cwd: "/repo/.worktrees/docs")
    with pytest.raises(WorktreeError, match="main checkout"):
        wt.merge_back("/repo/.worktrees/docs")


def test_merge_back_refuses_uncommitted_work(monkeypatch):
    _merge_stubs(monkeypatch, {("status", "--porcelain"): " M file.py\n"})
    with pytest.raises(WorktreeError, match="commit .* first"):
        wt.merge_back("/repo/.worktrees/docs")


def test_merge_back_reports_already_merged_without_touching_the_index(monkeypatch):
    _merge_stubs(monkeypatch)
    result = wt.merge_back("/repo/.worktrees/docs")
    assert result.already_merged is True
    assert result.commits == 0
    assert result.conflicted is False


def test_merge_back_reports_conflicts_without_merging_into_base(monkeypatch):
    _merge_stubs(
        monkeypatch,
        {
            ("rev-list", "--count", "main..wt/docs"): "2",
            ("merge", "--no-edit", "main"): "CONFLICT",
            ("diff", "--name-only", "--diff-filter=U"): "a.py\nb.py\n",
        },
    )
    result = wt.merge_back("/repo/.worktrees/docs")
    assert result.conflicted is True
    assert result.conflicted_files == ("a.py", "b.py")
    assert result.merged_sha == ""


def test_merge_result_conflicted_flag():
    assert MergeResult(branch="b", base="main", commits=1, merged_sha="x").conflicted is False
    assert MergeResult(
        branch="b", base="main", commits=1, merged_sha="", conflicted_files=("a.py",)
    ).conflicted is True
