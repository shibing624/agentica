# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Worktree layout helpers plus the create / lock / merge / remove lifecycle.

Layout tests stub ``main_root``. Lifecycle tests use a copied real git repo —
``git worktree add`` on a one-file tree is cheap, and the safety checks are
the behaviour that used to be wrong.
"""
from pathlib import Path

import pytest

from agentica import worktrees
from agentica.worktrees import WorktreeError, ensure, slug


class TestSlug:
    def test_spaces_and_case_are_normalised(self):
        assert slug("Gateway Peers") == "gateway-peers"

    def test_a_name_with_nothing_usable_is_refused(self):
        with pytest.raises(WorktreeError):
            slug("///")


class TestLayout:
    def test_default_is_inside_the_checkout(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
        assert worktrees.worktree_path(str(repo), "docs") == str(
            repo / ".agentica/worktrees" / "docs"
        )

    def test_sibling_is_opt_in(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.SIBLING_ROOT)
        assert worktrees.worktree_path(str(repo), "docs") == str(repo.parent / "proj-docs")

    def test_an_absolute_worktree_root_namespaces_by_repository(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        monkeypatch.setattr(worktrees, "_configured_root", lambda: str(tmp_path / "wt"))

        assert worktrees.worktree_path(str(repo), "docs") == str(tmp_path / "wt" / "proj" / "docs")

    def test_a_relative_worktree_root_resolves_inside_the_checkout(self, tmp_path, monkeypatch):
        repo = tmp_path / "codes" / "proj"
        monkeypatch.setattr(worktrees, "main_root", lambda _cwd: str(repo))
        monkeypatch.setattr(worktrees, "_configured_root", lambda: ".agentica/worktrees")

        assert worktrees.worktree_path(str(repo), "docs") == str(
            repo / ".agentica/worktrees" / "docs"
        )

    def test_the_branch_is_prefixed(self):
        assert worktrees.branch_for("docs") == "wt/docs"


class TestEnsureRefuses:
    def test_outside_a_repository_it_refuses(self, tmp_path):
        with pytest.raises(WorktreeError, match="not inside a git repository"):
            ensure(str(tmp_path), "docs")


def _git(cwd, *args):
    import subprocess

    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )


@pytest.fixture
def repo(clone_git_repo, tmp_path, monkeypatch):
    monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
    return clone_git_repo(tmp_path / "repo")


class TestLifecycle:
    def test_ensure_lands_under_agentica_worktrees(self, repo):
        wt = ensure(str(repo), "docs")
        assert Path(wt.path) == repo / ".agentica/worktrees" / "docs"
        assert (repo / ".agentica/worktrees" / ".gitignore").read_text().strip().endswith("*")
        assert wt.branch_short == "wt/docs"
        assert (Path(wt.path) / "a.py").is_file()

    def test_ensure_reuses_an_in_progress_worktree(self, repo):
        first = ensure(str(repo), "docs")
        (Path(first.path) / "wip.txt").write_text("keep me\n")
        second = ensure(str(repo), "docs")
        assert Path(second.path) == Path(first.path)
        assert (Path(second.path) / "wip.txt").read_text() == "keep me\n"

    def test_merge_then_remove_deletes_the_checkout_and_the_branch(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "add feature")

        result = worktrees.merge_back(wt.path)
        assert not result.conflicted
        assert (repo / "feature.py").is_file()

        worktrees.remove(wt.path)
        assert not Path(wt.path).exists()
        listed = worktrees.list_worktrees(str(repo))
        assert [e.path for e in listed if not e.is_main] == []
        branches = _git_output(repo, "branch", "--list", "wt/docs")
        assert branches.strip() == ""

    def test_remove_refuses_uncommitted_work(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "dirty.py").write_text("nope\n")
        with pytest.raises(WorktreeError, match="uncommitted"):
            worktrees.remove(wt.path)
        assert Path(wt.path).is_dir()

    def test_remove_refuses_commits_not_on_the_local_base(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "not merged yet")
        with pytest.raises(WorktreeError, match="not.*merged|local base|main"):
            worktrees.remove(wt.path)
        assert Path(wt.path).is_dir()

    def test_remove_of_a_clean_unused_worktree_is_allowed(self, repo):
        wt = ensure(str(repo), "docs")
        worktrees.remove(wt.path)
        assert not Path(wt.path).exists()

    def test_a_live_foreign_lock_blocks_remove(self, repo):
        wt = ensure(str(repo), "docs")
        worktrees.lock(wt.path, reason="agentica pid=1")
        with pytest.raises(WorktreeError, match="locked"):
            worktrees.remove(wt.path)
        assert Path(wt.path).is_dir()

    def test_a_dead_agentica_lock_is_stolen(self, repo):
        wt = ensure(str(repo), "docs")
        worktrees.lock(wt.path, reason="agentica pid=99999999")
        assert worktrees.claim_lock(wt.path) is True
        worktrees.remove(wt.path)
        assert not Path(wt.path).exists()

    def test_a_user_lock_is_not_stolen_or_removed(self, repo):
        wt = ensure(str(repo), "docs")
        worktrees.lock(wt.path, reason="keep this")
        assert worktrees.claim_lock(wt.path) is False
        with pytest.raises(WorktreeError, match="locked"):
            worktrees.remove(wt.path)

    def test_same_name_after_remove_forks_from_current_main(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "add feature")
        worktrees.merge_back(wt.path)
        worktrees.remove(wt.path)

        (repo / "later.py").write_text("after\n")
        _git(repo, "add", "later.py")
        _git(repo, "commit", "-q", "-m", "moved on")

        fresh = ensure(str(repo), "docs")
        assert (Path(fresh.path) / "later.py").is_file()
        assert (Path(fresh.path) / "feature.py").is_file()

    def test_remove_refuses_a_detached_foreign_worktree(self, repo, tmp_path):
        inspect = tmp_path / "inspect"
        _git(repo, "worktree", "add", "--detach", str(inspect), "HEAD")
        with pytest.raises(WorktreeError, match="not an agentica worktree"):
            worktrees.remove(str(inspect))
        assert inspect.is_dir()

    def test_remove_refuses_a_non_wt_branch(self, repo, tmp_path):
        other = tmp_path / "review"
        _git(repo, "worktree", "add", "-b", "review/docs", str(other), "HEAD")
        with pytest.raises(WorktreeError, match="not an agentica worktree"):
            worktrees.remove(str(other))
        assert other.is_dir()
        assert "review/docs" in _git_output(repo, "branch", "--list", "review/docs")

    def test_remove_refuses_when_there_is_no_local_base(self, repo):
        _git(repo, "branch", "-m", "main", "trunk")
        wt = ensure(str(repo), "docs", base="trunk")
        with pytest.raises(WorktreeError, match="no local 'main' or 'master'"):
            worktrees.remove(wt.path)
        assert Path(wt.path).is_dir()

    def test_a_sibling_worktree_is_reused_by_branch_after_the_default_moves_inside(
        self, repo, monkeypatch
    ):
        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.SIBLING_ROOT)
        old = ensure(str(repo), "docs")
        sibling = Path(old.path)
        (sibling / "wip.txt").write_text("from sibling\n")

        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.DEFAULT_ROOT)
        again = ensure(str(repo), "docs")
        assert Path(again.path) == sibling
        assert (sibling / "wip.txt").read_text() == "from sibling\n"


class TestSettings:
    def test_a_flat_worktree_root_key_is_read(self, monkeypatch):
        monkeypatch.setattr(
            "agentica.global_config.get_setting",
            lambda key, default=None: "sibling" if key == "worktree.root" else default,
        )
        assert worktrees._configured_root() == "sibling"

    def test_a_nested_worktree_root_block_is_read(self, monkeypatch):
        monkeypatch.setattr(
            "agentica.global_config.get_setting",
            lambda key, default=None: default,
        )
        monkeypatch.setattr(
            "agentica.global_config.load_global_config",
            lambda: {"settings": {"worktree": {"root": "sibling"}}},
        )
        assert worktrees._configured_root() == "sibling"


def _git_output(cwd, *args):
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )
    return result.stdout
