# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for agentica/worktrees.py — per-task worktrees of one repo.

Real git, real directories. What matters here is the promises the rest of the
feature leans on: reuse never recreates, nothing is ever deleted, paths hang off
the main checkout no matter where the call comes from, and a fresh worktree has
the gitignored files a session needs.
"""
import os
import subprocess
from pathlib import Path

import pytest

from agentica import worktrees
from agentica.worktrees import WorktreeError, ensure, find, list_worktrees, slug


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    """`<tmp>/codes/proj` on main, one commit, plus a gitignored .env."""
    root = tmp_path / "codes" / "proj"
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / ".gitignore").write_text(".env\n")
    (root / "a.py").write_text("print(1)\n")
    (root / ".env").write_text("API_KEY=secret\n")
    _git(root, "add", ".gitignore", "a.py")
    _git(root, "commit", "-q", "-m", "first")
    return root


class TestSlug:
    def test_spaces_and_case_are_normalised(self):
        assert slug("Gateway Peers") == "gateway-peers"

    def test_a_name_with_nothing_usable_is_refused(self):
        with pytest.raises(WorktreeError):
            slug("///")


class TestLayout:
    def test_a_worktree_is_a_sibling_of_the_main_checkout(self, repo):
        assert worktrees.worktree_path(str(repo), "docs") == str(repo.parent / "proj-docs")

    def test_the_branch_is_prefixed(self, repo):
        assert worktrees.branch_for("docs") == "wt/docs"

    def test_paths_are_derived_from_the_main_checkout_not_the_current_one(self, repo):
        """Called from inside proj-docs, "api" must be proj-api — never
        proj-docs-api. This is why main_root() uses --git-common-dir."""
        docs = ensure(str(repo), "docs")

        assert worktrees.worktree_path(docs.path, "api") == str(repo.parent / "proj-api")
        assert worktrees.main_root(docs.path) == str(repo)


class TestEnsure:
    def test_a_new_worktree_is_created_on_its_own_branch(self, repo):
        wt = ensure(str(repo), "docs")

        assert Path(wt.path).is_dir()
        assert wt.branch_short == "wt/docs"
        assert (Path(wt.path) / "a.py").exists()

    def test_calling_it_again_reuses_the_same_directory(self, repo):
        """"Bind me to <task>" is something a long-running session says twice."""
        first = ensure(str(repo), "docs")
        (Path(first.path) / "scratch.txt").write_text("work in progress\n")

        second = ensure(str(repo), "docs")

        assert second.path == first.path
        assert (Path(second.path) / "scratch.txt").read_text() == "work in progress\n"

    def test_reuse_works_from_inside_another_worktree(self, repo):
        first = ensure(str(repo), "docs")
        other = ensure(str(repo), "api")

        again = ensure(other.path, "docs")

        assert os.path.realpath(again.path) == os.path.realpath(first.path)

    def test_a_branch_that_outlived_its_directory_is_checked_out_again(self, repo):
        wt = ensure(str(repo), "docs")
        _git(repo, "worktree", "remove", wt.path)
        assert not Path(wt.path).exists()

        again = ensure(str(repo), "docs")

        assert Path(again.path).is_dir()
        assert again.branch_short == "wt/docs"

    def test_a_foreign_directory_in_the_way_is_an_error_not_a_cleanup(self, repo):
        squatter = repo.parent / "proj-docs"
        squatter.mkdir()
        (squatter / "important.txt").write_text("not ours\n")

        with pytest.raises(WorktreeError, match="already exists"):
            ensure(str(repo), "docs")

        assert (squatter / "important.txt").exists()

    def test_outside_a_repository_it_refuses(self, tmp_path):
        with pytest.raises(WorktreeError, match="not inside a git repository"):
            ensure(str(tmp_path), "docs")

    def test_master_only_repos_fork_from_master(self, tmp_path):
        root = tmp_path / "old"
        root.mkdir()
        _git(root, "init", "-q", "-b", "master")
        _git(root, "config", "user.email", "t@example.com")
        _git(root, "config", "user.name", "T")
        (root / "a").write_text("a\n")
        _git(root, "add", "a")
        _git(root, "commit", "-q", "-m", "first")

        wt = ensure(str(root), "task")

        assert Path(wt.path).is_dir()


class TestLinkedFiles:
    def test_env_is_symlinked_into_a_new_worktree(self, repo):
        """A fresh worktree without .env starts a session that cannot reach a
        model — and the symptom looks nothing like the cause."""
        wt = ensure(str(repo), "docs")

        env = Path(wt.path) / ".env"
        assert env.is_symlink()
        assert env.read_text() == "API_KEY=secret\n"

    def test_one_edit_reaches_every_worktree(self, repo):
        wt = ensure(str(repo), "docs")

        (repo / ".env").write_text("API_KEY=rotated\n")

        assert (Path(wt.path) / ".env").read_text() == "API_KEY=rotated\n"

    def test_a_missing_source_file_is_skipped(self, repo):
        (repo / ".env").unlink()

        wt = ensure(str(repo), "docs")

        assert not (Path(wt.path) / ".env").exists()

    def test_reuse_re_links_a_file_that_was_removed_by_hand(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / ".env").unlink()

        again = ensure(str(repo), "docs")

        assert (Path(again.path) / ".env").is_symlink()

    def test_an_existing_real_file_is_never_replaced_by_a_link(self, repo):
        wt = ensure(str(repo), "docs")
        env = Path(wt.path) / ".env"
        env.unlink()
        env.write_text("API_KEY=worktree-specific\n")

        ensure(str(repo), "docs")

        assert not env.is_symlink()
        assert env.read_text() == "API_KEY=worktree-specific\n"


class TestListing:
    def test_the_main_checkout_is_listed_first_and_marked(self, repo):
        ensure(str(repo), "docs")
        ensure(str(repo), "api")

        entries = list_worktrees(str(repo))

        assert entries[0].is_main
        assert os.path.realpath(entries[0].path) == os.path.realpath(str(repo))
        assert [e.name for e in entries[1:]] == ["proj-api", "proj-docs"]

    def test_listing_from_inside_a_worktree_sees_the_whole_repo(self, repo):
        docs = ensure(str(repo), "docs")

        entries = list_worktrees(docs.path)

        assert {e.name for e in entries} == {"proj", "proj-docs"}
        assert sum(1 for e in entries if e.is_main) == 1

    def test_find_matches_by_name(self, repo):
        ensure(str(repo), "docs")

        assert find(str(repo), "docs").branch_short == "wt/docs"
        assert find(str(repo), "nope") is None
