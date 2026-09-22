# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Worktree layout helpers plus the create / merge / remove lifecycle.

Layout tests stub ``main_root``. Lifecycle tests use a copied real git repo —
``git worktree add`` on a one-file tree is cheap, and the safety checks are
the behaviour that used to be wrong.
"""
import os
import shutil
from pathlib import Path

import pytest

from agentica.peers import worktrees
from agentica.peers.worktrees import WorktreeError, ensure, slug


class TestSlug:
    def test_spaces_and_case_are_normalised(self):
        assert slug("Gateway Peers") == "gateway-peers"

    def test_a_name_with_nothing_usable_is_refused(self):
        with pytest.raises(WorktreeError):
            slug("///")

    def test_the_base_branch_is_not_a_task_name(self):
        """``use(name="main")`` used to create a second checkout on ``wt/main``.

        ``main`` names the repository root, not a task — and the collision is
        with *branch* names, so it is exactly the local base names that are
        refused. ``home`` is not one of them.
        """
        for reserved in ("main", "master", "MAIN", " Master "):
            with pytest.raises(WorktreeError, match="main checkout"):
                slug(reserved)
        assert slug("home") == "home"


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


def _drop(repo, entry):
    worktrees.remove(entry, str(repo))


def _listed(repo, path):
    want = os.path.realpath(path)
    for entry in worktrees.list_worktrees(str(repo)):
        if os.path.realpath(entry.path) == want:
            return entry
    raise AssertionError(f"no worktree listed at {path}")


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

        _drop(repo, wt)
        assert not Path(wt.path).exists()
        listed = worktrees.list_worktrees(str(repo))
        assert [e.path for e in listed if not e.is_main] == []
        branches = _git_output(repo, "branch", "--list", "wt/docs")
        assert branches.strip() == ""

    def test_remove_refuses_uncommitted_work(self, repo):
        """Git refuses this one itself, and its sentence is what the caller gets.

        Verbatim, including the ``--force`` it advertises: rewriting git's text
        to hide that reads as censorship, works only in English, and the model
        can run the same command through ``execute`` anyway.
        """
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "dirty.py").write_text("nope\n")
        with pytest.raises(WorktreeError) as exc:
            _drop(repo, wt)
        message = str(exc.value)
        assert "modified or untracked" in message, "git's diagnosis must survive"
        assert Path(wt.path).is_dir()

    def test_remove_of_an_unmerged_branch_keeps_the_commits(self, repo):
        """The checkout goes; the work does not.

        This used to raise "merge them first" — a *workflow* opinion dressed as
        a safety rule, and the one that made ``remove`` and ``merge`` contradict
        each other on the same branch. What actually protects the commits is
        ``git branch -d``, which refuses an unmerged branch, so the branch stays
        and every commit on it stays reachable.
        """
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "not merged yet")

        _drop(repo, wt)

        assert not Path(wt.path).exists()
        assert _git_output(repo, "branch", "--list", "wt/docs").strip() != "", (
            "an unmerged branch must survive the checkout being removed"
        )
        assert "not merged yet" in _git_output(repo, "log", "--oneline", "wt/docs")

    def test_remove_of_a_clean_unused_worktree_is_allowed(self, repo):
        wt = ensure(str(repo), "docs")
        _drop(repo, wt)
        assert not Path(wt.path).exists()

    def test_same_name_after_remove_forks_from_current_main(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "add feature")
        worktrees.merge_back(wt.path)
        _drop(repo, wt)

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
            _drop(repo, _listed(repo, inspect))
        assert inspect.is_dir()

    def test_remove_refuses_a_non_wt_branch(self, repo, tmp_path):
        other = tmp_path / "review"
        _git(repo, "worktree", "add", "-b", "review/docs", str(other), "HEAD")
        with pytest.raises(WorktreeError, match="not an agentica worktree"):
            _drop(repo, _listed(repo, other))
        assert other.is_dir()
        assert "review/docs" in _git_output(repo, "branch", "--list", "review/docs")

    def test_remove_without_main_or_master_is_gits_decision(self, repo):
        """No local main/master is not a reason to refuse an *explicit* remove.
        Teardown still fail-closes via has_unique_work."""
        _git(repo, "branch", "-m", "main", "trunk")
        wt = ensure(str(repo), "docs", base="trunk")
        entry = worktrees.resolve_entry(wt.path)
        assert worktrees.has_unique_work(entry) is True
        _drop(repo, wt)
        assert not Path(wt.path).exists()

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
            "agentica.config.profiles.get_setting",
            lambda key, default=None: "sibling" if key == "worktree.root" else default,
        )
        assert worktrees._configured_root() == "sibling"

    def test_a_nested_worktree_root_block_is_read(self, monkeypatch):
        monkeypatch.setattr(
            "agentica.config.profiles.get_setting",
            lambda key, default=None: default,
        )
        monkeypatch.setattr(
            "agentica.config.profiles.load_global_config",
            lambda: {"settings": {"worktree": {"root": "sibling"}}},
        )
        assert worktrees._configured_root() == "sibling"


def _git_output(cwd, *args):
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True
    )
    return result.stdout


class TestDeletedWorkingDirectory:
    """When another session merges away the worktree this one stands in, every
    git call here runs with a ``cwd`` that no longer exists. The report must
    name that, and the session must still have a way out."""

    def test_a_deleted_cwd_is_not_reported_as_missing_git(self, repo):
        """A deleted directory must not be reported as a missing git binary.

        ``main_root`` answers from the nearest ancestor that still exists: the
        repository is still nameable, and a stale registration can only be
        removed by someone who can name it.
        """
        wt = ensure(str(repo), "docs")
        _drop(repo, wt)

        assert worktrees.main_root(wt.path) == str(repo)

        with pytest.raises(WorktreeError) as exc:
            worktrees.current_root(wt.path)

        message = str(exc.value)
        assert "no longer exists" in message
        assert "not installed" not in message, (
            "ENOENT here is the deleted directory, not a missing git binary — "
            "blaming git sends the reader to install what they already have"
        )

    def test_a_missing_git_binary_is_still_blamed_on_git(self, repo, monkeypatch):
        """The other ENOENT. Both reach ``subprocess`` the same way, and the
        message must not send the reader to look for a directory."""
        import subprocess

        def no_git(*_args, **_kwargs):
            raise FileNotFoundError(2, "No such file or directory", "git")

        monkeypatch.setattr(subprocess, "run", no_git)
        with pytest.raises(WorktreeError) as exc:
            worktrees._git(["status"], str(repo))

        message = str(exc.value)
        assert "not installed" in message
        assert "no longer exists" not in message

    def test_a_session_whose_worktree_was_removed_can_still_get_a_new_one(self, repo):
        """The escape hatch. ``ensure`` asks ``is_git_repo(cwd)`` first, which is
        False for a directory that is gone — so the one tool that could move the
        session refused to, leaving it unable to run anything at all."""
        wt = ensure(str(repo), "docs")
        _drop(repo, wt)

        rescued = ensure(wt.path, "rescue")

        assert Path(rescued.path) == repo / ".agentica/worktrees" / "rescue"
        assert Path(rescued.path).is_dir()


class TestStaleRegistrationIsNotDestroyed:
    """A missing or bare path is reported. ``ensure`` does not rmtree it."""

    def test_ensure_refuses_a_deleted_registration(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "work on docs")
        shutil.rmtree(wt.path)

        with pytest.raises(WorktreeError, match="not a checkout"):
            ensure(str(repo), "docs")

        _drop(repo, wt)
        again = ensure(str(repo), "docs")
        assert (Path(again.path) / "feature.py").is_file()

    def test_a_bare_directory_at_the_registered_path_is_left_alone(self, repo):
        wt = ensure(str(repo), "docs")
        shutil.rmtree(wt.path)
        Path(wt.path).mkdir(parents=True)
        (Path(wt.path) / "accident.py").write_text("not a checkout\n")

        listed = worktrees.find(str(repo), "docs")
        assert listed is not None
        assert not listed.exists
        assert "(not a checkout)" in listed.describe()

        with pytest.raises(WorktreeError, match="not a checkout"):
            ensure(str(repo), "docs")

        assert (Path(wt.path) / "accident.py").read_text() == "not a checkout\n"

    def test_remove_clears_a_gone_registration_without_touching_another(self, repo):
        alive = ensure(str(repo), "keeper")
        stale = ensure(str(repo), "docs")
        shutil.rmtree(stale.path)

        _drop(repo, stale)

        assert Path(alive.path).is_dir()
        assert worktrees.find(str(repo), "keeper") is not None
        assert worktrees.find(str(repo), "docs") is None

    def test_remove_clears_a_locked_gone_registration(self, repo):
        wt = ensure(str(repo), "docs")
        _git(repo, "worktree", "lock", "--reason", "agentica pid=1", wt.path)
        shutil.rmtree(wt.path)

        _drop(repo, wt)

        assert worktrees.find(str(repo), "docs") is None
        again = ensure(str(repo), "docs")
        assert Path(again.path).is_dir()

    def test_remove_of_a_bare_directory_clears_the_name_and_keeps_the_files(self, repo):
        wt = ensure(str(repo), "docs")
        shutil.rmtree(wt.path)
        Path(wt.path).mkdir(parents=True)
        (Path(wt.path) / "accident.py").write_text("not a checkout\n")

        _drop(repo, wt)

        assert worktrees.find(str(repo), "docs") is None
        assert (Path(wt.path) / "accident.py").read_text() == "not a checkout\n"
        with pytest.raises(WorktreeError, match="already exists but is not a worktree"):
            ensure(str(repo), "docs")

    def test_remove_of_a_worktree_whose_git_file_is_gone_does_not_delete_files(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "dirty.py").write_text("keep me\n")
        git_file = Path(wt.path) / ".git"
        assert git_file.is_file()
        git_file.unlink()

        _drop(repo, wt)

        assert worktrees.find(str(repo), "docs") is None
        assert (Path(wt.path) / "dirty.py").read_text() == "keep me\n"


class TestSiblingStaleRemove:
    """A sibling tree's parent is not a git repo. Path reverse-lookup
    cannot see the registration; remove must ask the session's checkout."""

    def test_a_gone_sibling_can_be_removed_then_recreated(self, repo, monkeypatch):
        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.SIBLING_ROOT)
        wt = ensure(str(repo), "docs")
        assert Path(wt.path).parent == repo.parent
        shutil.rmtree(wt.path)

        listed = worktrees.find(str(repo), "docs")
        assert listed is not None
        assert not listed.exists
        assert worktrees._entry_for_path(wt.path) is None

        with pytest.raises(WorktreeError, match="not a checkout"):
            ensure(str(repo), "docs")

        _drop(repo, listed)
        assert worktrees.find(str(repo), "docs") is None
        again = ensure(str(repo), "docs")
        assert Path(again.path) == Path(wt.path)
        assert Path(again.path).is_dir()

    def test_a_bare_sibling_directory_is_left_alone(self, repo, monkeypatch):
        monkeypatch.setattr(worktrees, "_configured_root", lambda: worktrees.SIBLING_ROOT)
        wt = ensure(str(repo), "docs")
        shutil.rmtree(wt.path)
        Path(wt.path).mkdir(parents=True)
        (Path(wt.path) / "accident.py").write_text("not a checkout\n")

        listed = worktrees.find(str(repo), "docs")
        assert listed is not None
        assert not listed.exists
        _drop(repo, listed)

        assert worktrees.find(str(repo), "docs") is None
        assert (Path(wt.path) / "accident.py").read_text() == "not a checkout\n"


class TestRemoveRefusesTheProcessCwd:
    def test_remove_refuses_when_this_process_stands_in_the_tree(self, repo, monkeypatch):
        wt = ensure(str(repo), "docs")
        monkeypatch.setattr(worktrees.os, "getcwd", lambda: wt.path)
        with pytest.raises(WorktreeError, match="this process's working directory"):
            _drop(repo, wt)
        assert Path(wt.path).is_dir()


class TestNestedWorktreesSelfExclude:
    def test_nested_worktrees_lists_in_repo_checkouts(self, repo):
        wt = ensure(str(repo), "docs")
        nested = worktrees.nested_worktrees(str(repo), ttl=0)
        assert os.path.realpath(wt.path) in nested

    def test_nested_checkouts_skips_the_search_base_itself(self, repo):
        from agentica.tools.builtin.file_tool import _nested_checkouts

        wt = ensure(str(repo), "docs")
        here = os.path.realpath(wt.path)
        skipped_from_main = _nested_checkouts(repo)
        assert here in skipped_from_main
        skipped_from_self = _nested_checkouts(Path(wt.path))
        assert here not in skipped_from_self


class TestMergeOfAnAlreadyLandedBranch:
    """"Everything is already on main" is the *success* state of a finished
    task, not an error. Refusing it taught a live session that the tool was a
    dead end for exactly the case it was built for — so it went around it with
    `execute` and deleted the directory it was standing in."""

    def test_merge_back_of_an_already_merged_branch_is_not_an_error(self, repo):
        wt = ensure(str(repo), "docs")
        (Path(wt.path) / "feature.py").write_text("x = 1\n")
        _git(wt.path, "add", "feature.py")
        _git(wt.path, "commit", "-q", "-m", "add feature")
        worktrees.merge_back(wt.path)

        # Same call again: main already has every commit of the branch.
        result = worktrees.merge_back(wt.path)

        assert not result.conflicted
        assert result.commits == 0
        assert result.already_merged is True
        assert (repo / "feature.py").is_file(), "the earlier merge must stand"

    def test_a_branch_with_nothing_on_it_at_all_is_also_not_an_error(self, repo):
        """A worktree opened and never committed to. Nothing to land, nothing
        lost by saying so — the caller's next step (remove) is the same."""
        wt = ensure(str(repo), "docs")

        result = worktrees.merge_back(wt.path)

        assert result.already_merged is True
        assert result.commits == 0
