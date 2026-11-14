# -*- coding: utf-8 -*-
"""One real git repo per session; tests copy it instead of ``git init``."""
import shutil
import subprocess

import pytest


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture(scope="session")
def git_repo_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("git_template") / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "a.py").write_text("print(1)\n")
    _git(root, "add", "a.py")
    _git(root, "commit", "-q", "-m", "first")
    return root


@pytest.fixture
def clone_git_repo(git_repo_template):
    def _clone(dest):
        shutil.copytree(git_repo_template, dest)
        return dest

    return _clone
