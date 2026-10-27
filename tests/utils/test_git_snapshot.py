# -*- coding: utf-8 -*-
"""<git_snapshot> on a session's first turn."""
import asyncio
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica import Agent
from agentica.agent.config import PromptConfig
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.utils.git_snapshot import GIT_SNAPSHOT_OPEN, build_git_snapshot


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})


class TestGitSnapshot(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        _git(self.repo, "init", "-q", "-b", "main")
        with open(os.path.join(self.repo, "a.py"), "w") as f:
            f.write("x = 1\n")
        _git(self.repo, "add", "a.py")
        _git(self.repo, "commit", "-q", "-m", "first commit")

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_lists_branch_changes_and_commits(self):
        with open(os.path.join(self.repo, "a.py"), "a") as f:
            f.write("y = 2\n")
        snap = build_git_snapshot(self.repo)
        self.assertIn("Branch: main", snap)
        self.assertIn("M a.py", snap)
        self.assertIn("first commit", snap)
        self.assertIn("check before acting", snap)

    def test_clean_tree_and_non_repo(self):
        self.assertIn("Working tree clean.", build_git_snapshot(self.repo))
        with tempfile.TemporaryDirectory() as plain:
            self.assertIsNone(build_git_snapshot(plain))

    def test_only_the_first_turn_gets_it(self):
        agent = Agent(model=OpenAIChat(id="gpt-4o", api_key="fake"), work_dir=self.repo,
                      add_history_to_context=True, prompt_config=PromptConfig(add_git_snapshot=True))
        seen = []

        async def answer(model_self, messages, **kw):
            seen.append([m.content for m in messages if m.role == "user"][-1])
            return ModelResponse(content="ok")

        with patch.object(OpenAIChat, "response", new=answer):
            agent.run_sync("commit it")
            agent.run_sync("and push")
        self.assertTrue(seen[0].startswith(GIT_SNAPSHOT_OPEN))
        self.assertTrue(seen[0].rstrip().endswith("commit it"))
        self.assertNotIn(GIT_SNAPSHOT_OPEN, seen[1])


if __name__ == "__main__":
    unittest.main()
