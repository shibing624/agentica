# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the /checkpoint CLI command (thin wrapper over
CheckpointManager). Isolated to a temp checkpoint root so it never touches
the real ~/.agentica.
"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.cli.commands import _cmd_checkpoint, CommandContext


class TestCheckpointCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = self._tmp.name
        self._root = tempfile.TemporaryDirectory()
        # Redirect the checkpoint store to a temp dir.
        self._patch = patch("agentica.checkpoint.DEFAULT_CHECKPOINT_ROOT", self._root.name)
        self._patch.start()
        agent = types.SimpleNamespace(session_id="cli-test", work_dir=self.work)
        self.ctx = CommandContext(agent_config={}, current_agent=agent)
        self.target = os.path.join(self.work, "calc.py")
        with open(self.target, "w") as f:
            f.write("x = 1\n")

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()
        self._root.cleanup()

    def _manager(self):
        from agentica.checkpoint import CheckpointManager
        return CheckpointManager(session_id="cli-test")

    def test_create_then_list(self):
        _cmd_checkpoint(self.ctx, "create before-refactor calc.py")
        items = self._manager().list()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].label, "before-refactor")
        self.assertEqual(len(items[0].files), 1)

    def test_create_requires_path(self):
        _cmd_checkpoint(self.ctx, "create onlylabel")
        self.assertEqual(len(self._manager().list()), 0)

    def test_restore_rolls_back_edit(self):
        _cmd_checkpoint(self.ctx, "create snap calc.py")
        cid = self._manager().list()[0].id
        # Mutate the file after the snapshot.
        with open(self.target, "w") as f:
            f.write("x = 999  # broken\n")
        _cmd_checkpoint(self.ctx, f"restore {cid}")
        self.assertEqual(open(self.target).read(), "x = 1\n")

    def test_restore_by_id_prefix(self):
        _cmd_checkpoint(self.ctx, "create snap calc.py")
        cid = self._manager().list()[0].id
        with open(self.target, "w") as f:
            f.write("changed\n")
        _cmd_checkpoint(self.ctx, f"restore {cid[:10]}")
        self.assertEqual(open(self.target).read(), "x = 1\n")

    def test_diff_shows_change(self):
        _cmd_checkpoint(self.ctx, "create snap calc.py")
        cid = self._manager().list()[0].id
        with open(self.target, "w") as f:
            f.write("x = 2\n")
        # Should not raise; diff is printed to console.
        _cmd_checkpoint(self.ctx, f"diff {cid}")


if __name__ == "__main__":
    unittest.main()
