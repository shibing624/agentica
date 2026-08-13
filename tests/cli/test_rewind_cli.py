"""Tests for the /rewind command and its helpers (turn-loop auto-checkpoint
wiring, Clue-Code-style rewind of code + conversation)."""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.runtime import _cmd_rewind
from agentica.cli.rewind import extract_rewrite_paths, truncate_conversation


class TestExtractRewritePaths(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()

    def test_write_file(self):
        p = extract_rewrite_paths("write_file", {"file_path": "a.py"}, self.work)
        self.assertEqual(len(p), 1)
        self.assertTrue(p[0].endswith("a.py"))

    def test_edit_file(self):
        p = extract_rewrite_paths("edit_file", {"file_path": "sub/b.py"}, self.work)
        self.assertEqual(len(p), 1)
        self.assertTrue(p[0].endswith(os.path.join("sub", "b.py")))

    def test_apply_patch_multi_file(self):
        patch = (
            "*** Begin Patch\n"
            "*** Update File: a.py\n"
            "@@\n-1\n+2\n"
            "*** Add File: b.py\n"
            "+x = 1\n"
            "*** Delete File: c.py\n"
            "*** End Patch\n"
        )
        paths = extract_rewrite_paths("apply_patch", {"patch": patch}, self.work)
        self.assertEqual(len(paths), 3)
        names = {os.path.basename(p) for p in paths}
        self.assertEqual(names, {"a.py", "b.py", "c.py"})

    def test_absolute_and_tilde(self):
        abs_path = os.path.join(self.work, "abs.py")
        paths = extract_rewrite_paths("write_file", {"file_path": abs_path}, self.work)
        self.assertEqual(paths, [os.path.realpath(abs_path)])

    def test_non_rewrite_tool_returns_empty(self):
        self.assertEqual(extract_rewrite_paths("read_file", {"file_path": "a.py"}, self.work), [])
        self.assertEqual(extract_rewrite_paths("grep", {"pattern": "x"}, self.work), [])


class TestTruncateConversation(unittest.TestCase):
    def _agent(self, n_messages):
        from agentica.memory.working import WorkingMemory
        from agentica.model.message import Message

        wm = WorkingMemory()
        for i in range(n_messages):
            role = "user" if i % 2 == 0 else "assistant"
            wm.messages.append(Message(role=role, content=f"m{i}"))
        return types.SimpleNamespace(working_memory=wm)

    def test_truncates_messages_and_runs(self):
        agent = self._agent(6)
        removed = truncate_conversation(agent, 4)
        self.assertEqual(removed, 2)
        self.assertEqual(len(agent.working_memory.messages), 4)

    def test_noop_when_boundary_at_or_after_end(self):
        agent = self._agent(3)
        self.assertEqual(truncate_conversation(agent, 3), 0)
        self.assertEqual(truncate_conversation(agent, 10), 0)
        self.assertEqual(len(agent.working_memory.messages), 3)


class TestRewindCommand(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = self._tmp.name
        self._root = tempfile.TemporaryDirectory()
        self._patch = patch("agentica.checkpoint.DEFAULT_CHECKPOINT_ROOT", self._root.name)
        self._patch.start()

        from agentica.memory.working import WorkingMemory

        wm = WorkingMemory()
        agent = types.SimpleNamespace(
            session_id="rewind-cli", work_dir=self.work, working_memory=wm
        )
        self.ctx = CommandContext(agent_config={}, current_agent=agent, tui_state={})
        self.target = os.path.join(self.work, "calc.py")
        with open(self.target, "w") as f:
            f.write("x = 1\n")

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()
        self._root.cleanup()

    def _seed_turn(self, turn, prompt="edit", msg_index=0):
        from agentica.cli.rewind import get_turn_checkpointer

        tc = get_turn_checkpointer(self.ctx.tui_state, "rewind-cli")
        tc.begin_turn(turn, prompt=prompt, msg_index=msg_index)
        tc.snapshot(self.target)
        tc.finalize_turn()

    def test_list_turns(self):
        self._seed_turn(1, prompt="turn one", msg_index=0)
        _cmd_rewind(self.ctx, "list")
        from agentica.checkpoint import TurnCheckpointer

        turns = TurnCheckpointer(session_id="rewind-cli").list_turns()
        self.assertEqual([t.turn for t in turns], [1])

    def test_rewind_requires_yes(self):
        self._seed_turn(1, msg_index=0)
        with open(self.target, "w") as f:
            f.write("x = 999  # broken\n")
        _cmd_rewind(self.ctx, "rewind 1")
        self.assertEqual(open(self.target).read(), "x = 999  # broken\n")

    def test_rewind_yes_restores_code_and_conversation(self):
        self._seed_turn(1, prompt="break it", msg_index=0)
        with open(self.target, "w") as f:
            f.write("x = 999  # broken\n")
        # conversation grew after the turn began
        from agentica.model.message import Message

        self.ctx.current_agent.working_memory.messages.append(
            Message(role="assistant", content="did it")
        )
        _cmd_rewind(self.ctx, "rewind 1 --yes")
        self.assertEqual(open(self.target).read(), "x = 1\n")
        self.assertEqual(len(self.ctx.current_agent.working_memory.messages), 0)

    def test_rewind_unknown_turn(self):
        _cmd_rewind(self.ctx, "rewind 99 --yes")

    def test_rewind_by_number_syntax(self):
        self._seed_turn(1, prompt="break it", msg_index=0)
        with open(self.target, "w") as f:
            f.write("x = 999  # broken\n")
        _cmd_rewind(self.ctx, "1 --yes")
        self.assertEqual(open(self.target).read(), "x = 1\n")


if __name__ == "__main__":
    unittest.main()
