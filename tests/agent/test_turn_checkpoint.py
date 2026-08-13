# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for the per-turn aggregating TurnCheckpointer (Reasonix
internal/checkpoint 移植的 per-turn 聚合层，复用 CheckpointManager)。

Key property under test: per-turn aggregation + path dedup + first-touch capture
eliminates the "one checkpoint per edit" explosion — a turn that edits the same
file N times still yields ONE file snapshot in ONE checkpoint.
"""
import os
import tempfile
import unittest

from agentica.checkpoint import RewindResult, RewindScope, TurnCheckpointer


class TestTurnCheckpointer(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = self._tmp.name
        self.tc = TurnCheckpointer(session_id="sess", root_dir=os.path.join(self.work, "_ckpts"))
        self.file_a = os.path.join(self.work, "a.py")
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("original\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_before_begin_turn_is_noop(self):
        self.assertFalse(self.tc.snapshot(self.file_a))
        self.assertIsNone(self.tc.finalize_turn())

    def test_path_dedup_within_turn(self):
        self.tc.begin_turn(1, "do a thing", msg_index=0)
        self.assertTrue(self.tc.snapshot(self.file_a))
        # second touch of the same path in the same turn is ignored
        self.assertFalse(self.tc.snapshot(self.file_a))
        ckpt = self.tc.finalize_turn()
        self.assertIsNotNone(ckpt)
        self.assertEqual(len(ckpt.files), 1)

    def test_first_touch_captures_turn_start_content(self):
        self.tc.begin_turn(1, "edit a.py", msg_index=0)
        self.tc.snapshot(self.file_a)
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("modified\n")
        # a later snapshot in the same turn must NOT overwrite the first-touch content
        self.tc.snapshot(self.file_a)
        self.tc.finalize_turn()
        result = self.tc.rewind(1, RewindScope.CODE)
        self.assertEqual(open(self.file_a).read(), "original\n")
        self.assertIn(os.path.realpath(self.file_a), [os.path.realpath(p) for p in result.restored_paths])

    def test_rewind_code_deletes_file_created_after_snapshot(self):
        new_file = os.path.join(self.work, "new.py")
        self.tc.begin_turn(1, "create new.py", msg_index=0)
        self.tc.snapshot(new_file)  # did not exist -> recorded as creation
        with open(new_file, "w", encoding="utf-8") as f:
            f.write("brand new\n")
        self.tc.finalize_turn()
        self.tc.rewind(1, RewindScope.CODE)
        self.assertFalse(os.path.exists(new_file), "rewind should delete file created during the turn")

    def test_rewind_conversation_returns_msg_index_without_code_restore(self):
        self.tc.begin_turn(1, "chat only", msg_index=7)
        self.tc.snapshot(self.file_a)
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("edited\n")
        self.tc.finalize_turn()
        result = self.tc.rewind(1, RewindScope.CONVERSATION)
        self.assertEqual(result.msg_index, 7)
        self.assertEqual(result.restored_paths, [])
        # conversation-only rewind leaves code untouched
        self.assertEqual(open(self.file_a).read(), "edited\n")

    def test_rewind_both_restores_code_and_returns_boundary(self):
        self.tc.begin_turn(1, "both", msg_index=3)
        self.tc.snapshot(self.file_a)
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("edited\n")
        self.tc.finalize_turn()
        result = self.tc.rewind(1, RewindScope.BOTH)
        self.assertEqual(result.msg_index, 3)
        self.assertEqual(open(self.file_a).read(), "original\n")

    def test_multiple_turns_aggregate_independently(self):
        self.tc.begin_turn(1, "turn one", msg_index=0)
        self.tc.snapshot(self.file_a)
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("v2\n")
        self.tc.finalize_turn()

        self.tc.begin_turn(2, "turn two", msg_index=4)
        self.tc.snapshot(self.file_a)
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("v3\n")
        self.tc.finalize_turn()

        # rewind to turn 1 restores turn-1-start content
        self.tc.rewind(1, RewindScope.CODE)
        self.assertEqual(open(self.file_a).read(), "original\n")
        # rewind to turn 2 restores turn-2-start content (v2)
        self.tc.rewind(2, RewindScope.CODE)
        self.assertEqual(open(self.file_a).read(), "v2\n")

    def test_turn_with_no_snapshots_still_recorded_for_conversation_rewind(self):
        self.tc.begin_turn(1, "no edits", msg_index=5)
        ckpt = self.tc.finalize_turn()
        self.assertIsNotNone(ckpt)
        self.assertEqual(ckpt.turn, 1)
        self.assertEqual(ckpt.msg_index, 5)
        self.assertEqual(len(ckpt.files), 0)

    def test_cross_process_persistence(self):
        self.tc.begin_turn(1, "persist", msg_index=2)
        self.tc.snapshot(self.file_a)
        self.tc.finalize_turn()

        tc2 = TurnCheckpointer(session_id="sess", root_dir=os.path.join(self.work, "_ckpts"))
        result = tc2.rewind(1, RewindScope.CONVERSATION)
        self.assertEqual(result.msg_index, 2)

    def test_rewind_unknown_turn_raises(self):
        with self.assertRaises(ValueError):
            self.tc.rewind(99, RewindScope.CODE)

    def test_begin_turn_finalizes_previous(self):
        self.tc.begin_turn(1, "first", msg_index=0)
        self.tc.snapshot(self.file_a)
        # begin_turn(2) implicitly finalizes turn 1
        self.tc.begin_turn(2, "second", msg_index=3)
        result = self.tc.rewind(1, RewindScope.CODE)
        self.assertIsInstance(result, RewindResult)


class TestRewindResultAndScope(unittest.TestCase):
    def test_scope_values(self):
        self.assertEqual({s.value for s in RewindScope}, {"code", "conversation", "both"})

    def test_rewind_result_fields(self):
        r = RewindResult(turn=1, checkpoint_id="c", scope=RewindScope.CODE, restored_paths=["/a"], msg_index=None)
        self.assertEqual(r.turn, 1)
        self.assertEqual(r.scope, RewindScope.CODE)


if __name__ == "__main__":
    unittest.main()
