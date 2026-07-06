# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the durable CheckpointManager primitive.
"""
import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.checkpoint import CheckpointManager


class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.work = self._tmp.name
        self.cm = CheckpointManager(
            session_id="test-sess",
            root_dir=os.path.join(self.work, "_ckpts"),
        )
        self.file_a = os.path.join(self.work, "a.py")
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("original\n")

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_and_get(self):
        ckpt = self.cm.create("first", [self.file_a])
        self.assertTrue(ckpt.id)
        loaded = self.cm.get(ckpt.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.label, "first")
        self.assertEqual(len(loaded.files), 1)
        self.assertTrue(loaded.files[0].existed)

    def test_restore_reverts_edit(self):
        ckpt = self.cm.create("before", [self.file_a])
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("modified content\n")
        self.assertEqual(open(self.file_a).read(), "modified content\n")

        restored = self.cm.restore(ckpt.id)
        self.assertIn(os.path.realpath(self.file_a), [os.path.realpath(p) for p in restored])
        self.assertEqual(open(self.file_a).read(), "original\n")

    def test_restore_deletes_newly_created_file(self):
        new_file = os.path.join(self.work, "new.py")
        # Snapshot BEFORE creation: file doesn't exist yet.
        ckpt = self.cm.create("pre-create", [new_file])
        with open(new_file, "w", encoding="utf-8") as f:
            f.write("brand new\n")
        self.assertTrue(os.path.exists(new_file))

        self.cm.restore(ckpt.id)
        self.assertFalse(os.path.exists(new_file), "restore should delete file that didn't exist at snapshot")

    def test_diff_reports_changes(self):
        ckpt = self.cm.create("v1", [self.file_a])
        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("changed line\n")
        diff = self.cm.diff(ckpt.id)
        self.assertIn("-original", diff)
        self.assertIn("+changed line", diff)

    def test_diff_no_changes(self):
        ckpt = self.cm.create("v1", [self.file_a])
        self.assertIn("no changes", self.cm.diff(ckpt.id).lower())

    def test_list_newest_first(self):
        import time
        c1 = self.cm.create("one", [self.file_a])
        time.sleep(0.01)
        c2 = self.cm.create("two", [self.file_a])
        ids = [c.id for c in self.cm.list()]
        self.assertEqual(ids[0], c2.id)
        self.assertIn(c1.id, ids)

    def test_multi_file_checkpoint(self):
        file_b = os.path.join(self.work, "b.py")
        with open(file_b, "w", encoding="utf-8") as f:
            f.write("b original\n")
        ckpt = self.cm.create("multi", [self.file_a, file_b])

        with open(self.file_a, "w", encoding="utf-8") as f:
            f.write("a new\n")
        with open(file_b, "w", encoding="utf-8") as f:
            f.write("b new\n")

        self.cm.restore(ckpt.id)
        self.assertEqual(open(self.file_a).read(), "original\n")
        self.assertEqual(open(file_b).read(), "b original\n")

    def test_clear_and_delete(self):
        c1 = self.cm.create("one", [self.file_a])
        self.cm.create("two", [self.file_a])
        self.assertTrue(self.cm.delete(c1.id))
        self.assertIsNone(self.cm.get(c1.id))
        removed = self.cm.clear()
        self.assertEqual(removed, 1)
        self.assertEqual(self.cm.list(), [])

    def test_cross_process_persistence(self):
        """A fresh manager pointed at the same root sees prior checkpoints."""
        ckpt = self.cm.create("persist", [self.file_a])
        cm2 = CheckpointManager(session_id="test-sess", root_dir=os.path.join(self.work, "_ckpts"))
        self.assertIsNotNone(cm2.get(ckpt.id))
        self.assertEqual(cm2.latest().id, ckpt.id)

    def test_restore_missing_raises(self):
        with self.assertRaises(ValueError):
            self.cm.restore("nonexistent-id")


class TestCheckpointPublicExport(unittest.TestCase):
    def test_importable_from_agentica(self):
        from agentica import CheckpointManager as CM, Checkpoint, CheckpointFile
        self.assertIsNotNone(CM)
        self.assertIsNotNone(Checkpoint)
        self.assertIsNotNone(CheckpointFile)


if __name__ == "__main__":
    unittest.main()
