# -*- coding: utf-8 -*-
"""Unit tests for LiveToolStore prefix flush."""

import unittest

from agentica.cli.display.live_blocks import LiveToolResult, LiveToolStore


class TestLiveToolStore(unittest.TestCase):
    def test_prefix_flush_stops_at_first_unfinished(self):
        store = LiveToolStore()
        store.start("a", "execute", {"command": "pytest"})
        store.start("b", "grep", {"pattern": "x"})
        store.finish("b", LiveToolResult(content="hit"))
        self.assertEqual(store.drain_prefix(), [])
        self.assertEqual(len(store), 2)
        store.finish("a", LiveToolResult(content="ok"))
        flushed = store.drain_prefix()
        self.assertEqual([b.tool_call_id for b in flushed], ["a", "b"])
        self.assertEqual(len(store), 0)

    def test_bind_run_attaches_to_first_unbound_task(self):
        store = LiveToolStore()
        store.start("t1", "task", {"description": "one"})
        store.start("t2", "task", {"description": "two"})
        self.assertEqual(store.bind_run("r1"), "t1")
        self.assertEqual(store.bind_run("r2"), "t2")
        self.assertEqual(store.parent_for_run("r1"), "t1")
        store.finish("t1", LiveToolResult(content="{}"))
        store.drain_prefix()
        self.assertIsNone(store.parent_for_run("r1"))
