# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""

import logging
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli import commands as cli_commands
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



class TestPendingQueueTimestamps(unittest.TestCase):
    """``PendingQueue`` must expose per-item submission timestamps so the TUI
    queue bar can label each pending message with when it was submitted.
    Timestamps must stay aligned with items across get / remove_index / clear.
    """

    def test_put_records_timestamp_for_each_item(self):
        import time as _t
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        before = _t.time()
        q.put("first")
        q.put("second")
        after = _t.time()

        pairs = q.peek_all_with_timestamps()
        self.assertEqual([p[0] for p in pairs], ["first", "second"])
        for _, ts in pairs:
            self.assertGreaterEqual(ts, before)
            self.assertLessEqual(ts, after)

    def test_get_pops_item_and_timestamp_in_lockstep(self):
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        q.put("b")
        q.put("c")

        self.assertEqual(q.get(timeout=0.1), "a")
        remaining = q.peek_all_with_timestamps()
        self.assertEqual([p[0] for p in remaining], ["b", "c"])
        self.assertEqual(len(remaining), 2)

    def test_remove_index_keeps_timestamps_aligned(self):
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        q.put("b")
        q.put("c")
        ts_b = q.peek_all_with_timestamps()[1][1]

        self.assertTrue(q.remove_index(0))
        pairs = q.peek_all_with_timestamps()
        self.assertEqual([p[0] for p in pairs], ["b", "c"])
        self.assertEqual(pairs[0][1], ts_b, "after removing index 0, 'b' must keep its original timestamp")

    def test_clear_drops_timestamps(self):
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        q.put("b")
        q.clear()
        self.assertEqual(q.peek_all_with_timestamps(), [])
        self.assertTrue(q.empty())

    def test_replace_index_updates_item_and_timestamp(self):
        """``replace_index`` edits in place and bumps the timestamp so the
        TUI queue bar's 'x seconds ago' label reflects the latest user
        intent. Other slots' timestamps must stay untouched.
        """
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        q.put("b")
        q.put("c")
        pairs_before = q.peek_all_with_timestamps()
        ts_a, ts_b, ts_c = (p[1] for p in pairs_before)

        # tiny sleep so the new timestamp is strictly greater
        import time

        time.sleep(0.001)

        self.assertTrue(q.replace_index(1, "b2"))
        pairs_after = q.peek_all_with_timestamps()
        self.assertEqual([p[0] for p in pairs_after], ["a", "b2", "c"])
        self.assertEqual(pairs_after[0][1], ts_a, "slot 0 timestamp must stay")
        self.assertEqual(pairs_after[2][1], ts_c, "slot 2 timestamp must stay")
        self.assertGreater(pairs_after[1][1], ts_b, "edited slot must get a fresher timestamp")

    def test_replace_index_out_of_range_returns_false(self):
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        self.assertFalse(q.replace_index(5, "x"))
        self.assertFalse(q.replace_index(-1, "x"))
        # original untouched
        self.assertEqual([p[0] for p in q.peek_all_with_timestamps()], ["a"])

    def test_insert_index_at_front_middle_and_end(self):
        """``insert_index`` accepts ``0..len`` (inclusive on the upper bound,
        equivalent to append) and keeps timestamps aligned with their slots.
        """
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        q.put("c")

        # insert at front
        self.assertTrue(q.insert_index(0, "head"))
        self.assertEqual([p[0] for p in q.peek_all_with_timestamps()], ["head", "a", "c"])

        # insert in the middle
        self.assertTrue(q.insert_index(2, "mid"))
        self.assertEqual([p[0] for p in q.peek_all_with_timestamps()], ["head", "a", "mid", "c"])

        # insert at the end (idx == len) is valid → append
        n = q.qsize()
        self.assertTrue(q.insert_index(n, "tail"))
        self.assertEqual([p[0] for p in q.peek_all_with_timestamps()], ["head", "a", "mid", "c", "tail"])

    def test_insert_index_out_of_range_returns_false(self):
        from agentica.cli.commands.context import PendingQueue

        q = PendingQueue()
        q.put("a")
        # idx == len(q) is allowed (append); idx == len(q) + 1 is not.
        self.assertFalse(q.insert_index(5, "x"))
        self.assertFalse(q.insert_index(-1, "x"))
        self.assertEqual([p[0] for p in q.peek_all_with_timestamps()], ["a"])


class TestQueueItemPreview(unittest.TestCase):
    """The queue bar shows every queued payload, and says how it will run.

    Regression: the bar used to drop every ``startswith("/")`` item, so a
    queued ``/requesting-code-review ...`` rendered as a blank row and looked
    like it never entered the queue.
    """

    def test_skill_and_cli_slash_prompts_preview(self):
        from agentica.cli.interactive.attachments import queue_item_preview

        self.assertEqual(
            queue_item_preview("/requesting-code-review git status的代码"),
            "/requesting-code-review git status的代码",
        )
        self.assertEqual(queue_item_preview("/status"), "/status")
        self.assertEqual(queue_item_preview("normal follow-up"), "normal follow-up")

    def test_image_payload_previews_its_text(self):
        from agentica.cli.interactive.attachments import queue_item_preview

        self.assertEqual(queue_item_preview(("describe this", ["/tmp/a.png"])), "describe this")

    def test_btw_tuple_preview(self):
        from agentica.cli.interactive.attachments import queue_item_preview

        self.assertEqual(
            queue_item_preview(("__BTW__", "what model is this?")),
            "__BTW__: what model is this?",
        )


class TestQueueCommandEditInsert(unittest.TestCase):
    """``/queue edit <n> <text>`` and ``/queue insert <n> <text>`` give users
    in-place editing of the pending queue without the
    remove-then-append dance (which would silently shuffle order).
    """

    def _ctx_with_queue(self):
        from unittest.mock import MagicMock
        from agentica.cli.commands.context import CommandContext, PendingQueue

        pq = PendingQueue()
        ctx = CommandContext(
            agent_config={"model_provider": "zhipuai", "model_name": "glm-5", "debug": False, "work_dir": None},
            current_agent=MagicMock(),
            extra_tools=[],
            workspace=None,
            pending_queue=pq,
        )
        # _cmd_queue checks ctx.agent_running; mark as not running so the
        # "Queued: ..." preview path is exercised cleanly when needed.
        ctx.agent_running = False
        return ctx, pq

    def test_edit_replaces_item_in_place(self):
        from agentica.cli.commands.runtime import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("first")
        pq.put("second")
        pq.put("third")

        _cmd_queue(ctx, "edit 2 SECOND v2")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["first", "SECOND v2", "third"])

    def test_edit_rejects_missing_text(self):
        from agentica.cli.commands.runtime import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("first")
        # No new text → must NOT mutate the queue.
        _cmd_queue(ctx, "edit 1")
        _cmd_queue(ctx, "edit 1   ")  # whitespace-only also rejected
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["first"])

    def test_edit_rejects_bad_index(self):
        from agentica.cli.commands.runtime import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("only")
        _cmd_queue(ctx, "edit 99 nope")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["only"])

    def test_insert_at_front(self):
        from agentica.cli.commands.runtime import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("a")
        pq.put("b")
        _cmd_queue(ctx, "insert 1 head")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["head", "a", "b"])

    def test_insert_at_back_equivalent_to_append(self):
        """``/queue insert <qsize+1> text`` is documented as 'back' and must
        be accepted, mapping to the same slot as a plain ``/queue text``."""
        from agentica.cli.commands.runtime import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("a")
        pq.put("b")
        # qsize+1 (1-based) → idx == len (0-based) → append
        _cmd_queue(ctx, f"insert {pq.qsize() + 1} tail")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["a", "b", "tail"])

    def test_insert_rejects_bad_index(self):
        from agentica.cli.commands.runtime import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("a")
        _cmd_queue(ctx, "insert 99 nope")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["a"])


if __name__ == "__main__":
    unittest.main()
