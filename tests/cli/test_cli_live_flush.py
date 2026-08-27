# -*- coding: utf-8 -*-
"""Unit tests for LiveToolStore prefix flush and live-window follow-ups."""

import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agentica.cli.display.live_blocks import LIVE_MAX_ROWS, LiveToolResult, LiveToolStore
from agentica.cli.display.stream import _strip_rich_markup
from agentica.model.message import Message
from agentica.run_response import RunEvent, RunResponse


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

    def test_bind_run_matches_task_description_out_of_order(self):
        store = LiveToolStore()
        store.start("t1", "task", {"description": "look"})
        store.start("t2", "task", {"description": "review"})
        self.assertEqual(store.bind_run("r2", task="review"), "t2")
        self.assertEqual(store.bind_run("r1", task="look"), "t1")
        self.assertEqual(store.parent_for_run("r2"), "t2")

    def test_concurrent_blocks_snapshot_does_not_raise(self):
        store = LiveToolStore()
        errors: list = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    for _ in store.blocks():
                        pass
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for i in range(80):
                store.start(f"a{i}", "execute", {"command": str(i)})
                store.start(f"b{i}", "grep", {"pattern": str(i)})
                store.finish(f"b{i}", LiveToolResult(content="x"))
                store.drain_prefix()
                store.finish(f"a{i}", LiveToolResult(content="y"))
                store.drain_prefix()
        finally:
            stop.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


class TestStreamLiveWindow(unittest.TestCase):
    def _mgr_and_buf(self):
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=120, force_terminal=False, no_color=True)
        return StreamDisplayManager(con), buf

    def test_compose_live_caps_at_live_max_rows(self):
        mgr, _ = self._mgr_and_buf()
        for i in range(LIVE_MAX_ROWS + 5):
            mgr.display_tool("grep", {"pattern": str(i)}, tool_call_id=f"g{i}")
        live = mgr.compose_live("⠋")
        self.assertEqual(len(live), LIVE_MAX_ROWS)
        self.assertTrue(live[-1].startswith("  … +"))
        hidden = (LIVE_MAX_ROWS + 5) - (LIVE_MAX_ROWS - 1)
        self.assertIn(f"+{hidden} more", live[-1])

    def test_abandon_live_flushes_mixed_blocks_in_start_order(self):
        mgr, buf = self._mgr_and_buf()
        mgr.display_tool("execute", {"command": "pytest"}, tool_call_id="e1")
        mgr.display_tool("grep", {"pattern": "x"}, tool_call_id="g1")
        mgr.display_tool_result(
            "grep", "hit", elapsed=0.01,
            tool_args={"pattern": "x"}, tool_call_id="g1",
        )
        self.assertNotIn("execute", buf.getvalue())
        self.assertNotIn("grep", buf.getvalue())
        mgr.abandon_live()
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        exec_idx = next(i for i, ln in enumerate(lines) if "execute" in ln)
        grep_idx = next(i for i, ln in enumerate(lines) if "grep" in ln)
        self.assertLess(exec_idx, grep_idx)
        self.assertEqual(len(mgr.compose_live("⠋")), 0)

    def test_mismatched_result_id_prints_call_before_body(self):
        mgr, buf = self._mgr_and_buf()
        mgr.display_tool("execute", {"command": "sleep 1"}, tool_call_id="e1")
        mgr.display_tool_result(
            "execute", "orphan-body", elapsed=0.2,
            tool_args={"command": "echo hi"}, tool_call_id="other",
        )
        out = buf.getvalue()
        self.assertLess(out.index("execute"), out.index("orphan-body"))
        live = "\n".join(mgr.compose_live("⠋"))
        self.assertIn("sleep 1", live)

    def test_strip_rich_markup_keeps_literal_brackets(self):
        from rich.markup import escape as rich_escape

        budget = rich_escape(" [turns≤5]")
        text = f"[dim cyan]⮕ explore[/dim cyan][dim]{budget}[/dim]"
        plain = _strip_rich_markup(text)
        self.assertIn("[turns≤5]", plain)
        self.assertNotIn("[dim", plain)

    def test_compose_live_survives_concurrent_mutations(self):
        mgr, _ = self._mgr_and_buf()
        errors: list = []
        stop = threading.Event()

        def spinner():
            try:
                while not stop.is_set():
                    mgr.compose_live("⠋")
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=spinner)
        thread.start()
        try:
            for i in range(60):
                mgr.display_tool("execute", {"command": str(i)}, tool_call_id=f"e{i}")
                mgr.display_tool("grep", {"pattern": str(i)}, tool_call_id=f"g{i}")
                mgr.display_tool_result(
                    "grep", "x", elapsed=0.01,
                    tool_args={"pattern": str(i)}, tool_call_id=f"g{i}",
                )
                mgr.display_tool_result(
                    "execute", "y", elapsed=0.01,
                    tool_args={"command": str(i)}, tool_call_id=f"e{i}",
                )
        finally:
            stop.set()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])


class TestGenericExceptionFlushesLive(unittest.TestCase):
    def test_provider_error_prints_in_flight_tool_call(self):
        from io import StringIO
        from rich.console import Console
        from agentica.cli.interactive.stream_loop import _process_stream_response

        buf = StringIO()
        con = Console(file=buf, width=120, force_terminal=False, no_color=True)

        def _stream(*_a, **_k):
            yield RunResponse(
                event=RunEvent.tool_call_started.value,
                tools=[{
                    "tool_call_id": "e1",
                    "tool_name": "execute",
                    "tool_args": {"command": "pytest"},
                }],
            )
            raise ConnectionError("provider dropped")

        agent = SimpleNamespace(
            model=SimpleNamespace(
                usage=SimpleNamespace(request_usage_entries=[], request_summary=lambda: None),
                supports_images=False,
                id="fake",
                context_window=128000,
            ),
            session_id="s",
            working_memory=SimpleNamespace(messages=[Message(role="user", content="hi")]),
            _cancelled=False,
            _running=False,
            _event_callback=None,
            name="Agent",
            run_response=SimpleNamespace(break_reason=None, break_message=None),
            run_stream_sync=_stream,
        )
        cp = MagicMock()
        tui_state = {
            "cost_usd": 0.0,
            "active_seconds": 0.0,
            "total_api_calls": 0,
            "goal_tokens_used": 0,
            "debug": False,
            "session_started_at": time.monotonic(),
            "work_dir": "/tmp",
        }
        with patch(
            "agentica.cli.interactive.stream_loop.get_console", return_value=con,
        ), patch(
            "agentica.cli.interactive.stream_loop.get_turn_checkpointer", return_value=cp,
        ), patch(
            "agentica.cli.interactive.stream_loop.display_agent_execution_error",
        ):
            _process_stream_response(agent, "run tests", tui_state)

        self.assertIn("execute", buf.getvalue())
        self.assertIn("pytest", buf.getvalue())
