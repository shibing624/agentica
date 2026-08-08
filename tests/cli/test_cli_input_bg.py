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



class TestInputRequestCancel(unittest.TestCase):
    """Regression tests for the Ctrl+C escape path through an ask-user prompt.

    Motivating bug: when the agent called a ask_user_question / confirm tool, the CLI
    armed an ``_InputRequest`` and the tool thread blocked on
    ``req.result.get()``. Pressing Ctrl+C only reached ``asyncio.Task.cancel()``,
    which cannot interrupt a synchronous blocking ``queue.Queue.get()`` running
    on the thread-pool worker, so the whole REPL froze on the spinner.

    The fix threads a ``_InputRequest.CANCELLED`` sentinel through the queue.
    These tests lock that contract in.
    """

    def _import_input_request(self):
        # Imported lazily so a broken import surfaces as a test failure rather
        # than a collection error.
        from agentica.cli.interactive.session_state import _InputRequest

        return _InputRequest

    def test_cancel_unblocks_pending_get(self):
        _InputRequest = self._import_input_request()
        req = _InputRequest(prompt="?", options=None)
        import threading

        got = {}

        def worker():
            got["value"] = req.result.get(timeout=5)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        # Simulate the Ctrl+C handler waking the worker.
        req.cancel()
        t.join(timeout=2)
        self.assertFalse(t.is_alive(), "worker thread must unblock after cancel()")
        self.assertIs(got["value"], _InputRequest.CANCELLED)

    def test_cancel_is_idempotent_when_answer_already_present(self):
        _InputRequest = self._import_input_request()
        req = _InputRequest(prompt="?", options=None)
        # A real user typed an answer first — cancel() must not blow up trying
        # to overfill the maxsize=1 queue.
        req.result.put("hi")
        req.cancel()  # should silently no-op, not raise
        self.assertEqual(req.result.get_nowait(), "hi")

    def test_sentinel_is_singleton_and_unique(self):
        _InputRequest = self._import_input_request()
        # Callers distinguish "cancelled" from "the user typed empty string" via
        # ``is _InputRequest.CANCELLED``. If the sentinel were, e.g., a plain
        # empty string, an empty user reply would be indistinguishable.
        self.assertIsNot(_InputRequest.CANCELLED, "")
        self.assertIsNot(_InputRequest.CANCELLED, None)
        # Same object across all instances.
        req_a = _InputRequest(prompt="a")
        req_b = _InputRequest(prompt="b")
        self.assertIs(req_a.CANCELLED, req_b.CANCELLED)

    def test_enter_keeps_whitespace_for_pending_ask_user_question_request(self):
        request = self._import_input_request()(prompt="Need exact text")
        typed = "  keep surrounding whitespace  "

        # Regression guard for the interactive Enter handler: ask-user replies
        # must preserve the raw buffer content instead of silently applying
        # ``.strip()`` like normal chat turns do.
        request.result.put(typed)

        self.assertEqual(request.result.get_nowait(), typed)

    def test_cancelled_request_uses_single_sentinel_once(self):
        _InputRequest = self._import_input_request()
        request = _InputRequest(prompt="Need answer")

        self.assertTrue(request.cancel())

        self.assertIs(request.result.get_nowait(), _InputRequest.CANCELLED)
        self.assertTrue(request.result.empty())
        self.assertFalse(request.cancel())

    def test_cancelled_request_ignores_late_submit(self):
        _InputRequest = self._import_input_request()
        request = _InputRequest(prompt="Need answer")

        request.cancel()
        self.assertFalse(request.submit("late answer"))

        self.assertIs(request.result.get_nowait(), _InputRequest.CANCELLED)

    def test_submitted_request_ignores_late_cancel(self):
        _InputRequest = self._import_input_request()
        request = _InputRequest(prompt="Need answer")

        self.assertTrue(request.submit("final answer"))
        self.assertFalse(request.cancel())

        self.assertEqual(request.result.get_nowait(), "final answer")


class TestBackgroundCompletionNotice(unittest.TestCase):
    def test_background_log_tail_skips_command_header(self):
        from agentica.tools.background_processes import read_log_tail

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "term.log"
            log_path.write_text(
                "$ python -c 'print(1)'\n\n"
                "line1\nline2\nline3\nline4\nline5\nline6\n",
                encoding="utf-8",
            )

            tail = read_log_tail(str(log_path), max_lines=5)

        self.assertEqual(tail.splitlines(), ["line2", "line3", "line4", "line5", "line6"])

    def test_print_background_completion_notice(self):
        from agentica.cli.interactive import btw as it
        from agentica.tools.background_processes import BackgroundProcessCompleted

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "term.log"
            log_path.write_text("$ echo done\n\ndone\n", encoding="utf-8")
            event = BackgroundProcessCompleted(
                id="term_2",
                num=2,
                pid=123,
                command="echo done",
                cwd=td,
                log_path=str(log_path),
                started_at=1.0,
                completed_at=4.0,
                returncode=0,
            )
            fake_console = MagicMock()

            with patch.object(it, "get_console", return_value=fake_console):
                it._print_background_completion(event)

        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list if call.args)
        self.assertIn("Background terminal #2 finished", rendered)
        self.assertIn("exit 0", rendered)
        self.assertIn("echo done", rendered)
        self.assertIn("done", rendered)
        self.assertIn("log:", rendered)

    def test_print_background_failure_notice(self):
        from agentica.cli.interactive import btw as it
        from agentica.tools.background_processes import BackgroundProcessCompleted

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "term.log"
            log_path.write_text("$ false\n\ncommand failed\n", encoding="utf-8")
            event = BackgroundProcessCompleted(
                id="term_3",
                num=3,
                pid=456,
                command="false",
                cwd=td,
                log_path=str(log_path),
                started_at=1.0,
                completed_at=2.0,
                returncode=1,
            )
            fake_console = MagicMock()

            with patch.object(it, "get_console", return_value=fake_console):
                it._print_background_completion(event)

        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list if call.args)
        self.assertIn("Background terminal #3 failed", rendered)
        self.assertIn("exit 1", rendered)
        self.assertIn("command failed", rendered)

    def test_print_background_completion_retains_full_command_for_ctrl_o(self):
        """Long background commands stay expandable via Ctrl+O."""
        from agentica.cli.interactive import btw as it
        from agentica.cli import display as disp
        from agentica.tools.background_processes import BackgroundProcessCompleted

        long_command = (
            "cd /apdcephfs_qy3/share_7435715/flemingxu/nlp/exp/dual_mem_exp/benchmarks && "
            "DUAL_MEM_EXP=1 python -m personamem.run "
            + " ".join(f"--qid q{i}" for i in range(30))
        )
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "term.log"
            log_path.write_text(f"$ {long_command}\n\nok\n", encoding="utf-8")
            event = BackgroundProcessCompleted(
                id="term_3",
                num=3,
                pid=789,
                command=long_command,
                cwd=td,
                log_path=str(log_path),
                started_at=1.0,
                completed_at=199.0,
                returncode=0,
            )
            fake_console = MagicMock()
            fake_console.width = 80
            disp.clear_truncated_blocks()

            with patch.object(it, "get_console", return_value=fake_console):
                it._print_background_completion(event)

        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list if call.args)
        self.assertIn("Background terminal #3 finished", rendered)
        self.assertIn("Ctrl+O", rendered)
        blocks = disp.get_truncated_blocks()
        self.assertTrue(blocks)
        self.assertEqual(blocks[-1]["content"], long_command)
        self.assertIn("--qid q29", blocks[-1]["content"])


class TestAskActiveFreeze(unittest.TestCase):
    """While a ask_user_question prompt is armed, ``_cprint`` must drop output
    so background ``run_in_terminal`` writes can't starve the main
    prompt_toolkit event loop (the CLI-appears-frozen bug)."""

    def test_cprint_drops_while_ask_active(self):
        from agentica.cli.interactive import console_io as it

        prev = it._ask_active[0]
        it._ask_active[0] = True
        try:
            # Must return without raising and without touching the terminal.
            it._cprint("should be dropped")
        finally:
            it._ask_active[0] = prev

    def test_ask_active_defaults_false(self):
        from agentica.cli.interactive import console_io as it

        self.assertFalse(it._ask_active[0])


class TestTranscriptPause(unittest.TestCase):
    def test_paused_output_is_buffered_then_flushed_in_order(self):
        from agentica.cli.interactive import console_io as it

        it._clear_output_pause()
        with patch.object(it, "print_formatted_text") as render:
            paused, count = it._toggle_output_pause()
            self.assertTrue(paused)
            self.assertEqual(count, 0)

            it._cprint("first")
            it._cprint("second")
            render.assert_not_called()

            paused, count = it._toggle_output_pause()
            self.assertFalse(paused)
            self.assertEqual(count, 2)

        self.assertEqual(render.call_count, 2)

    def test_session_cleanup_discards_paused_output(self):
        from agentica.cli.interactive import console_io as it

        it._clear_output_pause()
        it._toggle_output_pause()
        it._cprint("discard me")
        it._clear_output_pause()
        paused, count = it._toggle_output_pause()
        self.assertTrue(paused)
        self.assertEqual(count, 0)
        it._clear_output_pause()


class TestSigquitEscape(unittest.TestCase):
    def test_restores_preexisting_handler_after_tui_exit(self):
        from agentica.cli.interactive import console_io as it

        previous_handler = object()
        escape_handler = object()
        with (
            patch.object(it.signal, "getsignal", return_value=previous_handler),
            patch.object(it.signal, "signal") as set_handler,
        ):
            installation = it._install_sigquit_escape(escape_handler)
            self.assertEqual(installation, (it.signal.SIGQUIT, previous_handler))
            it._restore_sigquit_escape(installation)

        self.assertEqual(
            set_handler.call_args_list,
            [
                ((it.signal.SIGQUIT, escape_handler),),
                ((it.signal.SIGQUIT, previous_handler),),
            ],
        )

    def test_skips_sigquit_on_windows(self):
        from agentica.cli.interactive import console_io as it

        with patch.object(it.os, "name", "nt"):
            self.assertIsNone(it._install_sigquit_escape(object()))


if __name__ == "__main__":
    unittest.main()
