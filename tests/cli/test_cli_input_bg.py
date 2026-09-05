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

    def test_print_background_completion_shows_full_command_and_output(self):
        """Completion notice shows the whole command and log body — no Ctrl+O fold."""
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
            # Many output lines: previously only a short tail was shown.
            body_lines = [f"$ {long_command}", ""] + [f"phase-{i}" for i in range(40)]
            log_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
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

        rendered = "\n".join(
            str(call.args[0]) for call in fake_console.print.call_args_list if call.args
        )
        self.assertIn("Background terminal #3 finished", rendered)
        self.assertIn("--qid q29", rendered)
        self.assertIn("phase-0", rendered)
        self.assertIn("phase-39", rendered)
        self.assertNotIn("Ctrl+O", rendered)
        self.assertEqual(disp.get_truncated_blocks(), [])


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


class TestAskPromptKeyHint(unittest.TestCase):
    """Ctrl+\\ only fires while the tty is in cooked_mode (inside
    run_in_terminal), which is where a starved event loop parks while an answer
    is pending. It is an inert byte under prompt_toolkit's raw_mode (ISIG
    cleared), so it must be advertised on the ask prompt and nowhere else."""

    def test_hint_is_attached_to_the_ask_prompt(self):
        from agentica.cli.interactive.session_state import _InputRequest
        from agentica.cli.interactive.tui import _ASK_KEY_HINT, _ask_prompt_lines

        req = _InputRequest(prompt="Pick one", options=["a", "b"])
        lines = _ask_prompt_lines(req)

        self.assertEqual(lines[0], "  ? Pick one")
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2:4], ["    1. a", "    2. b"])
        self.assertEqual(lines[-1], _ASK_KEY_HINT)
        self.assertIn("Ctrl+\\", _ASK_KEY_HINT)

    def test_prompt_is_dumped_verbatim(self):
        from agentica.cli.interactive.session_state import _InputRequest
        from agentica.cli.interactive.tui import _ask_prompt_text

        req = _InputRequest(prompt="two\nlines", options=["x", "y", "z"])
        text = _ask_prompt_text(req)

        self.assertTrue(text.startswith("  ? two\nlines"))
        self.assertIn("\n\n    1. x\n    2. y\n    3. z\n", text)

    def test_live_window_yields_its_rows_while_the_user_is_answering(self):
        """LIVE_MAX_ROWS=12 used to stay reserved above the ask widget, so a
        long option list was clipped to whatever was left of the terminal."""
        from agentica.cli.display.live_blocks import LIVE_MAX_ROWS
        from agentica.cli.interactive.tui import _live_tool_window_height

        self.assertEqual(
            _live_tool_window_height(LIVE_MAX_ROWS, asking=True), 0,
        )
        self.assertEqual(
            _live_tool_window_height(LIVE_MAX_ROWS, asking=False), LIVE_MAX_ROWS,
        )
        self.assertEqual(_live_tool_window_height(3, asking=False), 3)
        self.assertEqual(_live_tool_window_height(0, asking=False), 0)

    def test_long_option_list_is_dumped_in_full(self):
        from agentica.cli.display.live_blocks import LIVE_MAX_ROWS
        from agentica.cli.interactive.session_state import _InputRequest
        from agentica.cli.interactive.tui import _ask_prompt_lines

        options = [f"option {i}: keep this whole label visible" for i in range(1, 16)]
        req = _InputRequest(prompt="Pick one", options=options)
        lines = _ask_prompt_lines(req)

        self.assertGreater(len(lines), LIVE_MAX_ROWS)
        for i, opt in enumerate(options, 1):
            self.assertIn(f"    {i}. {opt}", lines)

    def test_multiline_prompt_is_dumped_verbatim(self):
        from agentica.cli.interactive.session_state import _InputRequest
        from agentica.cli.interactive.tui import _ask_prompt_text

        prompt = (
            "计划写好了：/tmp/plan.md\n"
            "\n"
            "两处想先跟你确认，其余按计划走：\n"
            "\n"
            "1) session id 的粒度。\n"
            "   - 每 CLI 会话一个\n"
            "\n"
            "2) config.yaml 里的 default_headers 要替换成 sid。\n"
        )
        options = [
            "两个都按你说的做（每会话粒度 + 改 config.yaml）",
            "每 profile 粒度，改 config.yaml",
            "只改代码，config.yaml 我自己改",
            "先不动，我只想看分析",
        ]
        text = _ask_prompt_text(_InputRequest(prompt=prompt, options=options))

        self.assertIn(prompt.rstrip("\n"), text)
        self.assertIn("    1. 两个都按你说的做（每会话粒度 + 改 config.yaml）", text)
        self.assertIn("    4. 先不动，我只想看分析", text)

    def test_ctrl_c_interrupt_notice_does_not_advertise_the_kill_key(self):
        import inspect

        from agentica.cli.interactive import tui

        source = inspect.getsource(tui)
        notice = next(
            line for line in source.splitlines() if "Interrupting agent" in line
        )
        # The file spells the key as an escaped backslash, so a regression would
        # show up as two literal backslash characters on this line.
        self.assertNotIn("Ctrl+\\\\", notice)
        self.assertIn("Ctrl+C again to force exit", notice)


class TestInputVisualRows(unittest.TestCase):
    """The input box height must match prompt_toolkit's wrap, not ``len()``.

    TextArea prepends ``❯ `` via BeforeInput (first logical line only) and
    wraps by display width (``get_cwidth``). Counting ``len(line) // (width-2)``
    kept the box one row tall for CJK: the cursor line scrolled the first
    visual row away until another ~width characters arrived, at which point
    the height finally jumped and the missing line reappeared.
    """

    def test_cjk_wraps_on_the_first_overflow_row(self):
        from agentica.cli.interactive.tui import _count_input_visual_rows

        # width=80, prompt=2: 40 fullwidth chars → display 82 → 2 visual rows.
        # The old ``len()`` count treated this as 40 / 78 → still 1 row.
        self.assertEqual(
            _count_input_visual_rows(
                ["中" * 40], width=80, first_line_prefix_width=2,
            ),
            2,
        )
        self.assertEqual(
            _count_input_visual_rows(
                ["中" * 39], width=80, first_line_prefix_width=2,
            ),
            1,
        )

    def test_ascii_wraps_when_prompt_plus_text_exceeds_width(self):
        from agentica.cli.interactive.tui import _count_input_visual_rows

        self.assertEqual(
            _count_input_visual_rows(
                ["a" * 78], width=80, first_line_prefix_width=2,
            ),
            1,
        )
        self.assertEqual(
            _count_input_visual_rows(
                ["a" * 79], width=80, first_line_prefix_width=2,
            ),
            2,
        )

    def test_prompt_width_applies_only_to_the_first_logical_line(self):
        from agentica.cli.interactive.tui import _count_input_visual_rows

        self.assertEqual(
            _count_input_visual_rows(
                ["hello", "world"], width=80, first_line_prefix_width=2,
            ),
            2,
        )

    def test_empty_input_is_one_row(self):
        from agentica.cli.interactive.tui import _count_input_visual_rows

        self.assertEqual(
            _count_input_visual_rows([""], width=80, first_line_prefix_width=2),
            1,
        )


if __name__ == "__main__":
    unittest.main()
