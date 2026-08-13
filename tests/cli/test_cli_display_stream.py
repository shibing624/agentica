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



class TestToolResultAnchoring(unittest.TestCase):
    """A ``⎿`` result block must name its own call when it is not adjacent to it.

    Deferred tools (``read_file`` / ``grep`` / ...) render as ONE atomic block at
    completion, while ``execute`` prints its call line at START time and its
    result body at completion. In a parallel batch where a deferred tool is
    called FIRST, the deferred block lands between ``execute``'s call line and
    ``execute``'s body — so the ``⎿`` block reads as the deferred tool's result.
    """

    def _mgr_and_buf(self):
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=120, force_terminal=False, no_color=True)
        return StreamDisplayManager(con), buf

    def _render_read_then_execute(self):
        """Batch call order: read_file (deferred) then execute; in-order results."""
        mgr, buf = self._mgr_and_buf()
        mgr.display_tool("read_file", {"file_path": "/proj/working.py",
                                       "offset": 284, "limit": 30},
                         tool_call_id="c1")
        mgr.display_tool("execute", {"command": "grep -n needle cost_tracker.py"},
                         tool_call_id="c2")
        mgr.display_tool_result("read_file", "\n".join(f"line {i}" for i in range(34)),
                                elapsed=0.058, tool_args={"file_path": "/proj/working.py"},
                                tool_call_id="c1")
        mgr.display_tool_result("execute", "cost_tracker.py:357: needle here",
                                elapsed=0.025,
                                tool_args={"command": "grep -n needle cost_tracker.py"},
                                tool_call_id="c2")
        return [ln for ln in buf.getvalue().splitlines() if ln.strip()]

    def test_detached_execute_result_is_anchored_to_its_own_call(self):
        lines = self._render_read_then_execute()
        body_idx = next(i for i, ln in enumerate(lines) if "needle here" in ln)
        # The line directly above the result body must identify execute, not
        # read_file — otherwise the block reads as read_file's output.
        anchor = lines[body_idx - 1]
        self.assertIn("execute", anchor,
                      f"result body must be anchored to execute, got: {anchor!r}")
        self.assertNotIn("read_file", anchor)

    def test_adjacent_result_is_not_anchored_twice(self):
        """The common case (call line immediately followed by its own result)
        must stay byte-identical — no redundant re-statement of the call."""
        mgr, buf = self._mgr_and_buf()
        mgr.display_tool("execute", {"command": "ls"}, tool_call_id="c1")
        mgr.display_tool_result("execute", "a.py\nb.py", elapsed=0.01,
                                tool_args={"command": "ls"}, tool_call_id="c1")
        out = buf.getvalue()
        self.assertEqual(out.count("execute"), 1,
                         "execute must be named once when its result is adjacent")

    def test_deferred_result_keeps_single_merged_line(self):
        """Deferred tools are self-labelled already — never anchor them."""
        mgr, buf = self._mgr_and_buf()
        mgr.display_tool("execute", {"command": "ls"}, tool_call_id="c1")
        mgr.display_tool_result("execute", "a.py", elapsed=0.01,
                                tool_args={"command": "ls"}, tool_call_id="c1")
        mgr.display_tool_result("grep", "hit\nhit2", elapsed=0.02,
                                tool_args={"pattern": "x", "path": "p"},
                                tool_call_id="c2")
        out = buf.getvalue()
        self.assertEqual(out.count("grep"), 1,
                         "deferred merged line must not gain an extra anchor line")


class TestStreamDisplayManagerSubagent(unittest.TestCase):
    """Subagent rendering policy: tool-first by default, dedup, batch prefix."""

    def _make(self, verbosity: str = "all"):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        return fake, StreamDisplayManager(fake, subagent_verbosity=verbosity)

    @staticmethod
    def _printed(fake) -> str:
        return "\n".join(str(c) for c in fake.print.call_args_list)

    def test_default_renders_tool_started_not_completed(self):
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "explore", "task": "look"})
        dm.handle_event(
            {
                "type": "subagent.tool_started",
                "run_id": "r1",
                "agent_name": "explore",
                "tool_name": "read_file",
                "info": "a.py",
                "args": {},
            }
        )
        dm.handle_event(
            {
                "type": "subagent.tool_completed",
                "run_id": "r1",
                "agent_name": "explore",
                "tool_name": "read_file",
                "info": "a.py",
                "elapsed": 0.5,
                "is_error": False,
            }
        )
        out = self._printed(fake)
        self.assertIn("read_file", out, "tool_started must render in default mode")
        self.assertNotIn("✓", out, "tool_completed checkmark must not render in default mode")

    def test_default_dedups_consecutive_identical_tool(self):
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "explore", "task": "loop"})
        for _ in range(3):
            dm.handle_event(
                {
                    "type": "subagent.tool_started",
                    "run_id": "r1",
                    "agent_name": "explore",
                    "tool_name": "read_file",
                    "info": "a.py",
                    "args": {},
                }
            )
        out = self._printed(fake)
        self.assertEqual(out.count("read_file"), 1, "consecutive identical tool calls must collapse to one line")

    def test_default_does_not_dedup_when_args_change(self):
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "explore", "task": "loop"})
        for path in ("a.py", "b.py", "c.py"):
            dm.handle_event(
                {
                    "type": "subagent.tool_started",
                    "run_id": "r1",
                    "agent_name": "explore",
                    "tool_name": "read_file",
                    "info": path,
                    "args": {},
                }
            )
        out = self._printed(fake)
        self.assertEqual(out.count("read_file"), 3, "different args must each render their own line")

    def test_concurrent_subagents_get_index_prefix(self):
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "a", "task": "t1"})
        dm.handle_event({"type": "subagent.start", "run_id": "r2", "agent_name": "b", "task": "t2"})
        dm.handle_event(
            {
                "type": "subagent.tool_started",
                "run_id": "r1",
                "agent_name": "a",
                "tool_name": "glob",
                "info": "*.py",
                "args": {},
            }
        )
        dm.handle_event(
            {
                "type": "subagent.tool_started",
                "run_id": "r2",
                "agent_name": "b",
                "tool_name": "glob",
                "info": "*.md",
                "args": {},
            }
        )
        out = self._printed(fake)
        self.assertIn("[1]", out, "first concurrent subagent must get [1] prefix")
        self.assertIn("[2]", out, "second concurrent subagent must get [2] prefix")

    def test_single_subagent_has_no_index_prefix(self):
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "solo", "task": "t"})
        dm.handle_event(
            {
                "type": "subagent.tool_started",
                "run_id": "r1",
                "agent_name": "solo",
                "tool_name": "glob",
                "info": "*.py",
                "args": {},
            }
        )
        out = self._printed(fake)
        self.assertNotIn("[1]", out, "single subagent must not get noisy [N] prefix")

    def test_verbose_mode_renders_completion_with_elapsed(self):
        fake, dm = self._make("verbose")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "x", "task": "t"})
        dm.handle_event(
            {
                "type": "subagent.tool_completed",
                "run_id": "r1",
                "agent_name": "x",
                "tool_name": "read_file",
                "info": "a.py",
                "elapsed": 1.234,
                "is_error": False,
            }
        )
        out = self._printed(fake)
        self.assertIn("✓", out, "verbose mode must render completion checkmark")
        self.assertIn("1.2", out, "verbose mode must surface elapsed time")

    def test_off_mode_suppresses_intermediate_events_but_keeps_end(self):
        fake, dm = self._make("off")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "x", "task": "t"})
        dm.handle_event(
            {
                "type": "subagent.tool_started",
                "run_id": "r1",
                "agent_name": "x",
                "tool_name": "read_file",
                "info": "a.py",
                "args": {},
            }
        )
        dm.handle_event(
            {"type": "subagent.end", "run_id": "r1", "agent_name": "x", "response": "done", "tool_count": 1}
        )
        out = self._printed(fake)
        self.assertNotIn("read_file", out, "off mode must hide intermediate tools")
        self.assertNotIn("⮕", out, "off mode must hide start banner")
        self.assertIn("done", out, "off mode must still surface the final response")

    def test_default_still_surfaces_tool_errors_even_at_completion(self):
        # is_error completions are exempt from the "hide completed" policy:
        # silent failures are worse than slightly noisier output.
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "x", "task": "t"})
        dm.handle_event(
            {
                "type": "subagent.tool_completed",
                "run_id": "r1",
                "agent_name": "x",
                "tool_name": "exec",
                "info": "boom",
                "elapsed": 0.1,
                "is_error": True,
            }
        )
        out = self._printed(fake)
        self.assertIn("⚠", out, "errors must surface even in default mode")
        self.assertIn("exec", out)

    def test_subagent_slot_reclaimed_on_end(self):
        fake, dm = self._make("all")
        dm.handle_event({"type": "subagent.start", "run_id": "r1", "agent_name": "a", "task": "t1"})
        dm.handle_event({"type": "subagent.end", "run_id": "r1", "agent_name": "a", "response": "x", "tool_count": 0})
        # New subagent — only one active at a time again, so no [N] prefix.
        dm.handle_event({"type": "subagent.start", "run_id": "r2", "agent_name": "b", "task": "t2"})
        dm.handle_event(
            {
                "type": "subagent.tool_started",
                "run_id": "r2",
                "agent_name": "b",
                "tool_name": "glob",
                "info": "*.py",
                "args": {},
            }
        )
        out = self._printed(fake)
        self.assertNotIn("[2]", out)
        self.assertNotIn("[1]", out)


class TestSuppressConsoleLogging(unittest.TestCase):
    def test_suppress_console_logging_removes_all_non_file_stream_handlers(self):
        from agentica.utils.log import logger, suppress_console_logging

        original_handlers = list(logger.handlers)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        stdout_handler = logging.StreamHandler(sys.stdout)
        stderr_handler = logging.StreamHandler(sys.stderr)
        file_handler = logging.FileHandler(temp_file.name)

        try:
            logger.handlers = [stdout_handler, stderr_handler, file_handler]
            suppress_console_logging()
            self.assertFalse(
                any(
                    isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
                    for handler in logger.handlers
                )
            )
            self.assertTrue(any(isinstance(handler, logging.FileHandler) for handler in logger.handlers))
        finally:
            for handler in [stdout_handler, stderr_handler, file_handler]:
                handler.close()
            logger.handlers = original_handlers
            os.unlink(temp_file.name)


class TestStreamDisplayManagerCompletionTimestamp(unittest.TestCase):
    """The assistant turn must close with a dim rule whose body carries a
    compact per-turn summary in Plan A format:

        #N · HH:MM:SS · Xs · +Tk · +$C · N tools

    Users reviewing a long session can then see, for each turn, when it
    landed, how long it took (net), how much context it ate, how much it
    cost, and how many tools it fired. The status bar carries session
    totals; this rule carries per-turn deltas — zero overlap.
    """

    def _capture(self, render):
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=80, force_terminal=False, no_color=True)
        mgr = StreamDisplayManager(con)
        render(mgr)
        return buf.getvalue()

    def test_finalize_draws_rule_with_timestamp(self):
        def render(mgr):
            mgr.start_response()
            mgr.stream_response("hello")
            mgr.finalize()

        out = self._capture(render)
        # No box glyphs anywhere — gutter design replaced them
        self.assertNotIn("╭", out)
        self.assertNotIn("╰", out)
        self.assertNotIn("Response", out, "no 'Response' title — gutter design has no box header")
        # Rule glyph + timestamp on the closing line
        self.assertIn("─", out, "closing rule must be drawn")
        self.assertRegex(out, r"\d{2}:\d{2}:\d{2}", "rule must embed HH:MM:SS")

    def test_finalize_rule_includes_tool_count_and_elapsed(self):
        def render(mgr):
            # ``execute`` is NOT in ``_DEFERRED_TOOLS`` so it increments
            # ``tool_count`` at call time, giving finalize a non-zero count
            # to render in the summary.
            mgr.display_tool("execute", {"command": "ls"})
            mgr.stream_response("done")
            mgr.finalize()

        out = self._capture(render)
        self.assertRegex(out, r"1 tool\b", "rule must show N tools when >0")
        self.assertRegex(out, r"\d+\.\ds", "rule must show elapsed seconds")

    def test_finalize_rule_counts_deferred_and_write_tools(self):
        """Regression: ``read_file`` / ``grep`` (deferred) and ``edit_file`` /
        ``write_file`` (write-diff) used to be EXCLUDED from ``tool_count``
        because ``display_tool`` returned before incrementing. A turn that only
        read/edited files would then show "0 tools", contradicting the visible
        tool calls. Every tool call must count."""

        def render(mgr):
            mgr.display_tool("read_file", {"file_path": "a.py"})
            mgr.display_tool("grep", {"pattern": "x"})
            mgr.display_tool("edit_file", {"file_path": "a.py"})
            mgr.display_tool("write_file", {"file_path": "b.py", "content": "x"})
            mgr.stream_response("done")
            mgr.finalize()

        out = self._capture(render)
        self.assertRegex(out, r"4 tools\b", "all 4 deferred/write tools must count")

    def test_finalize_rule_shows_turn_number_and_deltas_when_provided(self):
        """Plan A: the closing separator carries per-turn deltas.

        When the interactive loop hands ``finalize`` a turn number, token
        delta and cost delta, they must appear as ``#N``, ``+Tk`` and
        ``+$C`` respectively so a user scrolling back can locate turn #7
        and see it cost 3.2K tokens / $0.08.
        """

        def render(mgr):
            mgr.stream_response("ok")
            mgr.finalize(turn_no=7, delta_tokens=3200, delta_cost_usd=0.08)

        out = self._capture(render)
        self.assertIn("#7", out, "turn number must appear as #N")
        self.assertIn("+3.2K", out, "delta tokens >=1000 shown with K suffix")
        self.assertIn("+$0.08", out, "at or above one cent, cost shows 2 decimals")

    def test_finalize_rule_keeps_four_decimals_for_sub_cent_cost(self):
        """A sub-cent turn must not render as a free one.

        Guards the footer's wiring to ``format_cost_usd``, not just the
        formatter: a fixed ``:.2f`` here turns $0.004 into ``+$0.00``.
        """

        def render(mgr):
            mgr.stream_response("cheap")
            mgr.finalize(turn_no=3, delta_tokens=120, delta_cost_usd=0.004)

        out = self._capture(render)
        self.assertIn("+$0.0040", out)

    def test_finalize_rule_floors_unrenderable_cost_without_a_plus(self):
        """Below 4-decimal resolution the footer shows the floor, unsigned."""

        def render(mgr):
            mgr.stream_response("tiny")
            mgr.finalize(turn_no=4, delta_tokens=30, delta_cost_usd=0.000014)

        out = self._capture(render)
        self.assertIn("<$0.0001", out)
        self.assertNotIn("+<", out)
        self.assertNotIn("$0.0000", out)
    def test_finalize_rule_shows_provider_usage_breakdown_when_provided(self):
        from agentica.cli.usage_display import ProviderUsageSummary

        def render(mgr):
            mgr.stream_response("ok")
            mgr.finalize(
                turn_no=9,
                delta_cost_usd=0.04,
                usage_summary=ProviderUsageSummary(
                    input_tokens=38_100,
                    fresh_input_tokens=1_000,
                    cache_read_tokens=37_100,
                    output_tokens=3_000,
                    total_tokens_override=41_100,
                ),
            )

        out = self._capture(render)
        compact = " ".join(out.split())
        self.assertIn("#9", out)
        self.assertIn("+4K", out)
        self.assertNotIn("+4K tok", out)
        self.assertIn("in 38.1K", out)
        self.assertIn("cache 37.1K / 97.4%", out)
        self.assertIn("out 3K", compact)
        self.assertIn("+$0.04", out)

    def test_finalize_rule_uses_raw_count_for_small_token_deltas(self):
        """<1000 tokens: no K suffix — show the raw number.

        Guards the K-suffix boundary so a 42-token turn doesn't render as
        the misleading ``+0.0K``.
        """

        def render(mgr):
            mgr.stream_response("tiny")
            mgr.finalize(turn_no=1, delta_tokens=42, delta_cost_usd=0.0)

        out = self._capture(render)
        self.assertIn("+42", out)
        self.assertNotIn("+0.0K", out)
        # Zero cost must be suppressed to avoid a noisy "+$0.00" on
        # free/local models.
        self.assertNotIn("+$0.00", out)

    def test_finalize_rule_omits_optional_fields_when_none(self):
        """Backward-compat: callers that pass no delta info get the old
        skeleton (timestamp + elapsed [+ tool count]) — no phantom ``#None``
        or stray ``+`` markers.
        """

        def render(mgr):
            mgr.stream_response("plain")
            mgr.finalize()  # no kwargs at all

        out = self._capture(render)
        self.assertNotIn("#", out, "no turn number when caller omits it")
        self.assertNotIn("+", out, "no delta markers when caller omits them")
        self.assertRegex(out, r"\d{2}:\d{2}:\d{2}", "timestamp still present")


class TestStreamDisplayManagerSegmentOrdering(unittest.TestCase):
    """Preamble text must land in the LLM's NATIVE emission order.

    ``stream_response`` buffers text silently; the buffer is flushed as plain
    text at the next boundary that produces its own live output — thinking
    start OR tool start. Whatever remains at ``finalize`` is the final answer
    (rendered as Markdown). Regression guard for the ``text -> thinking ->
    tool`` inversion where the buffered preamble used to surface AFTER the
    thinking block (it was only flushed on a tool call, never on thinking).
    """

    def _mgr_and_buf(self):
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=80, force_terminal=False, no_color=True)
        return StreamDisplayManager(con), buf

    def test_text_before_thinking_is_flushed_before_thinking(self):
        mgr, buf = self._mgr_and_buf()
        mgr.stream_response("PREAMBLETEXT")
        mgr.start_thinking()
        mgr.stream_thinking("THINKINGLINE\n")
        mgr.end_thinking()
        mgr.finalize()
        out = buf.getvalue()
        self.assertIn("PREAMBLETEXT", out)
        self.assertIn("THINKINGLINE", out)
        self.assertLess(
            out.index("PREAMBLETEXT"),
            out.index("THINKINGLINE"),
            "preamble text must appear BEFORE the thinking it preceded",
        )

    def test_preamble_before_tool_is_flushed_before_tool(self):
        mgr, buf = self._mgr_and_buf()
        mgr.stream_response("PREAMBLETEXT")
        mgr.display_tool("execute", {"command": "ls"})
        mgr.finalize()
        out = buf.getvalue()
        self.assertIn("PREAMBLETEXT", out)
        self.assertIn("execute", out)
        self.assertLess(
            out.index("PREAMBLETEXT"),
            out.index("execute"),
            "preamble text must appear BEFORE the tool call it preceded",
        )

    def test_final_segment_stays_buffered_until_finalize(self):
        """The final answer is silent while streaming; it only lands when
        ``finalize`` renders it in one shot (spinner covers the wait)."""
        mgr, buf = self._mgr_and_buf()
        mgr.stream_response("FINALANSWER")
        self.assertNotIn(
            "FINALANSWER",
            buf.getvalue(),
            "final answer must not be printed token-by-token during streaming",
        )
        mgr.finalize()
        self.assertIn("FINALANSWER", buf.getvalue())


class TestReadFileCountSummary(unittest.TestCase):
    """_result_count_summary counts read_file content lines via the footer
    range, not the wrapped string (footers used to inflate the count)."""

    @staticmethod
    def _read_result(body_lines, total, offset=0):
        end = offset + len(body_lines)
        result = "\n".join(body_lines)
        if end < total:
            result += f"\n\n[Showing lines {offset + 1}-{end} of {total} total lines]"
        return result

    def test_counts_content_not_footers(self):
        from agentica.cli.display import StreamDisplayManager
        body = [f"{i:6d}\tline {i}" for i in range(1, 501)]
        # 504-line file read L1-500 → 500, not 504 (the old bug)
        res = self._read_result(body, total=504)
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", res), "500 lines")

    def test_short_file_read_with_large_limit(self):
        from agentica.cli.display import StreamDisplayManager
        body = [f"{i:6d}\tline {i}" for i in range(1, 101)]
        # Read L1-500 on a 100-line file → 100, not 500
        res = self._read_result(body, total=100)
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", res), "100 lines")

    def test_range_text_in_content_does_not_false_match(self):
        from agentica.cli.display import StreamDisplayManager
        body = ["     1\tprint('range=3-9')", "     2\tx = 1"]
        res = self._read_result(body, total=2)
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", res), "2 lines")

    def test_read_past_eof_clamps_to_zero(self):
        from agentica.cli.display import StreamDisplayManager
        res = "\n\n[Showing lines 601-504 of 504 total lines]"
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", res), "0 lines")

    def test_fallback_when_no_footer(self):
        from agentica.cli.display import StreamDisplayManager
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", "a\nb\nc"), "3 lines")

    def test_file_metadata_footer_not_counted(self):
        from agentica.cli.display import StreamDisplayManager
        res = "     1\ta\n     2\tb\n\n[File metadata: path=/tmp/a.py, size=4 bytes, mtime=2026-08-07T12:00:00]"
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", res), "2 lines")

    def test_truncated_read_with_metadata_footer_counts_content_span(self):
        from agentica.cli.display import StreamDisplayManager
        body = [f"{i:6d}\tline {i}" for i in range(1, 501)]
        res = self._read_result(body, total=504)
        res += "\n\n[File metadata: path=/tmp/a.py, size=1 bytes, mtime=2026-08-07T12:00:00]"
        self.assertEqual(StreamDisplayManager._result_count_summary("read_file", res), "500 lines")

    def test_grep_count_unchanged(self):
        from agentica.cli.display import StreamDisplayManager
        self.assertEqual(StreamDisplayManager._result_count_summary("grep", "m1\nm2"), "2 lines")
        self.assertEqual(StreamDisplayManager._result_count_summary("grep", ""), "no matches")


if __name__ == "__main__":
    unittest.main()
