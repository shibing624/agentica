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



class TestCLIStreamDisplay(unittest.TestCase):
    """CLI helpers tests (TestCLIStreamDisplay)."""

    @staticmethod
    def _render_compact_event(event):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        StreamDisplayManager(fake).handle_event(event)
        return "\n".join(str(call) for call in fake.print.call_args_list)


    def test_stream_display_manager_no_gutter_and_short_separator(self):
        """Assistant turn should render as plain text (no left-side gutter
        bar), and close with a fixed-width ``──── HH:MM:SS ────`` separator
        rather than a full-width ``rich.rule.Rule``.

        Uses a real ``Console`` writing to StringIO so we can inspect the
        actual rendered characters.
        """
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=80, force_terminal=False, no_color=True)
        dm = StreamDisplayManager(con)
        dm.start_response()
        dm.stream_response("hello world")
        dm.finalize()
        out = buf.getvalue()
        self.assertNotIn("╭", out)
        self.assertNotIn("╰", out)
        self.assertNotIn("▏", out)
        self.assertIn("hello world", out)
        # Closing separator: fixed short edges + timestamp
        self.assertIn("────", out, "closing separator must have ──── edges")
        self.assertRegex(out, r"\d{2}:\d{2}:\d{2}", "closing separator must embed HH:MM:SS")
        # And crucially the separator must be short — not stretch full width.
        # 80-col terminal, separator should be under ~40 chars total.
        sep_lines = [ln for ln in out.splitlines() if "────" in ln]
        self.assertTrue(sep_lines, "separator line must exist")
        # The rendered line (without ANSI) should be substantially shorter
        # than the console width — fixed 4+1+summary+1+4 layout.
        self.assertLess(len(sep_lines[-1]), 60, "separator must be fixed-width, not stretch to full console width")


    def test_stream_display_manager_suppresses_evict_events(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.handle_event({"type": "compact.evict", "agent_name": "Agent", "evicted": 3})
        fake.print.assert_not_called()


    def test_stream_display_manager_renders_markdown_when_setting_on(self):
        from agentica.cli.display import StreamDisplayManager
        from rich.markdown import Markdown

        fake = MagicMock()
        fake.width = 80
        with patch("agentica.cli.display.stream.get_setting", return_value="on"):
            dm = StreamDisplayManager(fake)
        dm.start_response()
        dm.stream_response("# Title\n\n- item\n")
        dm.finalize()

        markdown_calls = [c for c in fake.print.call_args_list if c.args and isinstance(c.args[0], Markdown)]
        self.assertTrue(markdown_calls, "expected final response to render as Markdown")


    def test_stream_display_manager_keeps_plain_text_when_setting_off(self):
        from agentica.cli.display import StreamDisplayManager
        from rich.markdown import Markdown

        fake = MagicMock()
        fake.width = 80
        with patch("agentica.cli.display.stream.get_setting", return_value="off"):
            dm = StreamDisplayManager(fake)
        dm.start_response()
        dm.stream_response("# Title\n\n- item\n")
        dm.finalize()

        markdown_calls = [c for c in fake.print.call_args_list if c.args and isinstance(c.args[0], Markdown)]
        self.assertFalse(markdown_calls, "plain-text mode must not render Markdown")


    def test_stream_display_manager_buffers_incomplete_markdown_until_finalize(self):
        """An incomplete last block stays buffered; finalize flushes it.

        A heading with no following block is still open, so it must not land
        token-by-token. Uses a real ``Console`` (StringIO-backed).
        """
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=80, force_terminal=False, no_color=True)
        with patch("agentica.cli.display.stream.get_setting", return_value="on"):
            dm = StreamDisplayManager(con)

        dm.stream_response("# Title")
        pre_final = buf.getvalue()
        self.assertNotIn("Title", pre_final, "incomplete last block must stay buffered")

        dm.finalize()
        post_final = buf.getvalue()
        self.assertIn("Title", post_final, "finalize must flush the buffered markdown")
        self.assertNotIn("▏", post_final)


    def test_chatconsole_markdown_link_does_not_leak_osc8_payload(self):
        """Rich hyperlinks must not become visible ``8;id=...`` garbage.

        Rich renders Markdown links as OSC 8 terminal sequences, but
        prompt_toolkit's ANSI parser treats the OSC payload as ordinary text.
        The CLI adapter must remove the wrapper while preserving the label and
        regular ANSI styling.
        """
        from rich.markdown import Markdown
        from agentica.cli.interactive.console_io import ChatConsole

        rendered = []

        def capture(formatted_text):
            rendered.extend(formatted_text.__pt_formatted_text__())

        console = ChatConsole()
        with patch("agentica.cli.interactive.console_io.print_formatted_text", side_effect=capture):
            console.print(
                Markdown(
                    "入口位于 "
                    "[`reader.py:42`](/apdcephfs/share/dual_mem/retrieval/reader.py:42)"
                )
            )

        visible = "".join(fragment[1] for fragment in rendered)
        self.assertIn("入口位于 reader.py:42", visible)
        self.assertNotIn("8;id=", visible)
        self.assertNotIn("/apdcephfs/", visible)


    def test_fmt_elapsed_hides_fast_calls(self):
        """Fast tools render no timing; only slow calls (>= 1s) surface it.
        A fast grep/read_file reporting "(13ms)" is pure noise — the 1s
        cutoff keeps quick commands clean while long-running execute tasks
        still show their cost."""
        from agentica.cli.display import StreamDisplayManager

        f = StreamDisplayManager._fmt_elapsed
        # None / negative — no measurement, render nothing
        self.assertEqual(f(None), "")
        self.assertEqual(f(-0.1), "")
        # Sub-second — hidden
        self.assertEqual(f(0.0), "")
        self.assertEqual(f(0.123), "")
        self.assertEqual(f(0.999), "")
        # 1s..10s — 2 decimals
        self.assertEqual(f(1.0), " (1.00s)")
        self.assertEqual(f(1.234), " (1.23s)")
        self.assertEqual(f(9.99), " (9.99s)")
        # >= 10s — 1 decimal
        self.assertEqual(f(10.0), " (10.0s)")

    def test_fmt_elapsed_execute_hides_under_10s(self):
        """execute commands under 10s render no timing — compiles/tests
        legitimately take seconds, so the number is noise. Other tools keep
        the 1s cutoff."""
        from agentica.cli.display import StreamDisplayManager

        f = StreamDisplayManager._fmt_elapsed
        # execute: anything under 10s is hidden
        self.assertEqual(f(0.5, tool_name="execute"), "")
        self.assertEqual(f(1.0, tool_name="execute"), "")
        self.assertEqual(f(9.99, tool_name="execute"), "")
        # execute: 10s and up shows (1-decimal bucket)
        self.assertEqual(f(10.0, tool_name="execute"), " (10.0s)")
        self.assertEqual(f(65.43, tool_name="execute"), " (65.4s)")
        # other tools unchanged; no tool name defaults to the 1s cutoff
        self.assertEqual(f(1.0, tool_name="grep"), " (1.00s)")
        self.assertEqual(f(9.99, tool_name="grep"), " (9.99s)")
        self.assertEqual(f(1.0), " (1.00s)")
        self.assertEqual(f(123.456), " (123.5s)")


    def test_main_auto_compact_warns_and_points_to_new(self):
        output = self._render_compact_event({
            "type": "compact.auto",
            "is_main_agent": True,
            "compaction_count": 1,
        })

        self.assertIn("automatically compacted", output)
        self.assertIn("/new", output)
        self.assertNotIn("/resume", output)


    def test_repeated_main_auto_compact_escalates_warning(self):
        output = self._render_compact_event({
            "type": "compact.auto",
            "is_main_agent": True,
            "compaction_count": 2,
        })

        self.assertIn("auto-compacted 2 times", output)
        self.assertIn("summaries accumulate", output)


    def test_subagent_auto_compact_keeps_technical_notice(self):
        output = self._render_compact_event({
            "type": "compact.auto",
            "is_main_agent": False,
            "before": 20,
            "after": 4,
            "elapsed": 1.0,
        })

        self.assertIn("auto / LLM-summarised", output)
        self.assertNotIn("/new", output)


    def test_main_reactive_compact_warns_before_retry(self):
        output = self._render_compact_event({"type": "compact.reactive", "is_main_agent": True})

        self.assertIn("exceeded the model limit", output)
        self.assertIn("before retrying", output)
        self.assertIn("/new", output)


    def test_image_only_message_has_no_empty_gutter_line(self):
        """An image-only turn starts directly with its attachment label."""
        from pathlib import Path

        from agentica.cli.display import display_user_message

        console = Mock()
        with (
            patch("agentica.cli.display.messages.get_console", return_value=console),
            patch("pathlib.Path.stat") as stat,
        ):
            stat.return_value.st_size = 1024
            display_user_message("", images=[Path("/tmp/clip.png")])

        renderable = console.print.call_args.args[0]
        marker_column, content_column = renderable.renderable.columns
        self.assertEqual(marker_column._cells[0].plain, "❯")
        self.assertEqual(content_column._cells[0].plain, "📎 Image #1 attached: clip.png (1KB)")


    def test_truncated_blocks_are_remembered_for_expand(self):
        """Long execute output is stashed for Ctrl+O; user input is shown in full (not stashed)."""
        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        # Long user input (>20 lines) is NOT remembered — it is rendered in
        # full inline, so there is nothing to fold behind Ctrl+O.
        disp.clear_truncated_blocks()
        long_input = "\n".join(f"line {i}" for i in range(20))
        disp.display_user_message(long_input)
        block = disp.get_last_truncated()
        self.assertEqual(block["content"], "")

        # Long execute output IS remembered, with the launch command.
        long_output = "\n".join(f"out {i}" for i in range(50))
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "execute", long_output, is_error=False, elapsed=0.5,
            tool_args={"command": "pytest -q"}, tool_call_id="exec-1",
        )
        block = disp.get_last_truncated()
        self.assertEqual(block["title"], "execute")
        self.assertIn("$ pytest -q", block["content"])
        self.assertIn("out 0", block["content"])
        self.assertIn("out 49", block["content"])

        # Only the execute block is remembered (the query was shown in full).
        blocks = disp.get_truncated_blocks()
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["title"], "execute")

        # Short output is NOT remembered (no truncation → nothing to expand).
        disp.clear_truncated_blocks()
        dm.display_tool_result("execute", "only one line", is_error=False, elapsed=0.1)
        self.assertEqual(disp.get_last_truncated()["content"], "")


    def test_truncated_blocks_list_supports_expand_all(self):
        """Multiple folded blocks accumulate so Ctrl+O can expand ALL of them."""
        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        disp.clear_truncated_blocks()
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("execute", "\n".join(f"a{i}" for i in range(50)), is_error=False, elapsed=0.1)
        dm.display_tool_result("execute", "\n".join(f"b{i}" for i in range(50)), is_error=False, elapsed=0.1)
        blocks = disp.get_truncated_blocks()
        self.assertEqual(len(blocks), 2)
        self.assertIn("a0", blocks[0]["content"])
        self.assertIn("b0", blocks[1]["content"])


class TestSafeCommitEnd(unittest.TestCase):
    """Fence-aware blank-line boundary used to commit complete Markdown blocks."""

    def test_no_boundary_without_blank_line_or_leftover(self):
        from agentica.cli.display.stream import _safe_commit_end

        self.assertEqual(_safe_commit_end(""), 0)
        self.assertEqual(_safe_commit_end("hello"), 0)
        self.assertEqual(_safe_commit_end("hello\nworld"), 0)
        self.assertEqual(_safe_commit_end("# Title\n\n"), 0)

    def test_blank_line_commits_when_leftover_follows(self):
        from agentica.cli.display.stream import _safe_commit_end

        text = "# Title\n\n- item"
        self.assertEqual(_safe_commit_end(text), len("# Title\n\n"))

    def test_list_items_without_blank_line_stay_one_block(self):
        from agentica.cli.display.stream import _safe_commit_end

        self.assertEqual(_safe_commit_end("- a\n- b\n- c"), 0)
        text = "- a\n- b\n\nNext"
        self.assertEqual(_safe_commit_end(text), len("- a\n- b\n\n"))

    def test_unclosed_fence_is_not_split(self):
        from agentica.cli.display.stream import _safe_commit_end

        self.assertEqual(_safe_commit_end("```python\ncode\n\nstill"), 0)
        text = "intro\n\n```python\ncode\n\nstill"
        self.assertEqual(_safe_commit_end(text), len("intro\n\n"))

    def test_closed_fence_commits_when_leftover_follows(self):
        from agentica.cli.display.stream import _safe_commit_end

        closed = "```python\nx\n```\n"
        self.assertEqual(_safe_commit_end(closed), 0)
        text = closed + "\nnext"
        self.assertEqual(_safe_commit_end(text), len(closed + "\n"))

    def test_tilde_fence_matches_backtick_rules(self):
        from agentica.cli.display.stream import _safe_commit_end

        self.assertEqual(_safe_commit_end("~~~python\ncode\n\nstill"), 0)
        text = "~~~python\nx\n~~~\n\nnext"
        self.assertEqual(_safe_commit_end(text), len("~~~python\nx\n~~~\n\n"))

    def test_table_stays_together_until_blank_line(self):
        from agentica.cli.display.stream import _safe_commit_end

        table = "| a | b |\n|---|---|\n| 1 | 2 |"
        self.assertEqual(_safe_commit_end(table), 0)
        text = table + "\n\nnext"
        self.assertEqual(_safe_commit_end(text), len(table + "\n\n"))


class TestIncrementalMarkdownCommit(unittest.TestCase):
    """Complete Markdown blocks print during the stream; the tail waits."""

    def _mgr(self, setting="on"):
        from io import StringIO
        from rich.console import Console
        from agentica.cli.display import StreamDisplayManager

        buf = StringIO()
        con = Console(file=buf, width=80, force_terminal=False, no_color=True)
        with patch("agentica.cli.display.stream.get_setting", return_value=setting):
            dm = StreamDisplayManager(con)
        return dm, buf

    def test_heading_commits_when_next_block_starts(self):
        dm, buf = self._mgr()
        dm.stream_response("# Title\n\n")
        self.assertNotIn("Title", buf.getvalue())
        dm.stream_response("body still growing")
        self.assertIn("Title", buf.getvalue())
        self.assertNotIn("body still growing", buf.getvalue())
        dm.finalize()
        self.assertIn("body still growing", buf.getvalue())

    def test_unclosed_fence_waits_until_closed_or_finalize(self):
        dm, buf = self._mgr()
        dm.stream_response("```python\nprint(1)\n")
        self.assertNotIn("print(1)", buf.getvalue())
        dm.stream_response("print(2)\n")
        self.assertNotIn("print(2)", buf.getvalue())
        dm.finalize()
        out = buf.getvalue()
        self.assertIn("print(1)", out)
        self.assertIn("print(2)", out)

    def test_closed_fence_commits_before_following_paragraph(self):
        dm, buf = self._mgr()
        dm.stream_response("```python\nx = 1\n```\n\n")
        self.assertNotIn("x = 1", buf.getvalue())
        dm.stream_response("after")
        self.assertIn("x = 1", buf.getvalue())
        self.assertNotIn("after", buf.getvalue())
        dm.finalize()
        self.assertIn("after", buf.getvalue())

    def test_list_stays_together_until_blank_line(self):
        dm, buf = self._mgr()
        dm.stream_response("- a\n- b\n- c\n")
        self.assertNotIn("- a", buf.getvalue())
        dm.stream_response("\nNext para")
        out = buf.getvalue()
        self.assertIn("a", out)
        self.assertIn("b", out)
        self.assertIn("c", out)
        self.assertNotIn("Next para", out)

    def test_plain_mode_does_not_commit_markdown_mid_stream(self):
        dm, buf = self._mgr(setting="off")
        dm.stream_response("# Title\n\nbody")
        self.assertNotIn("Title", buf.getvalue())
        dm.finalize()
        self.assertIn("Title", buf.getvalue())

    def test_finalize_does_not_reprint_committed_blocks(self):
        dm, buf = self._mgr()
        dm.stream_response("# Title\n\nbody")
        mid = buf.getvalue()
        self.assertEqual(mid.count("Title"), 1)
        dm.finalize()
        self.assertEqual(buf.getvalue().count("Title"), 1)


if __name__ == "__main__":
    unittest.main()
