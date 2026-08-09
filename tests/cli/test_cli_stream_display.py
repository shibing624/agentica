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
        # No box glyphs — gutter design was itself replaced by plain text
        self.assertNotIn("╭", out)
        self.assertNotIn("╰", out)
        # Assistant gutter must NOT appear on the streamed line anymore
        self.assertNotIn("▏", out, "assistant ▏ gutter has been removed")
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


    def test_stream_display_manager_buffers_markdown_stream_until_finalize(self):
        """Markdown mode buffers the streamed text and only renders on finalize.

        Uses a real ``Console`` (StringIO-backed) instead of MagicMock because
        the gutter proxy needs a working ``capture()`` to inspect rendered
        ANSI. Assertions target the visible transcript, not mock call args.
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
        self.assertNotIn("Title", pre_final, "markdown mode should buffer stream text until finalize")

        dm.finalize()
        post_final = buf.getvalue()
        self.assertIn("Title", post_final, "finalize must flush the buffered markdown")
        # Assistant ▏ gutter no longer decorates markdown — plain output
        self.assertNotIn("▏", post_final)


    def test_gutter_console_works_with_chatconsole(self):
        """Regression: _GutteredConsole must not blow up when wrapping the
        CLI's ChatConsole. ChatConsole is a slim adapter (used inside the
        prompt_toolkit app) — it exposes ``render_ansi`` and ``print`` but
        NOT ``rich.Console.capture``. Earlier the gutter proxy hard-coded
        ``self._console.capture()``, raising ``AttributeError`` on the
        first ask_user_question turn inside ``process_loop``.
        """
        from agentica.cli.display.console import _GutteredConsole
        from agentica.cli.interactive.console_io import ChatConsole

        cc = ChatConsole()
        gutter_con = _GutteredConsole(cc, "▎", "cyan")
        # Should NOT raise
        gutter_con.print("hello from ChatConsole gutter")
        # Prefix cache should be a string (ANSI-rendered by ``render_ansi``)
        self.assertIsInstance(gutter_con.gutter_prefix_ansi, str)
        self.assertIn("▎", gutter_con.gutter_prefix_ansi)


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


    def test_fmt_elapsed_uses_ms_under_one_second(self):
        """Sub-second tools must surface ms-precision rather than being hidden.
        Every tool call has a real cost — silent <0.1s suppression made fast
        ops look like they didn't run."""
        from agentica.cli.display import StreamDisplayManager

        f = StreamDisplayManager._fmt_elapsed
        # None / negative — no measurement, render nothing
        self.assertEqual(f(None), "")
        self.assertEqual(f(-0.1), "")
        # Sub-millisecond — still surface a signal
        self.assertEqual(f(0.0), " (<1ms)")
        self.assertEqual(f(0.0005), " (<1ms)")
        # Milliseconds — integer ms
        self.assertEqual(f(0.001), " (1ms)")
        self.assertEqual(f(0.005), " (5ms)")
        self.assertEqual(f(0.123), " (123ms)")
        self.assertEqual(f(0.999), " (999ms)")
        # 1s..10s — 2 decimals
        self.assertEqual(f(1.0), " (1.00s)")
        self.assertEqual(f(1.234), " (1.23s)")
        self.assertEqual(f(9.99), " (9.99s)")
        # >= 10s — 1 decimal
        self.assertEqual(f(10.0), " (10.0s)")
        self.assertEqual(f(123.456), " (123.5s)")


    def test_stream_display_manager_keeps_rule_based_compact_visible(self):
        output = self._render_compact_event({
            "type": "compact.rule_based",
            "agent_name": "Agent",
            "is_main_agent": True,
            "before": 20,
            "after": 8,
            "elapsed": 0.25,
        })
        self.assertIn("compact", output)


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

        # Long execute output (>20 lines) IS remembered.
        long_output = "\n".join(f"out {i}" for i in range(50))
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("execute", long_output, is_error=False, elapsed=0.5)
        block = disp.get_last_truncated()
        self.assertIn("execute", block["title"])
        self.assertEqual(block["content"], long_output)

        # Only the execute block is remembered (the query was shown in full).
        blocks = disp.get_truncated_blocks()
        self.assertEqual(len(blocks), 1)
        self.assertIn("execute", blocks[0]["title"])

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




if __name__ == "__main__":
    unittest.main()
