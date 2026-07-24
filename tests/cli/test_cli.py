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
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli import commands as cli_commands
from agentica.cli import setup as cli_setup
from agentica.memory.session_log import SessionLog


class TestToolIcons(unittest.TestCase):
    """Test cases for TOOL_ICONS configuration."""

    def test_tool_icons_exists(self):
        """Test TOOL_ICONS dictionary exists."""
        self.assertIsInstance(TOOL_ICONS, dict)

    def test_default_icon_exists(self):
        """Test default icon exists."""
        self.assertIn("default", TOOL_ICONS)

    def test_common_icons_exist(self):
        """Test common tool icons exist."""
        expected_icons = ["read_file", "write_file", "execute", "web_search"]
        for icon in expected_icons:
            self.assertIn(icon, TOOL_ICONS)

    def test_icons_are_strings(self):
        """Test all icons are strings."""
        for key, value in TOOL_ICONS.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)


class TestToolRegistry(unittest.TestCase):
    """Test cases for TOOL_REGISTRY configuration."""

    def test_tool_registry_exists(self):
        """Test TOOL_REGISTRY dictionary exists."""
        self.assertIsInstance(TOOL_REGISTRY, dict)

    def test_registry_format(self):
        """Test registry entries have correct format."""
        for tool_name, (module_name, class_name, category, description) in TOOL_REGISTRY.items():
            self.assertIsInstance(tool_name, str)
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(class_name, str)
            self.assertIsInstance(category, str)
            self.assertIsInstance(description, str)

    def test_common_tools_registered(self):
        """Test common tools are registered."""
        expected_tools = ["arxiv", "duckduckgo", "wikipedia"]
        for tool in expected_tools:
            self.assertIn(tool, TOOL_REGISTRY)


class TestSpinnerRender(unittest.TestCase):
    """Test the braille spinner renders a live marker for every phase."""

    def test_thinking_phase(self):
        from agentica.cli.interactive import _render_spinner_text, _BRAILLE_SPINNER

        text = _render_spinner_text(0, "thinking", "", 2.0)
        self.assertIn(_BRAILLE_SPINNER[0], text)
        self.assertIn("thinking", text)
        self.assertIn("(2s)", text)

    def test_reasoning_phase(self):
        from agentica.cli.interactive import _render_spinner_text

        text = _render_spinner_text(3, "reasoning", "", 1.5)
        self.assertIn("reasoning", text)
        self.assertIn("(2s)", text)  # 1.5 -> :.0f rounds to 2

    def test_tool_phase_uses_base_label(self):
        from agentica.cli.interactive import _render_spinner_text

        text = _render_spinner_text(0, "tool", "🔧 grep", 5.0)
        self.assertIn("🔧 grep", text)
        self.assertIn("(5s)", text)
        self.assertNotIn("thinking", text)

    def test_answering_phase(self):
        from agentica.cli.interactive import _render_spinner_text

        text = _render_spinner_text(0, "answering", "", 3.0)
        self.assertIn("answering", text)
        self.assertIn("(3s)", text)

    def test_idle_phase_returns_empty(self):
        from agentica.cli.interactive import _render_spinner_text

        self.assertEqual(_render_spinner_text(0, "idle", "", 0.0), "")

    def test_frame_advances(self):
        from agentica.cli.interactive import _render_spinner_text, _BRAILLE_SPINNER

        t0 = _render_spinner_text(0, "thinking", "", 0.0)
        t1 = _render_spinner_text(1, "thinking", "", 0.0)
        self.assertIn(_BRAILLE_SPINNER[0], t0)
        self.assertIn(_BRAILLE_SPINNER[1], t1)
        self.assertNotEqual(t0, t1)


class TestCLIHelpers(unittest.TestCase):
    """Test cases for CLI helper functions."""

    def test_tool_icon_lookup(self):
        """Test looking up tool icons."""
        # Test existing icon
        icon = TOOL_ICONS.get("read_file", TOOL_ICONS["default"])
        self.assertIsNotNone(icon)

        # Test default fallback
        icon = TOOL_ICONS.get("nonexistent_tool", TOOL_ICONS["default"])
        self.assertEqual(icon, TOOL_ICONS["default"])

    def test_display_token_stats_shows_context_usage(self):
        from agentica.cli.display import display_token_stats

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)

        fake_console = MagicMock()
        display_token_stats(
            fake_console,
            tracker,
            context_window=128000,
            session_total_tokens=64000,
            tool_use_count=2,
            elapsed_seconds=5.32,
        )

        rendered = fake_console.print.call_args[0][0]
        self.assertIn("ctx 50.0%", rendered)
        self.assertIn("64K / 128K", rendered)
        self.assertIn("2 tools", rendered)
        self.assertIn("5.32s", rendered)

    def test_display_token_stats_singular_tool_use(self):
        from agentica.cli.display import display_token_stats

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=500, output_tokens=200)

        fake_console = MagicMock()
        display_token_stats(
            fake_console,
            tracker,
            context_window=128000,
            session_total_tokens=700,
            tool_use_count=1,
            elapsed_seconds=1.0,
        )

        rendered = fake_console.print.call_args[0][0]
        self.assertIn("1 tool", rendered)
        self.assertNotIn("1 tools", rendered)

    def test_display_token_stats_no_tools_no_tool_label(self):
        from agentica.cli.display import display_token_stats

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)

        fake_console = MagicMock()
        display_token_stats(
            fake_console,
            tracker,
            context_window=128000,
            session_total_tokens=150,
            tool_use_count=0,
            elapsed_seconds=0.5,
        )

        rendered = fake_console.print.call_args[0][0]
        self.assertNotIn("tool", rendered)

    def test_display_token_stats_fallback_without_session_tokens(self):
        """When session_total_tokens is 0, fall back to cost_tracker totals."""
        from agentica.cli.display import display_token_stats

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=2000, output_tokens=500)

        fake_console = MagicMock()
        display_token_stats(fake_console, tracker, context_window=128000)

        rendered = fake_console.print.call_args[0][0]
        self.assertIn("2.5K / 128K", rendered)

    def test_format_tokens_short(self):
        from agentica.cli.display import _format_tokens_short

        self.assertEqual(_format_tokens_short(500), "500")
        self.assertEqual(_format_tokens_short(1000), "1K")
        self.assertEqual(_format_tokens_short(1500), "1.5K")
        self.assertEqual(_format_tokens_short(64000), "64K")
        self.assertEqual(_format_tokens_short(128000), "128K")
        self.assertEqual(_format_tokens_short(1000000), "1M")
        self.assertEqual(_format_tokens_short(1500000), "1.5M")

    def test_context_pct_style(self):
        from agentica.cli.display import context_pct_style

        self.assertEqual(context_pct_style(30), "green")
        self.assertEqual(context_pct_style(50), "yellow")
        self.assertEqual(context_pct_style(80), "red")
        self.assertEqual(context_pct_style(95), "bold red")

    def test_build_context_bar(self):
        from agentica.cli.display import build_context_bar

        bar = build_context_bar(50.0, width=10)
        self.assertEqual(bar.count("█"), 5)
        self.assertEqual(bar.count("░"), 5)
        bar0 = build_context_bar(0, width=10)
        self.assertNotIn("█", bar0)
        bar100 = build_context_bar(100, width=10)
        self.assertNotIn("░", bar100)

    def test_build_status_bar_fragments_narrow(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            last_turn_seconds=12.3,
            terminal_width=40,
        )
        text = "".join(v for _, v in frags)
        self.assertIn("gpt-4o", text)
        self.assertIn("⏱ 12.3s", text)
        self.assertNotIn("64K", text)

    def test_build_status_bar_fragments_wide(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            cost_usd=0.05,
            active_seconds=105.0,
            last_turn_seconds=12.3,
            terminal_width=100,
        )
        text = "".join(v for _, v in frags)
        self.assertIn("64K/128K", text)
        self.assertIn("50%", text)
        self.assertIn("$0.05", text)
        self.assertIn("⏱ 12.3s", text)
        self.assertIn("Σ 1m45s", text)

    def test_build_status_bar_fragments_cost_in_medium(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            cost_usd=0.002,
            last_turn_seconds=5.0,
            terminal_width=60,
        )
        text = "".join(v for _, v in frags)
        self.assertIn("$0.0020", text)
        self.assertIn("50%", text)
        self.assertIn("⏱ 5.0s", text)

    def test_status_bar_agent_running_uses_active_classes(self):
        """When ``agent_running=True``, every ``class:sb*`` fragment must be
        rewritten to its ``-active`` variant. The CLI style sheet paints those
        with a darker ``bg:#0f0f1a`` background — a subtle visual downshift
        that tells the user "the agent is working right now" without hiding
        the (still updating) numeric fields.
        """
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            cost_usd=0.023,
            last_turn_seconds=3.4,
            spinner_text="⠋",
            terminal_width=120,
            agent_running=True,
        )
        classes = [cls for cls, _ in frags]
        # Every sb* class MUST end with -active — no idle class may leak
        for cls in classes:
            if cls.startswith("class:sb"):
                self.assertTrue(
                    cls.endswith("-active"),
                    f"idle status-bar class leaked while running: {cls!r}",
                )

    def test_status_bar_agent_running_prepends_spinner_leftmost(self):
        """The spinner glyph must be the leftmost fragment so it reads as a
        heartbeat at the far-left edge of the bar. Empty ``spinner_text``
        should NOT inject a fragment.
        """
        from agentica.cli.display import build_status_bar_fragments

        with_spinner = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            spinner_text="⠋",
            terminal_width=120,
            agent_running=True,
        )
        self.assertEqual(with_spinner[0][0], "class:sb-spin-active")
        self.assertIn("⠋", with_spinner[0][1])

        without_spinner = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            spinner_text="",
            terminal_width=120,
            agent_running=True,
        )
        # No leading spinner segment when text is empty
        self.assertNotEqual(without_spinner[0][0], "class:sb-spin-active")

    def test_status_bar_idle_keeps_base_classes(self):
        """When ``agent_running=False`` (default), fragments must keep their
        base ``class:sb`` / ``class:sb-dim`` / etc. names — no ``-active``
        suffix leaks into idle-state rendering.
        """
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            cost_usd=0.023,
            last_turn_seconds=3.4,
            terminal_width=120,
            agent_running=False,
        )
        for cls, _ in frags:
            self.assertFalse(
                cls.endswith("-active"),
                f"idle bar emitted active class: {cls!r}",
            )

    def test_build_status_bar_fragments_shows_profile_prefix(self):
        """After `/model profile <name>` the status bar must show the new
        profile prefix and provider/model label — driven entirely by the
        ``profile_name`` / ``model_provider`` / ``model_name`` args the
        interactive loop syncs via ``_apply_command_result``."""
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="deepseek-v4-flash",
            model_provider="deepseek",
            profile_name="work",
            context_tokens=1000,
            context_window=128000,
            last_turn_seconds=1.0,
            terminal_width=120,
        )
        text = "".join(v for _, v in frags)
        self.assertIn("profile:work", text)
        self.assertIn("deepseek/deepseek-v4-flash", text)

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

    def test_stream_display_manager_suppresses_micro_compact(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.handle_event({"type": "compact.micro", "agent_name": "Agent", "cleared": 3})
        fake.print.assert_not_called()

    def test_stream_display_manager_renders_markdown_when_setting_on(self):
        from agentica.cli.display import StreamDisplayManager
        from rich.markdown import Markdown

        fake = MagicMock()
        fake.width = 80
        with patch("agentica.cli.display.get_setting", return_value="on"):
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
        with patch("agentica.cli.display.get_setting", return_value="off"):
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
        with patch("agentica.cli.display.get_setting", return_value="on"):
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
        from agentica.cli.interactive import ChatConsole
        from agentica.cli.display import _GutteredConsole

        cc = ChatConsole()
        gutter_con = _GutteredConsole(cc, "▎", "cyan")
        # Should NOT raise
        gutter_con.print("hello from ChatConsole gutter")
        # Prefix cache should be a string (ANSI-rendered by ``render_ansi``)
        self.assertIsInstance(gutter_con.gutter_prefix_ansi, str)
        self.assertIn("▎", gutter_con.gutter_prefix_ansi)

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
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.handle_event(
            {
                "type": "compact.rule_based",
                "agent_name": "Agent",
                "before": 20,
                "after": 8,
                "elapsed": 0.25,
            }
        )
        calls = [str(c) for c in fake.print.call_args_list]
        self.assertTrue(any("compact" in c for c in calls))

    def test_display_tool_result_suppresses_write_todos_footer(self):
        """write_todos drops the result footer on success (call line lists tasks)."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("write_todos", '{"message":"ok"}', is_error=False, elapsed=0.002)
        fake.print.assert_not_called()

    def test_display_tool_defers_read_only_call_line(self):
        """Read-only tools skip the start-time call line (deferred to completion)."""
        from agentica.cli.display import StreamDisplayManager

        for name in ("read_file", "ls", "glob", "grep", "web_search", "fetch_url"):
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            dm.display_tool(name, {"file_path": "x.py"})
            fake.print.assert_not_called(), f"{name} call line must be deferred"

    def test_display_tool_result_merged_single_line_for_read_ops(self):
        """Read-only tools collapse call + result into one line with elapsed."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "grep",
            "path/a.py:1:match\npath/b.py:2:match\npath/c.py:3:match",
            is_error=False,
            elapsed=2.23,
            tool_args={"pattern": "foo", "path": "dual_mem"},
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        # one merged line: icon name params - count (elapsed)
        self.assertIn("grep", text)
        self.assertIn("'foo'", text)
        self.assertIn("3 lines", text)
        self.assertIn("(2.23s)", text)
        # matched content must not leak into the CLI
        self.assertNotIn("match", text)
        # no separate ⎿ footer — everything is on the merged line
        self.assertNotIn("⎿", text)

    def test_display_tool_result_merged_line_always_has_elapsed(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "read_file",
            "\n".join(f"line {i}" for i in range(311)),
            is_error=False,
            elapsed=0.027,
            tool_args={"file_path": "config.py", "offset": 130, "limit": 40},
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("config.py", text)
        self.assertIn("311 lines", text)
        self.assertIn("(27ms)", text)

    def test_display_tool_result_surfaces_errors_even_for_deferred_tools(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "read_file",
            "FileNotFoundError: nope",
            is_error=True,
            elapsed=0.01,
            tool_args={"file_path": "x.py"},
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("FileNotFoundError", text)

    def test_display_tool_defers_edit_tools_call_line(self):
        """edit_file/multi_edit_file defer the start-time call line (merged at completion)."""
        from agentica.cli.display import StreamDisplayManager

        for name in ("edit_file", "multi_edit_file"):
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            dm.display_tool(name, {"file_path": "/abs/path/to/config.py"})
            fake.print.assert_not_called(), f"{name} call line must be deferred"

    def test_display_edit_file_merged_shows_filename_and_diff(self):
        """edit_file: one summary line with filename (not abs path) + ✓ + a diff."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "edit_file",
            "Successfully applied 1 edit to config.py",
            is_error=False,
            elapsed=0.12,
            tool_args={
                "file_path": "/apdcephfs_cq11/share_2973545/flemingxu/config.py",
                "old_string": "DEBUG = False\n",
                "new_string": "DEBUG = True\n",
            },
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("edit_file", text)
        self.assertIn("config.py", text)
        # absolute path must NOT leak
        self.assertNotIn("/apdcephfs_cq11", text)
        self.assertIn("✓", text)
        self.assertIn("(120ms)", text)
        # diff content surfaces the change (rendered as a Syntax object)
        syntax_args = [c.args[0] for c in fake.print.call_args_list if c.args and "Syntax" in type(c.args[0]).__name__]
        self.assertTrue(syntax_args, "expected a diff Syntax block")
        self.assertIn("DEBUG", getattr(syntax_args[0], "code", ""))

    def test_display_multi_edit_file_shows_filename_and_edit_count(self):
        """multi_edit_file: filename + edit count, no absolute path."""
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "multi_edit_file",
            "Successfully applied 2 edits to config.py",
            is_error=False,
            elapsed=0.42,
            tool_args={
                "file_path": "/apdcephfs_cq11/share_2973545/flemingxu/config.py",
                "edits": [
                    {"old_string": "a=1\n", "new_string": "a=2\n"},
                    {"old_string": "b=3\n", "new_string": "b=4\n"},
                ],
            },
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("multi_edit_file", text)
        self.assertIn("config.py", text)
        self.assertNotIn("/apdcephfs_cq11", text)
        self.assertIn("2 edits", text)
        self.assertIn("(420ms)", text)

    def test_display_write_file_merged_shows_summary_and_diff(self):
        """write_file: one summary line (created/updated + line count) + a diff."""
        import tempfile
        from agentica.cli.display import StreamDisplayManager

        new_content = "\n".join(f"line {i}" for i in range(20)) + "\n"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "new_file.py")
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            # Call-start stashes pre-write content (file is new → empty old).
            dm.display_tool("write_file", {"file_path": path, "content": new_content})
            dm.display_tool_result(
                "write_file",
                f"Created file, absolute path: {path}",
                is_error=False,
                elapsed=0.12,
                tool_args={"file_path": path, "content": new_content},
            )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("write_file", text)
        self.assertIn("new_file.py", text)
        self.assertIn("created", text)
        self.assertIn("(120ms)", text)
        # A diff Syntax block is rendered; for a new file it's all additions.
        syntax_args = [c.args[0] for c in fake.print.call_args_list if c.args and "Syntax" in type(c.args[0]).__name__]
        self.assertTrue(syntax_args, "expected a diff Syntax block")
        self.assertIn("line 0", getattr(syntax_args[0], "code", ""))

    def test_display_write_file_diff_against_old_content(self):
        """write_file on an existing file diffs old→new (not all-additions)."""
        import tempfile
        from agentica.cli.display import StreamDisplayManager

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cfg.py")
            with open(path, "w") as f:
                f.write("DEBUG = False\nKEEP = 1\n")
            new_content = "DEBUG = True\nKEEP = 1\n"
            fake = MagicMock()
            fake.width = 80
            dm = StreamDisplayManager(fake)
            dm.display_tool("write_file", {"file_path": path, "content": new_content})
            dm.display_tool_result(
                "write_file",
                f"Updated file, absolute path: {path}",
                is_error=False,
                elapsed=0.10,
                tool_args={"file_path": path, "content": new_content},
            )
        syntax_args = [c.args[0] for c in fake.print.call_args_list if c.args and "Syntax" in type(c.args[0]).__name__]
        self.assertTrue(syntax_args)
        code = getattr(syntax_args[0], "code", "")
        # Real diff: -False / +True, and unchanged KEEP has no +/-.
        self.assertIn("-DEBUG = False", code)
        self.assertIn("+DEBUG = True", code)
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertIn("updated", text)

    def test_display_execute_head_tail_window(self):
        """execute shows up to 20 lines inline; beyond that head 10 + tail 10 with the middle hidden."""
        from agentica.cli.display import StreamDisplayManager

        # 10 lines: <= 20, fully shown, no fold hint.
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "execute",
            "\n".join(f"line {i}" for i in range(10)),
            is_error=False,
            elapsed=0.1,
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertNotIn("hidden", text)
        self.assertNotIn("...", text)

        # 30 lines: > 20, head 10 + tail 10 shown, 10 hidden in the middle.
        fake2 = MagicMock()
        fake2.width = 80
        dm2 = StreamDisplayManager(fake2)
        dm2.display_tool_result(
            "execute",
            "\n".join(f"line {i}" for i in range(30)),
            is_error=False,
            elapsed=0.1,
        )
        text2 = "\n".join(str(c) for c in fake2.print.call_args_list)
        # Head and tail present; middle lines hidden.
        self.assertIn("line 0", text2)
        self.assertIn("line 29", text2)
        self.assertNotIn("line 15", text2)

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


class TestCLIModelParams(unittest.TestCase):
    """get_model() and resolution should honour the extended tuning params."""

    def test_get_model_passes_top_p_and_context_window(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "deepseek",
            "deepseek-v4-flash",
            api_key="fake_key",
            max_tokens=4096,
            temperature=0.3,
            reasoning_effort="high",
            top_p=0.9,
            context_window=500000,
        )
        self.assertEqual(model.top_p, 0.9)
        self.assertEqual(model.context_window, 500000)
        self.assertEqual(model.max_tokens, 4096)
        self.assertEqual(model.reasoning_effort, "high")

    def test_get_model_context_window_overrides_catalog(self):
        from agentica.cli.runtime import get_model

        # Without an explicit value the catalog fills it (deepseek -> 1_000_000);
        # an explicit value must win.
        default_model = get_model("deepseek", "deepseek-v4-flash", api_key="k")
        override_model = get_model(
            "deepseek",
            "deepseek-v4-flash",
            api_key="k",
            context_window=42000,
        )
        self.assertNotEqual(default_model.context_window, 42000)
        self.assertEqual(override_model.context_window, 42000)

    def test_anthropic_accepts_top_p_and_reasoning_effort(self):
        # Anthropic now takes reasoning_effort too: the Claude model maps it to
        # adaptive thinking (thinking.type=adaptive + output_config.effort).
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-4-8",
            api_key="k",
            top_p=0.8,
            context_window=300000,
            reasoning_effort="high",
            base_url="https://ignored",
        )
        self.assertEqual(model.top_p, 0.8)
        self.assertEqual(model.context_window, 300000)
        self.assertEqual(model.reasoning_effort, "high")
        # And it must actually enable adaptive thinking in the request kwargs.
        kwargs = model.request_kwargs
        self.assertEqual(kwargs.get("thinking"), {"type": "adaptive"})
        self.assertEqual(kwargs.get("extra_body", {}).get("output_config"), {"effort": "high"})
        # Adaptive thinking requires temperature=1; it must be forced.
        self.assertEqual(kwargs.get("temperature"), 1)

    def test_get_model_passes_extra_body_and_extra_headers(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "openai",
            "hy3",
            api_key="k",
            base_url="http://api.taiji.woa.com/openapi/v2",
            extra_body={"chat_template_kwargs": {"reasoning_effort": "high"}},
            extra_headers={"X-Custom": "value"},
        )
        self.assertEqual(model.extra_body, {"chat_template_kwargs": {"reasoning_effort": "high"}})
        self.assertEqual(model.extra_headers, {"X-Custom": "value"})

    def test_get_model_skips_extra_body_for_anthropic(self):
        from agentica.cli.runtime import get_model

        model = get_model(
            "anthropic",
            "claude-opus-4-8",
            api_key="k",
            extra_body={"some": "thing"},
        )
        self.assertFalse(hasattr(model, "extra_body") and model.extra_body)

    def test_reasoning_effort_accepts_low_medium(self):
        import sys
        from agentica.cli.runtime import parse_args

        with patch.object(sys, "argv", ["agentica", "--reasoning_effort", "low"]):
            args = parse_args()
        self.assertEqual(args.reasoning_effort, "low")

    def test_resolve_model_config_carries_profile_tuning_params(self):
        import argparse
        from agentica.cli.setup import resolve_model_config

        profile = {
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com",
            "api_key": "sk-x",
            "reasoning_effort": "high",
            "max_tokens": 4096,
            "context_window": 500000,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        args = argparse.Namespace(
            model_provider=None,
            model_name=None,
            base_url=None,
            api_key=None,
            auxiliary_model_provider=None,
            auxiliary_model_name=None,
            auxiliary_base_url=None,
            auxiliary_api_key=None,
        )
        with patch("agentica.cli.setup.get_profile", return_value=profile):
            resolved = resolve_model_config(args, console=None)

        self.assertEqual(resolved["reasoning_effort"], "high")
        self.assertEqual(resolved["max_tokens"], 4096)
        self.assertEqual(resolved["context_window"], 500000)
        self.assertEqual(resolved["temperature"], 0.3)
        self.assertEqual(resolved["top_p"], 0.9)


class TestCLIImports(unittest.TestCase):
    """Test cases for CLI module imports."""

    def test_can_import_cli_module(self):
        """Test CLI module can be imported."""
        try:
            import agentica.cli

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import cli module: {e}")

    def test_can_import_agent(self):
        """Test Agent can be imported from CLI."""
        try:
            from agentica import Agent

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import Agent: {e}")


class TestCLIConfiguration(unittest.TestCase):
    """Test cases for CLI configuration."""

    def test_history_file_path(self):
        """Test history file path is set."""
        from agentica.cli import history_file

        self.assertIsInstance(history_file, str)
        self.assertTrue(history_file.endswith("cli_history.txt"))

    def test_parse_args_defaults_to_none_for_resolution(self):
        """parse_args leaves provider/model as None so saved config can apply.

        Final defaults (deepseek/deepseek-v4-flash) are filled in by
        resolve_model_config (args > config.yaml profile > hardcoded).
        """
        from agentica.cli.runtime import parse_args

        with patch.object(sys, "argv", ["agentica"]):
            args = parse_args()

        self.assertIsNone(args.model_provider)
        self.assertIsNone(args.model_name)
        self.assertIsNone(args.reasoning_effort)
        self.assertTrue(args.enable_diagnostics)
        self.assertIsNone(args.diagnostics_servers)

    def test_parse_diagnostics_flags(self):
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "agentica",
                "--enable-diagnostics",
                "--diagnostics-server",
                "pyright",
                "--diagnostics-server",
                "typescript-language-server",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.enable_diagnostics)
        self.assertEqual(args.diagnostics_servers, ["pyright", "typescript-language-server"])

    def test_parse_doctor_diagnostics_flags(self):
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            ["agentica", "doctor", "--enable-diagnostics", "--diagnostics-server", "pyright", "--work_dir", "."],
        ):
            args = parse_args()

        self.assertEqual(args.command, "doctor")
        self.assertTrue(args.enable_diagnostics)
        self.assertEqual(args.diagnostics_servers, ["pyright"])
        self.assertEqual(args.work_dir, ".")

    def test_resolve_model_config_defaults_to_deepseek_v4_flash(self):
        """With no flags/saved config, resolution falls back to DeepSeek v4 flash."""
        import argparse
        from agentica.cli.setup import resolve_model_config

        args = argparse.Namespace(
            model_provider=None,
            model_name=None,
            base_url=None,
            api_key=None,
            auxiliary_model_provider=None,
            auxiliary_model_name=None,
            auxiliary_base_url=None,
            auxiliary_api_key=None,
        )
        with patch("agentica.cli.setup.get_profile", return_value={}):
            resolved = resolve_model_config(args, console=None)

        self.assertEqual(resolved["model_provider"], "deepseek")
        self.assertEqual(resolved["model_name"], "deepseek-v4-flash")

    def test_get_model_defaults_deepseek_cli_reasoning_effort_to_max(self):
        """CLI DeepSeek usage should default to max effort for agentic tasks."""
        from agentica.cli.runtime import get_model

        model = get_model("deepseek", "deepseek-v4-flash", api_key="fake_key")

        self.assertEqual(model.reasoning_effort, "max")

    def test_get_model_respects_explicit_deepseek_reasoning_effort(self):
        """Explicit CLI reasoning effort should override the agentic default."""
        from agentica.cli.runtime import get_model

        model = get_model(
            "deepseek",
            "deepseek-v4-flash",
            api_key="fake_key",
            reasoning_effort="high",
        )

        self.assertEqual(model.reasoning_effort, "high")

    def test_create_agent_uses_deepseek_cli_reasoning_default(self):
        """DeepAgent creation should inherit the CLI's max-thinking default."""
        from agentica.cli.runtime import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with patch("agentica.agent.deep.DeepAgent", FakeDeepAgent):
            create_agent(
                {
                    "model_provider": "deepseek",
                    "model_name": "deepseek-v4-flash",
                    "debug": False,
                    "work_dir": None,
                },
                extra_tools=[],
                workspace=None,
                skills_registry=None,
            )

        self.assertEqual(captured["model"].reasoning_effort, "max")

    def test_create_agent_passes_diagnostics_controls_to_deep_agent(self):
        from agentica.cli.runtime import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch(
                "agentica.agent.deep.DeepAgent",
                FakeDeepAgent,
            ),
        ):
            create_agent(
                {
                    "model_provider": "deepseek",
                    "model_name": "deepseek-v4-flash",
                    "debug": False,
                    "work_dir": None,
                    "enable_diagnostics": True,
                    "diagnostics_servers": ["pyright"],
                },
                extra_tools=[],
                workspace=None,
                skills_registry=None,
            )

        self.assertTrue(captured["enable_diagnostics"])
        self.assertEqual(captured["diagnostics_servers"], ["pyright"])

    def test_model_command_freeform_is_rejected_and_does_not_mutate_config(self):
        """`/model openai/gpt-5` must NOT silently overwrite the active profile.

        This is the regression test for the original "config.yaml 乱掉" bug:
        the legacy free-form path called ``_persist_model_choice`` which
        clobbered whatever main/aux/tuning the active profile had stored.
        """
        from agentica import global_config as gc

        ctx = cli_commands.CommandContext(
            agent_config={
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key": "sk-original",
                "debug": False,
                "work_dir": None,
            },
            current_agent=None,
            extra_tools=[],
            workspace=None,
            skills_registry=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_commands, "get_console", return_value=MagicMock()),
                patch.object(cli_commands, "get_model", return_value=MagicMock()),
                patch.object(cli_commands, "create_agent", return_value=MagicMock()),
            ):
                gc.upsert_profile(
                    "default",
                    {
                        "model_provider": "deepseek",
                        "model_name": "deepseek-v4-flash",
                        "base_url": "https://api.deepseek.com",
                        "api_key": "sk-original",
                    },
                )
                cli_commands._cmd_model(ctx, "openai/gpt-5")
                saved = gc.get_profile("default")

        # Config.yaml is byte-for-byte unchanged.
        self.assertEqual(saved["model_provider"], "deepseek")
        self.assertEqual(saved["model_name"], "deepseek-v4-flash")
        self.assertEqual(saved["base_url"], "https://api.deepseek.com")
        # Live session config is also untouched (no partial mutation).
        self.assertEqual(ctx.agent_config["model_provider"], "deepseek")
        self.assertEqual(ctx.agent_config["model_name"], "deepseek-v4-flash")

    def test_model_command_switch_to_saved_profile_does_not_mutate_config(self):
        """`/model <profile_name>` is session-only; config.yaml is not rewritten.

        It MAY update the ``active_profile`` pointer (that's a pointer flip,
        not a profile body change), but each profile's main/aux/tuning fields
        must survive intact.
        """
        from agentica import global_config as gc

        ctx = cli_commands.CommandContext(
            agent_config={
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key": "sk-ds",
                "debug": False,
                "work_dir": None,
            },
            current_agent=None,
            extra_tools=[],
            workspace=None,
            skills_registry=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_commands, "get_console", return_value=MagicMock()),
                patch.object(cli_commands, "get_model", return_value=MagicMock()) as mock_get_model,
                patch.object(cli_commands, "create_agent", return_value=MagicMock()),
                # /model <profile> -> _apply_profile -> set_project_profile(work_dir, name);
                # work_dir=None falls back to os.getcwd(), which would otherwise leak a
                # real ~/.agentica/projects/<repo>/profile file on every test run.
                patch.object(cli_commands, "set_project_profile"),
            ):
                gc.upsert_profile(
                    "deepseek",
                    {
                        "model_provider": "deepseek",
                        "model_name": "deepseek-v4-flash",
                        "base_url": "https://api.deepseek.com",
                        "api_key": "sk-ds",
                    },
                    make_active=True,
                )
                gc.upsert_profile(
                    "opus",
                    {
                        "model_provider": "anthropic",
                        "model_name": "claude-opus-4",
                        "base_url": "https://api.anthropic.com",
                        "api_key": "sk-anthropic",
                    },
                )
                cli_commands._cmd_model(ctx, "opus")
                ds_after = gc.get_profile("deepseek")
                opus_after = gc.get_profile("opus")

        # Each profile body survives intact.
        self.assertEqual(ds_after["model_name"], "deepseek-v4-flash")
        self.assertEqual(opus_after["model_name"], "claude-opus-4")
        # Live session swapped to the opus profile.
        self.assertEqual(ctx.agent_config["model_provider"], "anthropic")
        self.assertEqual(ctx.agent_config["model_name"], "claude-opus-4")
        self.assertEqual(mock_get_model.call_args.kwargs["model_provider"], "anthropic")

    def test_model_command_switch_updates_status_bar_profile_resolution(self):
        """Regression: after `/model <profile>`, the status bar must see the switch.

        `_apply_profile` persists the project override keyed by
        ``agent_config.get("work_dir") or os.getcwd()``. The status-bar sync in
        interactive.py's ``_apply_command_result`` must resolve with the exact
        same fallback — resolving with a bare (possibly-None) ``work_dir`` looks
        up the wrong key and silently falls back to the stale global default.
        """
        from agentica import global_config as gc

        ctx = cli_commands.CommandContext(
            agent_config={
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-a",
                "debug": False,
                "work_dir": None,
            },
            current_agent=None,
            extra_tools=[],
            workspace=None,
            skills_registry=None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            home = os.path.join(tmp, "agentica_home")
            os.makedirs(home, exist_ok=True)
            cfg_path = os.path.join(home, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.dict(os.environ, {"AGENTICA_HOME": home}),
                patch.object(cli_commands, "get_console", return_value=MagicMock()),
                patch.object(cli_commands, "get_model", return_value=MagicMock()),
                patch.object(cli_commands, "create_agent", return_value=MagicMock()),
            ):
                gc.upsert_profile(
                    "venus",
                    {
                        "model_provider": "openai",
                        "model_name": "gpt-4o",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-a",
                    },
                    make_active=True,
                )
                gc.upsert_profile(
                    "ark",
                    {
                        "model_provider": "openai",
                        "model_name": "gpt-5",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-b",
                    },
                    make_active=False,
                )

                cli_commands._cmd_model(ctx, "ark")

                # Mirrors interactive.py's _apply_command_result exactly.
                fixed_name, fixed_source = gc.resolve_active_profile_name(
                    work_dir=ctx.agent_config.get("work_dir") or os.getcwd()
                )
                # Sanity: the pre-fix call (no os.getcwd() fallback) misses the
                # override and silently shows the stale global default instead.
                stale_name, stale_source = gc.resolve_active_profile_name(work_dir=ctx.agent_config.get("work_dir"))

        self.assertEqual((fixed_name, fixed_source), ("ark", "project"))
        self.assertEqual((stale_name, stale_source), ("venus", "global"))

    def test_parse_goal_budget_flags(self):
        from agentica.cli.commands import _parse_goal_set_args

        objective, budgets, err = _parse_goal_set_args("--turns 5 --tokens 80000 --wall 1800 修复 API")

        self.assertIsNone(err)
        self.assertEqual(objective, "修复 API")
        self.assertEqual(budgets["turn_budget"], 5)
        self.assertEqual(budgets["token_budget"], 80000)
        self.assertEqual(budgets["wall_clock_budget_sec"], 1800)

    def test_parse_extensions_remove_command(self):
        """CLI supports `agentica extensions remove <skill-name>`."""
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            ["agentica", "extensions", "remove", "learn-from-experience"],
        ):
            args = parse_args()

        self.assertEqual(args.command, "skills")
        self.assertEqual(args.skills_command, "remove")
        self.assertEqual(args.skill_name, "learn-from-experience")

    def test_parse_extensions_install_command(self):
        """CLI parses local install sources without network access."""
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            ["agentica", "extensions", "install", "/tmp/mock-skill-repo"],
        ):
            args = parse_args()

        self.assertEqual(args.command, "skills")
        self.assertEqual(args.skills_command, "install")
        self.assertEqual(args.source, "/tmp/mock-skill-repo")

    def test_parse_experience_flags(self):
        """CLI exposes explicit DeepAgent self-evolution controls."""
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "agentica",
                "--no-experience",
                "--sync-experience-to-global-agent-md",
                "--enable-skill-upgrade",
                "--skill-upgrade-mode",
                "draft",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.no_experience)
        self.assertTrue(args.sync_experience_to_global_agent_md)
        self.assertTrue(args.enable_skill_upgrade)
        self.assertEqual(args.skill_upgrade_mode, "draft")

    def test_parse_memory_sync_flag(self):
        """CLI exposes explicit DeepAgent memory global-sync control."""
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "agentica",
                "--sync-memories-to-global-agent-md",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.sync_memories_to_global_agent_md)

    def test_interactive_extensions_install_reports_replaced_symlinked_skill(self):
        """Interactive install prints when it replaces a symlinked skill."""
        import agentica.cli.commands as commands
        from agentica.cli.commands import CommandContext
        from agentica.skills.skill import Skill
        from agentica.skills.skill_registry import SkillRegistry

        refreshed_registry = SkillRegistry()
        refreshed_registry.register(
            Skill(
                name="learn-from-experience",
                description="Learn from feedback",
                path=MagicMock(),
                location="user",
            )
        )
        installed_skill = Skill(
            name="learn-from-experience",
            description="Learn from feedback",
            path=MagicMock(),
            location="user",
        )

        def fake_install_skills(source, destination_dir=None, force=False, replaced_symlinked_skills=None):
            self.assertTrue(force)
            self.assertEqual(source, "/tmp/mock-skill-repo")
            replaced_symlinked_skills.append("learn-from-experience")
            return [installed_skill]

        ctx = CommandContext(
            agent_config={"model_provider": "zhipuai", "model_name": "glm-5", "debug": False, "work_dir": None},
            current_agent=MagicMock(),
            extra_tools=[],
            workspace=None,
            skills_registry=SkillRegistry(),
        )

        printed = []

        def mock_print(*args, **kwargs):
            if args:
                printed.append(str(args[0]))

        with (
            patch.object(commands, "install_skills", side_effect=fake_install_skills),
            patch.object(commands, "reset_skill_registry"),
            patch.object(commands, "load_skills"),
            patch.object(commands, "get_skill_registry", return_value=refreshed_registry),
            patch.object(commands, "create_agent", return_value=MagicMock()),
            patch("agentica.cli.commands.Path") as MockPath,
            patch("agentica.cli.commands.get_console") as mock_get_console,
        ):
            # Make Path(source).expanduser().exists() return True so the local
            # install branch is taken instead of falling through to hub_install.
            mock_path_inst = MagicMock()
            mock_path_inst.expanduser.return_value.exists.return_value = True
            MockPath.return_value = mock_path_inst
            mock_console = MagicMock()
            mock_console.print = mock_print
            mock_get_console.return_value = mock_console
            commands._cmd_skills(ctx, cmd_args="install /tmp/mock-skill-repo --force")

        self.assertTrue(
            any("replaced existing" in msg.lower() for msg in printed),
            f"Expected 'replaced existing' in output, got: {printed}",
        )

    def test_create_agent_moves_skills_summary_out_of_instructions(self):
        """CLI should not stuff skill summaries into static instructions."""
        from agentica.cli.runtime import create_agent
        from agentica.skills.skill import Skill
        from agentica.skills.skill_registry import SkillRegistry

        registry = SkillRegistry()
        registry.register(
            Skill(
                name="learn-from-experience",
                description="Learn from feedback",
                path=MagicMock(),
                location="user",
            )
        )

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                self.instructions = kwargs.get("instructions")
                self.tools = []
                self.session_guidance = []

            def add_session_guidance(self, text):
                self.session_guidance.append(text)

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch(
                "agentica.agent.deep.DeepAgent",
                FakeDeepAgent,
            ),
        ):
            agent = create_agent(
                {
                    "model_provider": "zhipuai",
                    "model_name": "glm-5",
                    "debug": False,
                    "work_dir": None,
                },
                extra_tools=[],
                workspace=None,
                skills_registry=registry,
            )

        self.assertIsNone(agent.instructions)
        self.assertEqual(len(agent.session_guidance), 1)
        self.assertIn("Available Skills", agent.session_guidance[0])

    def test_create_agent_passes_experience_controls_to_deep_agent(self):
        """CLI flags should map to DeepAgent experience settings deterministically."""
        from agentica.cli.runtime import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch(
                "agentica.agent.deep.DeepAgent",
                FakeDeepAgent,
            ),
        ):
            create_agent(
                {
                    "model_provider": "zhipuai",
                    "model_name": "glm-5",
                    "debug": False,
                    "work_dir": None,
                    "enable_experience_capture": False,
                    "sync_experience_to_global_agent_md": True,
                    "enable_skill_upgrade": True,
                    "skill_upgrade_mode": "draft",
                },
                extra_tools=[],
                workspace=None,
                skills_registry=None,
            )

        self.assertFalse(captured["enable_experience_capture"])
        self.assertTrue(captured["experience_config"].capture_tool_errors)
        self.assertTrue(captured["experience_config"].capture_user_corrections)
        self.assertFalse(captured["experience_config"].capture_success_patterns)
        self.assertTrue(captured["experience_config"].sync_to_global_agent_md)
        self.assertIsNotNone(captured["experience_config"].skill_upgrade)
        self.assertEqual(captured["experience_config"].skill_upgrade.mode, "draft")

    def test_create_agent_passes_memory_sync_control_to_deep_agent(self):
        """CLI memory sync flag should map to DeepAgent long-term memory config."""
        from agentica.cli.runtime import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch(
                "agentica.agent.deep.DeepAgent",
                FakeDeepAgent,
            ),
        ):
            create_agent(
                {
                    "model_provider": "zhipuai",
                    "model_name": "glm-5",
                    "debug": False,
                    "work_dir": None,
                    "sync_memories_to_global_agent_md": True,
                },
                extra_tools=[],
                workspace=None,
                skills_registry=None,
            )

        self.assertTrue(captured["long_term_memory_config"].sync_memories_to_global_agent_md)


class TestToolRegistryIntegrity(unittest.TestCase):
    """Test cases for tool registry integrity."""

    def test_all_tools_have_valid_module_names(self):
        """Test all tools have valid module names."""
        for tool_name, (module_name, class_name, category, description) in TOOL_REGISTRY.items():
            # Module name should not be empty
            self.assertTrue(len(module_name) > 0, f"Empty module name for {tool_name}")
            # Class name should not be empty
            self.assertTrue(len(class_name) > 0, f"Empty class name for {tool_name}")
            # Class name should be PascalCase (start with uppercase)
            self.assertTrue(class_name[0].isupper(), f"Class name {class_name} should start with uppercase")

    def test_no_duplicate_tools(self):
        """Test no duplicate tool names in registry."""
        tool_names = list(TOOL_REGISTRY.keys())
        self.assertEqual(len(tool_names), len(set(tool_names)))


class TestPendingQueueTimestamps(unittest.TestCase):
    """``PendingQueue`` must expose per-item submission timestamps so the TUI
    queue bar can label each pending message with when it was submitted.
    Timestamps must stay aligned with items across get / remove_index / clear.
    """

    def test_put_records_timestamp_for_each_item(self):
        import time as _t
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.commands import PendingQueue

        q = PendingQueue()
        q.put("a")
        q.put("b")
        q.put("c")

        self.assertEqual(q.get(timeout=0.1), "a")
        remaining = q.peek_all_with_timestamps()
        self.assertEqual([p[0] for p in remaining], ["b", "c"])
        self.assertEqual(len(remaining), 2)

    def test_remove_index_keeps_timestamps_aligned(self):
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.commands import PendingQueue

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
        from agentica.cli.interactive import queue_item_preview

        self.assertEqual(
            queue_item_preview("/requesting-code-review git status的代码"),
            "/requesting-code-review git status的代码",
        )
        self.assertEqual(queue_item_preview("/status"), "/status")
        self.assertEqual(queue_item_preview("normal follow-up"), "normal follow-up")

    def test_shell_mode_items_are_marked(self):
        from agentica.cli.interactive import queue_item_preview

        self.assertEqual(queue_item_preview("pwd", shell_mode=True), "$ pwd")
        self.assertEqual(queue_item_preview("pwd", shell_mode=False), "pwd")

    def test_shell_mode_exempt_commands_are_not_marked(self):
        from agentica.cli.interactive import queue_item_preview

        self.assertEqual(queue_item_preview("/model", shell_mode=True), "/model")
        self.assertEqual(
            queue_item_preview("/requesting-code-review x", shell_mode=True),
            "$ /requesting-code-review x",
        )

    def test_image_payload_previews_its_text(self):
        from agentica.cli.interactive import queue_item_preview

        self.assertEqual(queue_item_preview(("describe this", ["/tmp/a.png"])), "describe this")

    def test_btw_tuple_preview(self):
        from agentica.cli.interactive import queue_item_preview

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
        from agentica.cli import commands as cli_commands

        pq = cli_commands.PendingQueue()
        ctx = cli_commands.CommandContext(
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
        from agentica.cli.commands import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("first")
        pq.put("second")
        pq.put("third")

        _cmd_queue(ctx, "edit 2 SECOND v2")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["first", "SECOND v2", "third"])

    def test_edit_rejects_missing_text(self):
        from agentica.cli.commands import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("first")
        # No new text → must NOT mutate the queue.
        _cmd_queue(ctx, "edit 1")
        _cmd_queue(ctx, "edit 1   ")  # whitespace-only also rejected
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["first"])

    def test_edit_rejects_bad_index(self):
        from agentica.cli.commands import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("only")
        _cmd_queue(ctx, "edit 99 nope")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["only"])

    def test_insert_at_front(self):
        from agentica.cli.commands import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("a")
        pq.put("b")
        _cmd_queue(ctx, "insert 1 head")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["head", "a", "b"])

    def test_insert_at_back_equivalent_to_append(self):
        """``/queue insert <qsize+1> text`` is documented as 'back' and must
        be accepted, mapping to the same slot as a plain ``/queue text``."""
        from agentica.cli.commands import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("a")
        pq.put("b")
        # qsize+1 (1-based) → idx == len (0-based) → append
        _cmd_queue(ctx, f"insert {pq.qsize() + 1} tail")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["a", "b", "tail"])

    def test_insert_rejects_bad_index(self):
        from agentica.cli.commands import _cmd_queue

        ctx, pq = self._ctx_with_queue()
        pq.put("a")
        _cmd_queue(ctx, "insert 99 nope")
        self.assertEqual([p[0] for p in pq.peek_all_with_timestamps()], ["a"])


class TestRenameCommand(unittest.TestCase):
    """`/rename <name>` persists a recognizable label for `/resume`."""

    def _ctx_with_agent(self, tmp_dir, session_id="sess-cli"):
        session_log = SessionLog(session_id, base_dir=str(tmp_dir))
        session_log.append("user", "first turn")

        agent = MagicMock()
        agent.session_id = session_id
        agent._session_log = session_log

        context = cli_commands.CommandContext(
            agent_config={"model_provider": "zhipuai", "model_name": "glm-5"},
            current_agent=agent,
            extra_tools=[],
            workspace=None,
        )
        return context, session_log

    def test_rename_current_session_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)

            cli_commands._cmd_rename(context, "My favourite session")

            self.assertEqual(session_log.get_name(), "My favourite session")

    def test_rename_strips_name(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)

            cli_commands._cmd_rename(context, "  Release investigation  ")

            self.assertEqual(session_log.get_name(), "Release investigation")

    def test_rename_rejects_empty_name(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)

            cli_commands._cmd_rename(context, "   ")

            self.assertIsNone(session_log.get_name())

    def test_rename_requires_active_session(self):
        context = cli_commands.CommandContext(agent_config={}, current_agent=None)
        console = MagicMock()
        with patch("agentica.cli.commands.get_console", return_value=console):
            cli_commands._cmd_rename(context, "Orphan")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
        self.assertIn("No active session", printed)

    def test_rename_reports_metadata_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)
            console = MagicMock()
            with (
                patch.object(session_log, "set_name", side_effect=OSError("disk full")),
                patch("agentica.cli.commands.get_console", return_value=console),
            ):
                cli_commands._cmd_rename(context, "Important session")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
        self.assertIn("Failed to rename session: disk full", printed)

    def test_rename_replaces_session_command(self):
        self.assertIs(
            cli_commands.COMMAND_REGISTRY["/rename"][0],
            cli_commands._cmd_rename,
        )
        self.assertNotIn("/session", cli_commands.COMMAND_REGISTRY)


class TestResumeArchivedFilter(unittest.TestCase):
    """``/resume`` must respect the ``archived`` sidecar flag: the picker
    (bare ``/resume`` or ``/resume <number>``) hides archived sessions, but
    an explicit id/prefix still resumes them directly — same cross-surface
    semantics the Web UI sidebar already enforces.
    """

    def _sessions(self):
        return [
            {
                "session_id": "sess-active-1111",
                "path": "/tmp/sess-active-1111.jsonl",
                "size_bytes": 100,
                "last_timestamp": "2026-01-01T00:00:00",
                "name": "Release investigation",
                "archived": False,
            },
            {
                "session_id": "sess-archived-2222",
                "path": "/tmp/sess-archived-2222.jsonl",
                "size_bytes": 100,
                "last_timestamp": "2026-01-02T00:00:00",
                "name": "Archived work",
                "archived": True,
            },
            {
                "session_id": "sess-active-3333",
                "path": "/tmp/sess-active-3333.jsonl",
                "size_bytes": 100,
                "last_timestamp": "2026-01-03T00:00:00",
                "name": None,
                "archived": False,
            },
        ]

    def _resume(self, target, sessions=None):
        context = cli_commands.CommandContext(
            agent_config={"model_provider": "zhipuai", "model_name": "glm-5"},
            current_agent=None,
        )
        with (
            patch(
                "agentica.memory.session_log.SessionLog.list_sessions",
                return_value=sessions or self._sessions(),
            ),
            patch("agentica.memory.session_log.SessionLog.list_user_messages", return_value=[]),
            patch("agentica.memory.session_log.SessionLog.exists", return_value=False),
            patch("agentica.cli.commands.create_agent") as create_agent,
            patch("agentica.cli.commands.GoalManager") as goal_manager,
        ):
            agent = MagicMock()
            agent._session_log = None
            create_agent.return_value = agent
            goal_manager.return_value.load.return_value = None
            result = cli_commands._cmd_resume(context, target)
        return create_agent, result

    def test_picker_listing_excludes_archived(self):
        ctx = cli_commands.CommandContext(agent_config={}, current_agent=None)
        with (
            patch("agentica.memory.session_log.SessionLog.list_sessions", return_value=self._sessions()),
            patch(
                "agentica.memory.session_log.SessionLog.session_preview",
                return_value={"user_count": 0, "first_user": None},
            ),
        ):
            with patch("agentica.cli.commands.get_console") as mock_console:
                console = MagicMock()
                mock_console.return_value = console
                cli_commands._cmd_resume(ctx, "")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        self.assertGreaterEqual(printed.count("sess-act"), 2)  # both active sessions listed
        self.assertIn("Release investigation", printed)
        self.assertIn("/resume <number|name|id-prefix>", printed)
        self.assertNotIn("sess-arc", printed)

    def test_resume_by_name(self):
        create_agent, result = self._resume("release investigation")

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-active-1111")
        self.assertIsNotNone(result)

    def test_resume_name_may_contain_at(self):
        sessions = self._sessions()
        sessions[0]["name"] = "Looking at performance issues"

        create_agent, result = self._resume("Looking at performance issues", sessions)

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-active-1111")
        self.assertIsNone(create_agent.call_args[0][0]["_resume_at_uuid"])
        self.assertIsNotNone(result)

    def test_resume_at_parses_valid_uuid_suffix(self):
        message_uuid = "12345678-1234-1234-1234-123456789abc"

        create_agent, result = self._resume(f"sess-active-1111 at {message_uuid}")

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-active-1111")
        self.assertEqual(create_agent.call_args[0][0]["_resume_at_uuid"], message_uuid)
        self.assertIsNotNone(result)

    def test_resume_numeric_name_when_not_a_valid_index(self):
        sessions = self._sessions()
        sessions[0]["name"] = "99"

        create_agent, result = self._resume("99", sessions)

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-active-1111")
        self.assertIsNotNone(result)

    def test_archived_duplicate_name_does_not_make_visible_name_ambiguous(self):
        sessions = self._sessions()
        sessions[1]["name"] = "Release investigation"

        create_agent, result = self._resume("Release investigation", sessions)

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-active-1111")
        self.assertIsNotNone(result)

    def test_picker_by_number_skips_archived_indices(self):
        """Numeric selection indexes into the visible session list."""
        create_agent, result = self._resume("2")

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-active-3333")
        self.assertIsNotNone(result)

    def test_explicit_id_prefix_still_resumes_archived(self):
        create_agent, result = self._resume("sess-archived")

        self.assertEqual(create_agent.call_args[0][0]["session_id"], "sess-archived-2222")
        self.assertIsNotNone(result)


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
        self.assertIn("+$0.08", out, "delta cost shown with 2-decimal $ prefix")

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


class TestLessLesskeyDetection(unittest.TestCase):
    """Ctrl+O expand pager: detect --lesskey-content support (less's help
    misprints the option as --lesskey-context, so detection must probe the real
    option, not parse help text)."""

    def setUp(self):
        # Reset the module-level cache between tests.
        from agentica.cli import interactive as it

        it._LESS_LESSKEY_OK = None

    def test_supported_when_no_error(self):
        from agentica.cli import interactive as it

        fake = MagicMock(returncode=0, stderr="")
        with patch("agentica.cli.interactive.subprocess.run", return_value=fake):
            self.assertTrue(it._less_supports_lesskey("/usr/bin/less"))

    def test_unsupported_when_stderr_mentions_option(self):
        from agentica.cli import interactive as it

        fake = MagicMock(returncode=0, stderr="There is no lesskey-content=... option")
        with patch("agentica.cli.interactive.subprocess.run", return_value=fake):
            self.assertFalse(it._less_supports_lesskey("/usr/bin/less"))


class TestCompileLesskey(unittest.TestCase):
    """Old-less fallback: compile a lesskey file to bind Ctrl+O to quit when
    --lesskey-content is unavailable. Esc is not bound (escape-sequence
    prefix would break arrow keys)."""

    def test_returns_path_when_lesskey_compiles(self):
        from agentica.cli import interactive as it

        run_calls = []

        def fake_run(cmd, **kw):
            run_calls.append(cmd)
            # lesskey -o <out> <src>: create the compiled file to simulate success.
            if cmd[0].endswith("lesskey"):
                with open(cmd[2], "w") as fh:
                    fh.write("COMPILED")
                return MagicMock(returncode=0, stderr="")
            return MagicMock(returncode=0, stderr="")

        with (
            patch("agentica.cli.interactive.shutil.which", return_value="/usr/bin/lesskey"),
            patch("agentica.cli.interactive.subprocess.run", side_effect=fake_run),
        ):
            out = it._compile_lesskey("\n#command\n^O quit\n")
        self.assertTrue(out and out.endswith(".bin"))
        self.assertTrue(run_calls and run_calls[0][0].endswith("lesskey"))
        import os as _os

        _os.unlink(out)

    def test_returns_none_when_no_lesskey_binary(self):
        from agentica.cli import interactive as it

        with patch("agentica.cli.interactive.shutil.which", return_value=None):
            self.assertIsNone(it._compile_lesskey("\n#command\n^O quit\n"))


class TestBuildSiblingModel(unittest.TestCase):
    """_build_sibling_model: same-provider inherits main base_url/api_key;
    cross-provider does NOT (would silently produce a broken client)."""

    def _cfg(self, **over):
        base = dict(
            model_provider="deepseek",
            model_name="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="sk-main",
            max_tokens=None,
            temperature=None,
            reasoning_effort=None,
            top_p=None,
            context_window=None,
        )
        base.update(over)
        return base

    def test_none_when_no_sibling_name(self):
        from agentica.cli.runtime import _build_sibling_model

        with patch("agentica.cli.runtime.get_model") as gm:
            self.assertIsNone(_build_sibling_model(self._cfg(), "auxiliary"))
            gm.assert_not_called()

    def test_same_provider_inherits_main_base_and_key(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(auxiliary_model_name="deepseek-chat")  # only name; same provider
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertEqual(kw["model_provider"], "deepseek")
        self.assertEqual(kw["model_name"], "deepseek-chat")
        self.assertEqual(kw["base_url"], "https://api.deepseek.com")
        self.assertEqual(kw["api_key"], "sk-main")

    def test_cross_provider_uses_sibling_base_and_key(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_provider="zhipuai",
            auxiliary_model_name="glm-4.7-flash",
            auxiliary_base_url="https://open.bigmodel.cn/api/paas/v4",
            auxiliary_api_key="sk-zhipu",
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertEqual(kw["model_provider"], "zhipuai")
        self.assertEqual(kw["base_url"], "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(kw["api_key"], "sk-zhipu")

    def test_cross_provider_missing_key_not_filled_with_main_key(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_provider="zhipuai",
            auxiliary_model_name="glm-4.7-flash",
            auxiliary_base_url="https://open.bigmodel.cn/api/paas/v4",
            auxiliary_api_key=None,  # no sibling key
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertIsNone(kw["api_key"])  # must NOT fall back to sk-main
        self.assertEqual(kw["base_url"], "https://open.bigmodel.cn/api/paas/v4")

    def test_cross_provider_missing_base_not_filled_with_main_base(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_provider="zhipuai",
            auxiliary_model_name="glm-4.7-flash",
            auxiliary_base_url=None,  # no sibling base_url
            auxiliary_api_key="sk-zhipu",
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertIsNone(kw["base_url"])  # must NOT fall back to deepseek base_url
        self.assertEqual(kw["api_key"], "sk-zhipu")

    def test_auxiliary_extra_body_passed_and_never_inherits_main(self):
        from agentica.cli.runtime import _build_sibling_model

        cfg = self._cfg(
            auxiliary_model_name="deepseek-chat",  # same provider as main
            extra_body={"main": True},  # main model's own extra_body
            auxiliary_extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
            auxiliary_extra_headers={"X-Aux": "1"},
        )
        with patch("agentica.cli.runtime.get_model") as gm:
            _build_sibling_model(cfg, "auxiliary")
        _args, kw = gm.call_args
        self.assertEqual(kw["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "low"}})
        self.assertEqual(kw["extra_headers"], {"X-Aux": "1"})


class TestCLIAwareness(unittest.TestCase):
    """CLI self-awareness and capability management (Phase 3).

    Covers environment_context injection at create_agent time, /status and
    /agents command registration, _apply_profile carrying the auxiliary_model block
    and refreshing environment_context, and /tools add-from path-traversal
    rejection. All LLM access is mocked (api_key="fake_openai_key").
    """

    @staticmethod
    def _make_demo_tool():
        """Build a real Tool with one registered function for env-context checks."""
        from agentica.tools.base import Tool

        def custom_demo_tool(file_path: str) -> str:
            """Read a file's contents (demo)."""
            return ""

        tool = Tool(name="builtin_file")
        tool.register(custom_demo_tool)
        return tool

    @staticmethod
    def _fake_agent_class():
        """A DeepAgent stand-in that stores tools and accepts session guidance."""

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                self.tools = list(kwargs.get("tools") or [])
                self.session_guidance = []
                self.environment_context = None

            def add_session_guidance(self, text):
                self.session_guidance.append(text)

        return FakeDeepAgent

    def test_environment_context_injected(self):
        """create_agent injects framework/model/tools/subagent/auxiliary self-description."""
        from agentica.cli.runtime import create_agent

        agent_config = {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "fake_openai_key",
            "debug": False,
            "work_dir": None,
            "session_id": "test-session",
            "auxiliary_model_provider": "zhipuai",
            "auxiliary_model_name": "glm-4.7-flash",
            "auxiliary_base_url": "https://open.bigmodel.cn/api/paas/v4",
            "auxiliary_api_key": "fake_openai_key",
        }
        with patch("agentica.agent.deep.DeepAgent", self._fake_agent_class()):
            agent = create_agent(
                agent_config,
                extra_tools=[self._make_demo_tool()],
                workspace=None,
                skills_registry=None,
            )

        ctx = agent.environment_context
        self.assertIsNotNone(ctx)
        self.assertIn("Agentica", ctx)
        self.assertIn("Model: openai/gpt-4o", ctx)
        self.assertIn("Active tools:", ctx)
        self.assertIn("custom_demo_tool", ctx)
        self.assertIn("Subagent types: explore, research, code", ctx)
        self.assertIn("Auxiliary model: zhipuai/glm-4.7-flash", ctx)

    def test_environment_context_omits_auxiliary_when_none(self):
        """No auxiliary_model_* fields -> environment_context has no auxiliary line."""
        from agentica.cli.runtime import create_agent

        agent_config = {
            "model_provider": "openai",
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "fake_openai_key",
            "debug": False,
            "work_dir": None,
            "session_id": "test-session",
        }
        with patch("agentica.agent.deep.DeepAgent", self._fake_agent_class()):
            agent = create_agent(
                agent_config,
                extra_tools=[self._make_demo_tool()],
                workspace=None,
                skills_registry=None,
            )

        ctx = agent.environment_context
        self.assertIsNotNone(ctx)
        self.assertIn("Model: openai/gpt-4o", ctx)
        self.assertNotIn("Auxiliary model:", ctx)

    def test_cmd_status_registered(self):
        self.assertIn("/status", cli_commands.COMMAND_REGISTRY)
        self.assertIs(cli_commands.COMMAND_REGISTRY["/status"][0], cli_commands._cmd_status)

    def test_cmd_agents_registered(self):
        self.assertIn("/agents", cli_commands.COMMAND_REGISTRY)
        self.assertIn("/agent", cli_commands.COMMAND_REGISTRY)
        self.assertIs(cli_commands.COMMAND_REGISTRY["/agents"][0], cli_commands._cmd_agents)
        self.assertIs(cli_commands.COMMAND_REGISTRY["/agent"][0], cli_commands._cmd_agents)

    def _make_apply_profile_ctx(self):
        """Build a CommandContext whose mock agent survives a profile switch.

        Pre-state carries a zhipuai auxiliary so a profile switch to a different auxiliary
        (or to no auxiliary) is observable in agent_config and environment_context.
        """
        mock_agent = MagicMock()
        mock_agent.tools = []
        mock_agent.working_memory.runs = []
        return cli_commands.CommandContext(
            agent_config={
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "fake_openai_key",
                "debug": False,
                "work_dir": None,
                "auxiliary_model_provider": "zhipuai",
                "auxiliary_model_name": "glm-4.7-flash",
                "auxiliary_base_url": "https://open.bigmodel.cn/api/paas/v4",
                "auxiliary_api_key": "fake_openai_key",
            },
            current_agent=mock_agent,
            extra_tools=[],
            workspace=None,
            skills_registry=None,
        )

    def test_apply_profile_carries_auxiliary_model(self):
        """Switching to a profile with an auxiliary_model block rebuilds the auxiliary model
        and refreshes environment_context with the new auxiliary line."""
        from agentica import global_config as gc

        ctx = self._make_apply_profile_ctx()
        # Pre-state is zhipuai/glm-4.7-flash; the profile switches auxiliary to deepseek.
        self.assertEqual(ctx.agent_config["auxiliary_model_name"], "glm-4.7-flash")

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_commands, "get_console", return_value=MagicMock()),
                patch.object(cli_commands, "get_model", return_value=MagicMock()),
                patch.object(cli_commands, "_build_sibling_model", return_value=MagicMock()) as mock_auxiliary,
                # _apply_profile persists a project-scoped override via
                # set_project_profile(work_dir, name); work_dir=None here falls
                # back to os.getcwd(), which would otherwise leak a real
                # ~/.agentica/projects/<repo>/profile file on every test run.
                patch.object(cli_commands, "set_project_profile"),
            ):
                gc.upsert_profile(
                    "withaux",
                    {
                        "model_provider": "openai",
                        "model_name": "gpt-4o",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-main",
                        "auxiliary_model": {
                            "model_provider": "deepseek",
                            "model_name": "deepseek-chat",
                            "base_url": "https://api.deepseek.com",
                            "api_key": "sk-auxiliary",
                        },
                    },
                    make_active=True,
                )
                cli_commands._apply_profile(ctx, "withaux")

        self.assertEqual(ctx.agent_config["auxiliary_model_name"], "deepseek-chat")
        self.assertEqual(ctx.agent_config["auxiliary_model_provider"], "deepseek")
        self.assertIs(ctx.agent_config["auxiliary_model"], mock_auxiliary.return_value)
        self.assertIsNotNone(ctx.current_agent.environment_context)
        self.assertIn(
            "Auxiliary model: deepseek/deepseek-chat",
            ctx.current_agent.environment_context,
        )

    def test_apply_profile_switches_extra_body(self):
        """Switching to a profile with extra_body wires it into agent_config +
        the rebuilt model, main and auxiliary independently."""
        from agentica import global_config as gc

        ctx = self._make_apply_profile_ctx()
        self.assertIsNone(ctx.agent_config.get("extra_body"))

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_commands, "get_console", return_value=MagicMock()),
                patch.object(cli_commands, "get_model") as mock_get_model,
                patch.object(cli_commands, "_build_sibling_model", return_value=MagicMock()),
                patch.object(cli_commands, "set_project_profile"),
            ):
                gc.upsert_profile(
                    "hy3",
                    {
                        "model_provider": "openai",
                        "model_name": "hy3",
                        "base_url": "http://api.taiji.woa.com/openapi/v2",
                        "api_key": "sk-main",
                        "extra_body": {"chat_template_kwargs": {"reasoning_effort": "high"}},
                        "auxiliary_model": {
                            "model_provider": "deepseek",
                            "model_name": "deepseek-chat",
                            "base_url": "https://api.deepseek.com",
                            "api_key": "sk-auxiliary",
                            "extra_body": {"aux": True},
                        },
                    },
                    make_active=True,
                )
                cli_commands._apply_profile(ctx, "hy3")

        self.assertEqual(ctx.agent_config["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "high"}})
        self.assertEqual(ctx.agent_config["auxiliary_extra_body"], {"aux": True})
        # get_model (main model rebuild) received the profile's extra_body.
        _args, kw = mock_get_model.call_args
        self.assertEqual(kw["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "high"}})

    def test_apply_profile_without_auxiliary_clears(self):
        """Switching to a profile without an auxiliary_model block clears the auxiliary fields."""
        from agentica import global_config as gc

        ctx = self._make_apply_profile_ctx()
        self.assertIsNotNone(ctx.agent_config["auxiliary_model_name"])

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_commands, "get_console", return_value=MagicMock()),
                patch.object(cli_commands, "get_model", return_value=MagicMock()),
                patch.object(cli_commands, "_build_sibling_model") as mock_sibling,
                # See test_apply_profile_switches_auxiliary_model above: avoid
                # leaking a real project-profile file to ~/.agentica/projects/.
                patch.object(cli_commands, "set_project_profile"),
            ):
                gc.upsert_profile(
                    "noaux",
                    {
                        "model_provider": "openai",
                        "model_name": "gpt-4o",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-main",
                    },
                    make_active=True,
                )
                cli_commands._apply_profile(ctx, "noaux")

        self.assertIsNone(ctx.agent_config["auxiliary_model_name"])
        self.assertIsNone(ctx.agent_config["auxiliary_model_provider"])
        self.assertIsNone(ctx.agent_config["auxiliary_model"])
        mock_sibling.assert_not_called()
        self.assertNotIn("Auxiliary model:", ctx.current_agent.environment_context)

    def test_cmd_tools_add_from_rejects_path_traversal(self):
        """/tools add-from ../evil is rejected before any module is loaded."""
        ctx = cli_commands.CommandContext(
            agent_config={
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "fake_openai_key",
                "debug": False,
                "work_dir": None,
            },
            current_agent=MagicMock(),
            extra_tools=[],
            workspace=None,
            skills_registry=None,
        )
        mock_console = MagicMock()
        with (
            patch.object(cli_commands, "get_console", return_value=mock_console),
            patch.object(cli_commands, "_load_custom_tool_module") as mock_load,
        ):
            cli_commands._cmd_tools(ctx, "add-from ../evil")

        printed = "\n".join(str(c) for c in mock_console.print.call_args_list)
        self.assertIn("Invalid tool name", printed)
        self.assertIn("../evil", printed)
        mock_load.assert_not_called()


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
        from agentica.cli.interactive import _InputRequest

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


class TestAskActiveFreeze(unittest.TestCase):
    """While a ask_user_question prompt is armed, ``_cprint`` must drop output
    so background ``run_in_terminal`` writes can't starve the main
    prompt_toolkit event loop (the CLI-appears-frozen bug)."""

    def test_cprint_drops_while_ask_active(self):
        import agentica.cli.interactive as it

        prev = it._ask_active[0]
        it._ask_active[0] = True
        try:
            # Must return without raising and without touching the terminal.
            it._cprint("should be dropped")
        finally:
            it._ask_active[0] = prev

    def test_ask_active_defaults_false(self):
        import agentica.cli.interactive as it

        self.assertFalse(it._ask_active[0])


class TestTranscriptPause(unittest.TestCase):
    def test_paused_output_is_buffered_then_flushed_in_order(self):
        import agentica.cli.interactive as it

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
        import agentica.cli.interactive as it

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
        import agentica.cli.interactive as it

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
        import agentica.cli.interactive as it

        with patch.object(it.os, "name", "nt"):
            self.assertIsNone(it._install_sigquit_escape(object()))


class TestCmdPermissions(unittest.TestCase):
    """`/permissions` reads/writes the Agent's own permission_mode directly —
    no separate PermissionManager object anymore."""

    def _make_ctx(self, agent):
        return cli_commands.CommandContext(
            agent_config={"work_dir": None},
            current_agent=agent,
            extra_tools=[],
            workspace=None,
            skills_registry=None,
        )

    def test_set_valid_mode_calls_agent_set_permission_mode(self):
        from agentica.agent import Agent

        agent = Agent()
        ctx = self._make_ctx(agent)
        with patch.object(cli_commands, "get_console", return_value=MagicMock()):
            cli_commands._cmd_permissions(ctx, "ask")

        self.assertEqual(agent.tool_config.permission_mode, "ask")

    def test_set_invalid_mode_does_not_mutate_agent(self):
        from agentica.agent import Agent

        agent = Agent()
        ctx = self._make_ctx(agent)
        with patch.object(cli_commands, "get_console", return_value=MagicMock()):
            cli_commands._cmd_permissions(ctx, "strict")

        self.assertEqual(agent.tool_config.permission_mode, "allow-all")

    def test_no_args_prints_current_mode_without_error(self):
        from agentica.agent import Agent
        from agentica.agent.config import ToolConfig

        agent = Agent(tool_config=ToolConfig(permission_mode="auto"))
        ctx = self._make_ctx(agent)
        console = MagicMock()
        with patch.object(cli_commands, "get_console", return_value=console):
            cli_commands._cmd_permissions(ctx, "")

        self.assertTrue(console.print.called)

    def test_yolo_command_removed_from_registry(self):
        self.assertNotIn("/yolo", cli_commands.COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
