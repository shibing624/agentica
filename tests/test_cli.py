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

    def test_stream_display_manager_box_decorations(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.start_response()
        self.assertTrue(dm._box_opened)
        dm.stream_response("hello")
        dm.finalize()
        calls = [str(c) for c in fake.print.call_args_list]
        box_open = any("╭" in c for c in calls)
        box_close = any("╰" in c for c in calls)
        self.assertTrue(box_open, "Expected ╭ box opening")
        self.assertTrue(box_close, "Expected ╰ box closing")

    def test_stream_display_manager_suppresses_micro_compact(self):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.handle_event({"type": "compact.micro", "agent_name": "Agent", "cleared": 3})
        fake.print.assert_not_called()

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
            "read_file", "FileNotFoundError: nope", is_error=True, elapsed=0.01,
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

    def test_display_execute_folds_at_ten_lines(self):
        """execute shows up to 10 lines then folds; other tools still fold at 4."""
        from agentica.cli.display import StreamDisplayManager

        # 10 lines: fully shown, no fold hint
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result(
            "execute", "\n".join(f"line {i}" for i in range(10)),
            is_error=False, elapsed=0.1,
        )
        text = "\n".join(str(c) for c in fake.print.call_args_list)
        self.assertNotIn("more lines", text)

        # 12 lines: folds, hint shows 2 more
        fake2 = MagicMock()
        fake2.width = 80
        dm2 = StreamDisplayManager(fake2)
        dm2.display_tool_result(
            "execute", "\n".join(f"line {i}" for i in range(12)),
            is_error=False, elapsed=0.1,
        )
        text2 = "\n".join(str(c) for c in fake2.print.call_args_list)
        self.assertIn("2 more lines", text2)

    def test_truncated_blocks_are_remembered_for_expand(self):
        """Long user input and long tool output are stashed for Ctrl+o expansion."""
        from agentica.cli import display as disp
        from agentica.cli.display import StreamDisplayManager

        # Long user input (>12 lines) is remembered.
        disp.remember_truncated("", "")  # reset
        long_input = "\n".join(f"line {i}" for i in range(20))
        disp.display_user_message(long_input)
        block = disp.get_last_truncated()
        self.assertIn("User input", block["title"])
        self.assertEqual(block["content"], long_input)

        # Long tool output (>max_lines) is remembered.
        long_output = "\n".join(f"out {i}" for i in range(50))
        fake = MagicMock()
        fake.width = 80
        dm = StreamDisplayManager(fake)
        dm.display_tool_result("execute", long_output, is_error=False, elapsed=0.5)
        block = disp.get_last_truncated()
        self.assertIn("execute", block["title"])
        self.assertEqual(block["content"], long_output)

        # Short output is NOT remembered (no truncation → nothing to expand).
        disp.remember_truncated("", "")
        dm.display_tool_result("execute", "only one line", is_error=False, elapsed=0.1)
        self.assertEqual(disp.get_last_truncated()["content"], "")


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
        from agentica.cli.config import get_model

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
        from agentica.cli.config import get_model

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

    def test_anthropic_accepts_top_p_skips_reasoning_effort(self):
        from agentica.cli.config import get_model

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
        self.assertFalse(hasattr(model, "reasoning_effort"))

    def test_reasoning_effort_accepts_low_medium(self):
        import sys
        from agentica.cli.config import parse_args

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
            aux_model_provider=None,
            aux_model_name=None,
            aux_base_url=None,
            aux_api_key=None,
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
        from agentica.cli.config import parse_args

        with patch.object(sys, "argv", ["agentica"]):
            args = parse_args()

        self.assertIsNone(args.model_provider)
        self.assertIsNone(args.model_name)
        self.assertIsNone(args.reasoning_effort)
        self.assertFalse(args.enable_diagnostics)
        self.assertIsNone(args.diagnostics_servers)

    def test_parse_diagnostics_flags(self):
        from agentica.cli.config import parse_args

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
        from agentica.cli.config import parse_args

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
            aux_model_provider=None,
            aux_model_name=None,
            aux_base_url=None,
            aux_api_key=None,
        )
        with patch("agentica.cli.setup.get_profile", return_value={}):
            resolved = resolve_model_config(args, console=None)

        self.assertEqual(resolved["model_provider"], "deepseek")
        self.assertEqual(resolved["model_name"], "deepseek-v4-flash")

    def test_get_model_defaults_deepseek_cli_reasoning_effort_to_max(self):
        """CLI DeepSeek usage should default to max effort for agentic tasks."""
        from agentica.cli.config import get_model

        model = get_model("deepseek", "deepseek-v4-flash", api_key="fake_key")

        self.assertEqual(model.reasoning_effort, "max")

    def test_get_model_respects_explicit_deepseek_reasoning_effort(self):
        """Explicit CLI reasoning effort should override the agentic default."""
        from agentica.cli.config import get_model

        model = get_model(
            "deepseek",
            "deepseek-v4-flash",
            api_key="fake_key",
            reasoning_effort="high",
        )

        self.assertEqual(model.reasoning_effort, "high")

    def test_create_agent_uses_deepseek_cli_reasoning_default(self):
        """DeepAgent creation should inherit the CLI's max-thinking default."""
        from agentica.cli.config import create_agent

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
        from agentica.cli.config import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.config.get_model", return_value=MagicMock()),
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

    def test_persist_model_choice_writes_base_url(self):
        from agentica import global_config as gc
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with patch("agentica.global_config.global_config_path", return_value=cfg_path):
                gc.upsert_profile("default", {
                    "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com", "api_key": "sk-x",
                })
                cli_commands._persist_model_choice(
                    "openai",
                    "gpt-5",
                    "https://api.openai.com/v1",
                )
                saved = gc.get_profile("default")

        self.assertEqual(saved["model_provider"], "openai")
        self.assertEqual(saved["model_name"], "gpt-5")
        self.assertEqual(saved["base_url"], "https://api.openai.com/v1")
        # api_key already in the profile is preserved across the live switch.
        self.assertEqual(saved["api_key"], "sk-x")

    def test_model_command_switch_provider_resets_and_persists_base_url(self):
        from agentica import global_config as gc
        ctx = cli_commands.CommandContext(
            agent_config={
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key": "fake_openai_key",
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
            ):
                gc.upsert_profile("default", {
                    "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com", "api_key": "sk-x",
                })
                cli_commands._cmd_model(ctx, "openai/gpt-5")
                saved = gc.get_profile("default")

        self.assertEqual(ctx.agent_config["base_url"], "https://api.openai.com/v1")
        self.assertEqual(saved["base_url"], "https://api.openai.com/v1")
        self.assertEqual(saved["model_provider"], "openai")
        self.assertEqual(saved["model_name"], "gpt-5")
        self.assertEqual(mock_get_model.call_args.kwargs["base_url"], "https://api.openai.com/v1")

    def test_model_command_same_provider_preserves_custom_base_url(self):
        from agentica import global_config as gc
        ctx = cli_commands.CommandContext(
            agent_config={
                "model_provider": "openai",
                "model_name": "gpt-4o",
                "base_url": "https://proxy.example/v1",
                "api_key": "fake_openai_key",
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
            ):
                gc.upsert_profile("default", {
                    "model_provider": "openai", "model_name": "gpt-4o",
                    "base_url": "https://proxy.example/v1", "api_key": "sk-x",
                })
                cli_commands._cmd_model(ctx, "gpt-5")
                saved = gc.get_profile("default")

        self.assertEqual(ctx.agent_config["base_url"], "https://proxy.example/v1")
        self.assertEqual(saved["base_url"], "https://proxy.example/v1")
        self.assertEqual(mock_get_model.call_args.kwargs["base_url"], "https://proxy.example/v1")

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
        from agentica.cli.config import parse_args

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
        from agentica.cli.config import parse_args

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
        from agentica.cli.config import parse_args

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
        from agentica.cli.config import parse_args

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
        from agentica.cli.config import create_agent
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
            patch("agentica.cli.config.get_model", return_value=MagicMock()),
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
        from agentica.cli.config import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.config.get_model", return_value=MagicMock()),
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
        from agentica.cli.config import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.config.get_model", return_value=MagicMock()),
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


class TestStreamDisplayManagerCompletionTimestamp(unittest.TestCase):
    """The Response box must close with a small ``(HH:MM:SS)`` label embedded
    on its bottom-right corner so users reviewing a long session can see
    when each answer landed.
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

    def test_finalize_embeds_completion_timestamp_in_close_rule(self):
        import re

        def render(mgr):
            mgr.start_response()
            mgr.stream_response("hello")
            mgr.finalize()

        out = self._capture(render)
        self.assertIn("Response", out)
        self.assertRegex(out, r"\(\d{2}:\d{2}:\d{2}\)", "close rule must embed a HH:MM:SS timestamp")
        # Timestamp must appear on the closing rule (the line containing ╯),
        # not somewhere in the body.
        close_lines = [ln for ln in out.splitlines() if "╯" in ln]
        self.assertTrue(close_lines, "response box must have a close rule")
        self.assertRegex(close_lines[-1], r"\(\d{2}:\d{2}:\d{2}\)")


class TestLessLesskeyDetection(unittest.TestCase):
    """Ctrl+o expand pager: detect --lesskey-content support (less's help
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
    """Old-less fallback: compile a lesskey file to bind Ctrl+o to quit when
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

        with patch("agentica.cli.interactive.shutil.which", return_value="/usr/bin/lesskey"), \
             patch("agentica.cli.interactive.subprocess.run", side_effect=fake_run):
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
        from agentica.cli.config import _build_sibling_model
        with patch("agentica.cli.config.get_model") as gm:
            self.assertIsNone(_build_sibling_model(self._cfg(), "aux"))
            gm.assert_not_called()

    def test_same_provider_inherits_main_base_and_key(self):
        from agentica.cli.config import _build_sibling_model
        cfg = self._cfg(aux_model_name="deepseek-chat")  # only name; same provider
        with patch("agentica.cli.config.get_model") as gm:
            _build_sibling_model(cfg, "aux")
        _args, kw = gm.call_args
        self.assertEqual(kw["model_provider"], "deepseek")
        self.assertEqual(kw["model_name"], "deepseek-chat")
        self.assertEqual(kw["base_url"], "https://api.deepseek.com")
        self.assertEqual(kw["api_key"], "sk-main")

    def test_cross_provider_uses_sibling_base_and_key(self):
        from agentica.cli.config import _build_sibling_model
        cfg = self._cfg(
            aux_model_provider="zhipuai",
            aux_model_name="glm-4.7-flash",
            aux_base_url="https://open.bigmodel.cn/api/paas/v4",
            aux_api_key="sk-zhipu",
        )
        with patch("agentica.cli.config.get_model") as gm:
            _build_sibling_model(cfg, "aux")
        _args, kw = gm.call_args
        self.assertEqual(kw["model_provider"], "zhipuai")
        self.assertEqual(kw["base_url"], "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(kw["api_key"], "sk-zhipu")

    def test_cross_provider_missing_key_not_filled_with_main_key(self):
        from agentica.cli.config import _build_sibling_model
        cfg = self._cfg(
            aux_model_provider="zhipuai",
            aux_model_name="glm-4.7-flash",
            aux_base_url="https://open.bigmodel.cn/api/paas/v4",
            aux_api_key=None,  # no sibling key
        )
        with patch("agentica.cli.config.get_model") as gm:
            _build_sibling_model(cfg, "aux")
        _args, kw = gm.call_args
        self.assertIsNone(kw["api_key"])  # must NOT fall back to sk-main
        self.assertEqual(kw["base_url"], "https://open.bigmodel.cn/api/paas/v4")

    def test_cross_provider_missing_base_not_filled_with_main_base(self):
        from agentica.cli.config import _build_sibling_model
        cfg = self._cfg(
            aux_model_provider="zhipuai",
            aux_model_name="glm-4.7-flash",
            aux_base_url=None,  # no sibling base_url
            aux_api_key="sk-zhipu",
        )
        with patch("agentica.cli.config.get_model") as gm:
            _build_sibling_model(cfg, "aux")
        _args, kw = gm.call_args
        self.assertIsNone(kw["base_url"])  # must NOT fall back to deepseek base_url
        self.assertEqual(kw["api_key"], "sk-zhipu")


if __name__ == "__main__":
    unittest.main()
