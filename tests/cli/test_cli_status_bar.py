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



class TestCLIStatusBar(unittest.TestCase):
    """CLI helpers tests (TestCLIStatusBar)."""

    @staticmethod
    def _render_compact_event(event):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        StreamDisplayManager(fake).handle_event(event)
        return "\n".join(str(call) for call in fake.print.call_args_list)


    def test_display_token_stats_shows_context_usage(self):
        from agentica.cli.display import display_token_stats

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=50)

        fake_console = MagicMock()
        display_token_stats(
            fake_console,
            tracker,
            context_window=128000,
            context_tokens=64000,
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
            context_tokens=700,
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
            context_tokens=150,
            tool_use_count=0,
            elapsed_seconds=0.5,
        )

        rendered = fake_console.print.call_args[0][0]
        self.assertNotIn("tool", rendered)


    def test_display_token_stats_does_not_treat_turn_usage_as_context(self):
        """API consumption cannot be used as the current context watermark."""
        from agentica.cli.display import display_token_stats

        tracker = CostTracker()
        tracker.record("gpt-4o-mini", input_tokens=2000, output_tokens=500)

        fake_console = MagicMock()
        display_token_stats(fake_console, tracker, context_window=128000)

        rendered = fake_console.print.call_args[0][0]
        self.assertIn("0 / 128K", rendered)
        self.assertNotIn("2.5K / 128K", rendered)


    def test_format_tokens_short(self):
        from agentica.cli.display.status_bar import _format_tokens_short

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
        self.assertNotIn("ctx ", text)
        self.assertIn("50%", text)
        self.assertIn("$0.05", text)
        self.assertIn("⏱ 12.3s", text)
        self.assertIn("Σ 1m45s", text)
        self.assertNotIn("░", text)
        self.assertNotIn("█", text)


    def test_build_status_bar_fragments_shows_goal_tokens(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=1000,
            context_window=128000,
            goal_tokens_used=12_300,
            goal_token_budget=500_000,
            terminal_width=120,
        )
        text = "".join(v for _, v in frags)
        self.assertIn("goal 12.3K/500K", text)


    def test_build_status_bar_fragments_shows_project_identity(self):
        from pathlib import Path

        from agentica.cli.display import build_status_bar_fragments

        work_dir = str(Path.home() / "Documents" / "Codes" / "dual-mem")
        frags = build_status_bar_fragments(
            model_name="gpt-5.6-sol",
            thinking_mode="high",
            work_dir=work_dir,
            git_branch="main",
            profile_name="default",
            context_tokens=24_200,
            context_window=1_100_000,
            terminal_width=180,
        )
        text = "".join(v for _, v in frags)

        self.assertIn("gpt-5.6-sol high", text)
        self.assertIn("~/Documents/Codes/dual-mem", text)
        self.assertIn("default gpt-5.6-sol high", text)
        self.assertNotIn("profile:", text)
        self.assertIn("~/Documents/Codes/dual-mem · main", text)
        self.assertNotIn("main [default]", text)
        self.assertIn("24.2K/1.1M 2%", text)
        self.assertNotIn("░", text)


    def test_build_status_bar_fragments_shows_peer_identity_between_branch_and_context(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-5",
            model_provider="openai",
            work_dir="/repo",
            git_branch="main",
            peer_name="agentica-aa",
            context_tokens=9_000,
            context_window=100_000,
            cost_usd=0.13,
            terminal_width=120,
        )
        text = "".join(v for _, v in frags)

        self.assertIn("openai/gpt-5 │ /repo · main │ agentica-aa │ 9K/100K 9% │ $0.13", text)
        self.assertLess(text.index("main"), text.index("agentica-aa"))
        self.assertLess(text.index("agentica-aa"), text.index("9K/100K"))


    def test_build_status_bar_fragments_hides_peer_identity_when_narrow(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-5",
            model_provider="openai",
            work_dir="/repo",
            git_branch="main",
            peer_name="agentica-aa",
            context_tokens=9_000,
            context_window=100_000,
            cost_usd=0.13,
            terminal_width=42,
        )
        text = "".join(v for _, v in frags)

        self.assertNotIn("agentica-aa", text)


    def test_build_status_bar_fragments_compacts_project_to_fit(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-5.6-sol",
            thinking_mode="high",
            work_dir="/very/long/path/to/dual-mem",
            git_branch="main",
            profile_name="default",
            context_tokens=24_200,
            context_window=1_100_000,
            terminal_width=80,
        )
        text = "".join(v for _, v in frags)

        self.assertLessEqual(len(text), 80)
        self.assertIn("gpt-5.6-sol high", text)
        self.assertIn("2%", text)


    def test_build_status_bar_fragments_preserves_timing_on_the_right(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-5.6-sol",
            model_provider="openai",
            thinking_mode="high",
            work_dir="/very/long/path/to/dual-mem",
            git_branch="main",
            profile_name="proxy-gpt-5.6-sol",
            context_tokens=24_200,
            context_window=1_100_000,
            cost_usd=0.64,
            last_turn_seconds=239.0,
            active_seconds=238.0,
            terminal_width=100,
        )
        text = "".join(v for _, v in frags)

        self.assertLessEqual(len(text), 100)
        self.assertTrue(text.rstrip().endswith("│ ⏱ 239.0s  Σ 3m58s"))


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
        self.assertNotIn("ctx ", text)
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


    def test_status_bar_shows_background_terminal_count(self):
        from agentica.cli.display import build_status_bar_fragments

        frags = build_status_bar_fragments(
            model_name="gpt-4o",
            context_tokens=64000,
            context_window=128000,
            spinner_text="⠋",
            terminal_width=140,
            agent_running=True,
            background_terminal_count=1,
        )
        text = "".join(v for _, v in frags)

        self.assertIn("1 background terminal running", text)
        self.assertIn("/ps to view", text)
        # The hint must carry the argument: a bare /stop no longer stops anything.
        self.assertIn("/stop <id> to close", text)


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


    def test_build_status_bar_fragments_shows_profile(self):
        """After `/model profile <name>` the status bar must show the new
        profile and provider/model label — driven entirely by the
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
        self.assertIn("work deepseek/deepseek-v4-flash", text)
        self.assertNotIn("profile:", text)
        self.assertNotIn("[work]", text)
        self.assertIn("deepseek/deepseek-v4-flash", text)
        self.assertLess(text.index("work"), text.index("deepseek/deepseek-v4-flash"))




if __name__ == "__main__":
    unittest.main()
