# -*- coding: utf-8 -*-
"""Slash-command completion: full list on `/`, then fuzzy rank."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cli.commands.registry import COMMAND_REGISTRY
from agentica.cli.interactive.complete import (
    rank_slash_commands,
    score_slash_command,
    slash_command_rows,
)


def _rows(*pairs):
    return [(name, name, desc) for name, desc in pairs]


class TestSlashCommandScore(unittest.TestCase):
    def test_bare_slash_keeps_every_row_in_registry_order(self):
        rows = slash_command_rows((name, desc) for name, (_, desc) in COMMAND_REGISTRY.items())
        ranked = rank_slash_commands("/", rows)

        self.assertEqual([name for name, _d, _m in ranked], list(COMMAND_REGISTRY))
        self.assertGreater(len(ranked), 10)

    def test_prefix_still_wins(self):
        rows = _rows(
            ("/resume", "Resume a session"),
            ("/reset", "Clear screen"),
            ("/peername", "Show the peer name"),
        )
        ranked = [name for name, _d, _m in rank_slash_commands("/res", rows)]

        self.assertEqual(ranked, ["/reset", "/resume"])

    def test_subsequence_finds_peername_from_pn(self):
        rows = _rows(
            ("/peername", "Show the peer name"),
            ("/permissions", "Set permission mode"),
            ("/status", "Session status"),
        )
        ranked = [name for name, _d, _m in rank_slash_commands("/pn", rows)]

        self.assertIn("/peername", ranked)
        self.assertLess(ranked.index("/peername"), ranked.index("/permissions"))
        self.assertNotIn("/status", ranked)

    def test_substring_inside_the_name(self):
        rows = _rows(("/list-agents", "List live sessions"), ("/help", "Show commands"))
        ranked = [name for name, _d, _m in rank_slash_commands("/agent", rows)]

        self.assertEqual(ranked, ["/list-agents"])

    def test_description_match_needs_three_characters(self):
        rows = _rows(("/help", "Toggle verbose debug logging"))

        # One letter would otherwise hit every description that contains it.
        self.assertEqual(rank_slash_commands("/v", rows), [])
        ranked = [name for name, _d, _m in rank_slash_commands("/verbose", rows)]
        self.assertEqual(ranked, ["/help"])

    def test_unknown_query_is_empty(self):
        self.assertEqual(rank_slash_commands("/zzzz", _rows(("/help", "Show commands"))), [])

    def test_skill_auto_commands_follow_the_registry_and_skip_collisions(self):
        rows = slash_command_rows(
            [("/help", "Show commands")],
            [("/help", "/help (Dup)", "should skip"), ("/review", "/review (Review)", "Review a PR")],
        )
        ranked = rank_slash_commands("/", rows)

        self.assertEqual(ranked, [("/help", "/help", "Show commands"), ("/review", "/review (Review)", "Review a PR")])

    def test_non_slash_query_scores_nothing(self):
        self.assertIsNone(score_slash_command("help", "/help", "Show commands"))


class TestCompletionsMenuSitsAboveTheInput(unittest.TestCase):
    """The TUI is pinned to the bottom of the terminal.

    A ycursor Float opens downward and is clipped to the status-bar row
    (or nothing). The menu has to live in the HSplit above the input so
    typing ``/`` actually shows the command list.
    """

    def test_completions_menu_is_above_the_input_area(self):
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.menus import CompletionsMenu

        from agentica.cli.commands.context import PendingQueue
        from agentica.cli.interactive.session_state import SessionState
        from agentica.cli.interactive.tui import _setup_tui

        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "agentica.cli.interactive.tui.history_file",
                return_value=os.path.join(tmp, "history"),
            ):
                app = _setup_tui(
                    SessionState(),
                    skills_registry=None,
                    tui_state={},
                    pending_queue=PendingQueue(),
                    image_counter_ref=[0],
                )

        body = app.layout.container
        self.assertIsInstance(body, HSplit)
        children = list(body.children)
        menu_at = next(i for i, child in enumerate(children) if isinstance(child, CompletionsMenu))
        # TextArea unwraps to a Window; it must sit directly under the menu.
        input_window = children[menu_at + 1]
        self.assertIsInstance(input_window, Window)
        self.assertIsInstance(input_window.content, BufferControl)


if __name__ == "__main__":
    unittest.main()
