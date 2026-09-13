# -*- coding: utf-8 -*-
"""Slash-command completion: full list on `/`, then fuzzy rank."""

import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
