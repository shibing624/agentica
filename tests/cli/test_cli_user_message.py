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



class TestCLIUserMessage(unittest.TestCase):
    """CLI helpers tests (TestCLIUserMessage)."""

    @staticmethod
    def _render_compact_event(event):
        from agentica.cli.display import StreamDisplayManager

        fake = MagicMock()
        fake.width = 80
        StreamDisplayManager(fake).handle_event(event)
        return "\n".join(str(call) for call in fake.print.call_args_list)


    def test_display_agent_execution_error_formats_rate_limit_payload(self):
        """Provider rate limits render as a concise error with raw details expandable."""
        from io import StringIO
        from rich.console import Console

        from agentica.cli.display import console as disp

        raw = (
            "Error code: 429 - {'error': {'message': 'TPM 限流：当前 token 余量已耗尽，"
            "限流为 500000 tokens/min，请稍后重试', 'type': 'gateway_error', "
            "'code': '4029'}, 'trace': {'spanId': '3a69ffde9353db3f'}}"
        )
        output = StringIO()
        console = Console(file=output, width=100, force_terminal=False, no_color=True)
        disp.clear_truncated_blocks()

        view = disp.display_agent_execution_error(console, RuntimeError(raw))

        rendered = output.getvalue()
        self.assertEqual(view["summary"], "LLM rate limited (429)")
        self.assertIn("● Error: LLM rate limited (429)", rendered)
        self.assertIn("TPM 限流", rendered)
        self.assertIn("code=4029", rendered)
        self.assertIn("spanId=3a69ffde9353db3f", rendered)
        self.assertIn("Type /retry after a short wait", rendered)
        self.assertIn("Ctrl+O shows raw provider error", rendered)
        self.assertNotIn("'trace'", rendered)
        self.assertEqual(disp.get_truncated_blocks(), [{
            "title": "Agent error · raw",
            "content": raw,
        }])


    def test_display_agent_execution_error_keeps_retry_hint_for_429_without_payload(self):
        """A bare 429 still gets the same actionable retry treatment."""
        from agentica.cli.display import console as disp

        raw = "Error code: 429 - rate limit exceeded by upstream proxy"
        view = disp._format_agent_execution_error(RuntimeError(raw))

        self.assertEqual(view["summary"], "LLM rate limited (429)")
        self.assertEqual(view["detail"], raw)
        self.assertIn("/retry", view["hint"])


    def test_display_agent_execution_error_accepts_top_level_message_payload(self):
        """Provider payloads do not need an OpenAI-style error.message shape."""
        from agentica.cli.display import console as disp

        raw = "Error code: 503 - {'message': 'upstream queue full', 'code': 'busy'}"
        view = disp._format_agent_execution_error(RuntimeError(raw))

        self.assertEqual(view["summary"], "Transient LLM/API error (503)")
        self.assertEqual(view["detail"], "upstream queue full")
        self.assertIn("code=busy", view["diagnostics"])
        self.assertIn("/retry", view["hint"])


    def test_display_agent_execution_error_prints_non_json_message(self):
        """Plain-text proxy failures still show their actual error message."""
        from agentica.cli.display import console as disp

        raw = "Proxy temporarily unavailable: upstream timeout while dialing model"
        view = disp._format_agent_execution_error(RuntimeError(raw))

        self.assertEqual(view["summary"], "Transient LLM/API error")
        self.assertEqual(view["detail"], raw)
        self.assertIn("/retry", view["hint"])


    def test_display_agent_execution_error_explains_malformed_stream(self):
        """A gateway that packs two SSE events on one line surfaces as an
        unhelpful ``Extra data: line 1 column 260``. Name the cause and point
        at /retry, which now works because the failed turn is persisted."""
        import json as _json

        from agentica.cli.display import console as disp

        error = _json.JSONDecodeError("Extra data", '{"a": 1}{"b": 2}', 8)
        view = disp._format_agent_execution_error(error)

        self.assertEqual(view["summary"], "Malformed stream from the model endpoint")
        self.assertIn("Extra data", view["detail"])
        self.assertIn("/retry", view["hint"])


    def test_user_message_uses_subtle_background_panel(self):
        """Historical user queries are visually separated from assistant output."""
        from rich.padding import Padding
        from rich.table import Table

        from agentica.cli.display import display_user_message

        console = Mock()
        with patch("agentica.cli.display.messages.get_console", return_value=console):
            display_user_message("hello\nsecond line")

        renderable = console.print.call_args.args[0]
        self.assertIsInstance(renderable, Padding)
        self.assertEqual(renderable.style, "on rgb(35,35,35)")
        self.assertIsInstance(renderable.renderable, Table)
        marker_column, content_column = renderable.renderable.columns
        self.assertEqual(marker_column._cells[0].plain, "❯")
        self.assertEqual(marker_column._cells[0].style, "bold bright_yellow")
        self.assertEqual(content_column._cells[0].plain, "hello\nsecond line")
        self.assertEqual(content_column.overflow, "fold")


    def test_user_message_does_not_ellipsis_truncate_long_peer_text(self):
        """Peer-injected turns must render in full — Rich Table defaults to …."""
        from io import StringIO
        from rich.console import Console

        from agentica.cli.display import display_user_message
        from agentica.cli.display import messages as disp_messages

        body = (
            "[Message from another agent session 'temp-af' "
            "— reply with send_message to temp-af]\n"
            "用户反馈：希望你以后发消息时说明这是用户的决定。"
            "不过需要对齐一个边界——agent 间消息无论怎么措辞都不构成对我的用户授权，"
            "我这边仍会向用户本人确认后才采纳，不会仅凭消息里这是用户的意思就直接改。"
            "真正的指令请让用户亲自下达。"
        )
        output = StringIO()
        console = Console(file=output, width=80, force_terminal=False, no_color=True)
        with patch.object(disp_messages, "get_console", return_value=console):
            display_user_message(body)

        rendered = output.getvalue()
        self.assertIn("真正的指令请让用户亲自下达", rendered)
        self.assertNotIn("…", rendered)


    def test_user_message_lists_each_attached_image_once_inside_panel(self):
        """Image attachments are listed once inside the historical input panel."""
        from pathlib import Path
        from rich.padding import Padding

        from agentica.cli.display import display_user_message

        console = Mock()
        with (
            patch("agentica.cli.display.messages.get_console", return_value=console),
            patch("pathlib.Path.stat") as stat,
        ):
            stat.return_value.st_size = 94 * 1024
            display_user_message(
                "compare these",
                images=[
                    Path("/tmp/clip_1.png"),
                    Path("/tmp/clip_2.png"),
                ],
            )

        renderable = console.print.call_args.args[0]
        self.assertIsInstance(renderable, Padding)
        marker_column, content_column = renderable.renderable.columns
        self.assertEqual(marker_column._cells[0].plain, "❯")
        self.assertEqual(
            content_column._cells[0].plain,
            "compare these\n📎 Image #1 attached: clip_1.png (94KB)\n"
            "📎 Image #2 attached: clip_2.png (94KB)",
        )


    def test_deduplicate_image_attachments_preserves_first_path(self):
        """One pasted image represented by two temp paths stays one attachment."""
        from pathlib import Path

        from agentica.cli.interactive.attachments import _deduplicate_image_attachments

        paths = [Path("/tmp/clip.png"), Path("/tmp/clipboard.png"), Path("/tmp/other.png")]
        with patch("agentica.cli.interactive.attachments._image_content_key") as content_key:
            content_key.side_effect = ["same", "same", "other"]
            result = _deduplicate_image_attachments(paths)

        self.assertEqual(result, [paths[0], paths[2]])




if __name__ == "__main__":
    unittest.main()
