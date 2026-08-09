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
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.registry import COMMAND_REGISTRY
from agentica.cli.commands import model_config as cli_model_config
from agentica.cli.commands import session as cli_session
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



class TestRenameCommand(unittest.TestCase):
    """`/rename <name>` persists a recognizable label for `/resume`."""

    def _ctx_with_agent(self, tmp_dir, session_id="sess-cli"):
        session_log = SessionLog(session_id, base_dir=str(tmp_dir))
        session_log.append("user", "first turn")

        agent = MagicMock()
        agent.session_id = session_id
        agent._session_log = session_log

        context = CommandContext(
            agent_config={"model_provider": "zhipuai", "model_name": "glm-5"},
            current_agent=agent,
            extra_tools=[],
            workspace=None,
        )
        return context, session_log

    def test_rename_current_session_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)

            cli_session._cmd_rename(context, "My favourite session")

            self.assertEqual(session_log.get_name(), "My favourite session")

    def test_rename_strips_name(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)

            cli_session._cmd_rename(context, "  Release investigation  ")

            self.assertEqual(session_log.get_name(), "Release investigation")

    def test_rename_rejects_empty_name(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)

            console = MagicMock()
            with patch("agentica.cli.commands.session.get_console", return_value=console):
                cli_session._cmd_rename(context, "   ")

            self.assertIsNone(session_log.get_name())
            printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
            self.assertIn("Current session:", printed)
            self.assertIn("sess-cli", printed)
            self.assertIn("Usage: /rename <name>", printed)

    def test_rename_without_name_shows_current_name_and_id(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)
            session_log.set_name("Release investigation")
            console = MagicMock()

            with patch("agentica.cli.commands.session.get_console", return_value=console):
                cli_session._cmd_rename(context, "")

            printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
            self.assertIn("Release investigation (sess-cli)", printed)

    def test_rename_requires_active_session(self):
        context = CommandContext(agent_config={}, current_agent=None)
        console = MagicMock()
        with patch("agentica.cli.commands.session.get_console", return_value=console):
            cli_session._cmd_rename(context, "Orphan")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
        self.assertIn("No active session", printed)

    def test_rename_reports_metadata_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            context, session_log = self._ctx_with_agent(directory)
            console = MagicMock()
            with (
                patch.object(session_log, "set_name", side_effect=OSError("disk full")),
                patch("agentica.cli.commands.session.get_console", return_value=console),
            ):
                cli_session._cmd_rename(context, "Important session")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
        self.assertIn("Failed to rename session: disk full", printed)

    def test_rename_replaces_session_command(self):
        self.assertIs(
            COMMAND_REGISTRY["/rename"][0],
            cli_session._cmd_rename,
        )
        self.assertNotIn("/session", COMMAND_REGISTRY)


class TestStatusSessionIdentity(unittest.TestCase):
    def test_status_shows_current_session_name_and_id(self):
        session_log = MagicMock()
        session_log.get_name.return_value = "Release investigation"
        agent = MagicMock()
        agent.session_id = "sess-current-1234"
        agent._session_log = session_log
        agent.tools = []
        agent.tool_config.permission_mode = "allow-all"
        agent.run_response.cost_tracker = None
        context = CommandContext(
            agent_config={"model_provider": "openai", "model_name": "gpt-4o"},
            current_agent=agent,
            tui_state={},
            peer_session=SimpleNamespace(name="agentica-73", peer_id="735ac7e4"),
        )
        console = MagicMock()

        with (
            patch("agentica.cli.commands.model_config.get_console", return_value=console),
            patch("agentica.cli.commands.model_config.resolve_active_profile_name", return_value=("default", "default")),
            patch("agentica.cli.commands.model_config.get_subagent_configs", return_value={}),
        ):
            cli_model_config._cmd_status(context)

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list)
        self.assertIn("Session:", printed)
        self.assertIn("Release investigation (sess-current-1234)", printed)
        self.assertIn("Peer:", printed)
        self.assertIn("agentica-73", printed)
        self.assertIn("735ac7e4", printed)


class TestResumeArchivedFilter(unittest.TestCase):
    """``/resume`` must respect the ``archived`` sidecar flag: the picker
    (bare ``/resume`` or ``/resume <number>``) hides archived sessions, but
    an explicit id/prefix still resumes them directly — same cross-surface
    semantics the Web UI sidebar already enforces.
    """

    def _sessions(self):
        # work_dir matches the process cwd so these cases never hit the
        # "session started elsewhere" prompt — that path has its own tests.
        here = os.getcwd()
        return [
            {
                "session_id": "sess-active-1111",
                "path": "/tmp/sess-active-1111.jsonl",
                "base_dir": "/tmp",
                "work_dir": here,
                "size_bytes": 100,
                "mtime": 1.0,
                "last_timestamp": "2026-01-01T00:00:00",
                "name": "Release investigation",
                "archived": False,
            },
            {
                "session_id": "sess-archived-2222",
                "path": "/tmp/sess-archived-2222.jsonl",
                "base_dir": "/tmp",
                "work_dir": here,
                "size_bytes": 100,
                "mtime": 2.0,
                "last_timestamp": "2026-01-02T00:00:00",
                "name": "Archived work",
                "archived": True,
            },
            {
                "session_id": "sess-active-3333",
                "path": "/tmp/sess-active-3333.jsonl",
                "base_dir": "/tmp",
                "work_dir": here,
                "size_bytes": 100,
                "mtime": 3.0,
                "last_timestamp": "2026-01-03T00:00:00",
                "name": None,
                "archived": False,
            },
        ]

    def _resume(self, target, sessions=None):
        context = CommandContext(
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
            patch("agentica.cli.commands.session.create_agent") as create_agent,
            patch("agentica.cli.commands.session.GoalManager") as goal_manager,
        ):
            agent = MagicMock()
            agent._session_log = None
            create_agent.return_value = agent
            goal_manager.return_value.load.return_value = None
            result = cli_session._cmd_resume(context, target)
        return create_agent, result

    def test_picker_listing_excludes_archived(self):
        ctx = CommandContext(agent_config={}, current_agent=None)
        with (
            patch("agentica.memory.session_log.SessionLog.list_sessions", return_value=self._sessions()),
            patch(
                "agentica.memory.session_log.SessionLog.session_preview",
                return_value={"user_count": 0, "first_user": None},
            ),
        ):
            with patch("agentica.cli.commands.session.get_console") as mock_console:
                console = MagicMock()
                mock_console.return_value = console
                cli_session._cmd_resume(ctx, "")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        self.assertGreaterEqual(printed.count("sess-act"), 2)  # both active sessions listed
        self.assertIn("Release investigation", printed)
        self.assertIn("/resume <number|name|id-prefix>", printed)
        self.assertNotIn("sess-arc", printed)

    def test_picker_marks_current_session(self):
        current_agent = MagicMock()
        current_agent.session_id = "sess-active-3333"
        current_agent._session_log = None
        ctx = CommandContext(agent_config={}, current_agent=current_agent)
        with (
            patch("agentica.memory.session_log.SessionLog.list_sessions", return_value=self._sessions()),
            patch(
                "agentica.memory.session_log.SessionLog.session_preview",
                return_value={"user_count": 0, "first_user": None},
            ),
            patch("agentica.cli.commands.session.get_console") as mock_console,
        ):
            console = MagicMock()
            mock_console.return_value = console
            cli_session._cmd_resume(ctx, "")

        printed = "\n".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        current_line = next(line for line in printed.splitlines() if "sess-act" in line and "current" in line)
        self.assertIn("(current)", current_line)

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


if __name__ == "__main__":
    unittest.main()
