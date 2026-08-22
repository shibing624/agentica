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

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.registry import COMMAND_REGISTRY
from agentica.cli.commands import tools_skills as cli_tools_skills
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



class TestCmdPermissions(unittest.TestCase):
    """`/permissions` reads/writes the Agent's own permission_mode directly —
    no separate PermissionManager object anymore."""

    def _make_ctx(self, agent):
        return CommandContext(
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
        with patch.object(cli_tools_skills, "get_console", return_value=MagicMock()):
            cli_tools_skills._cmd_permissions(ctx, "ask")

        self.assertEqual(agent.tool_config.permission_mode, "ask")

    def test_set_invalid_mode_does_not_mutate_agent(self):
        from agentica.agent import Agent

        agent = Agent()
        ctx = self._make_ctx(agent)
        with patch.object(cli_tools_skills, "get_console", return_value=MagicMock()):
            cli_tools_skills._cmd_permissions(ctx, "strict")

        self.assertEqual(agent.tool_config.permission_mode, "allow-all")

    def test_no_args_prints_current_mode_without_error(self):
        from agentica.agent import Agent
        from agentica.agent.config import ToolConfig

        agent = Agent(tool_config=ToolConfig(permission_mode="auto"))
        ctx = self._make_ctx(agent)
        console = MagicMock()
        with patch.object(cli_tools_skills, "get_console", return_value=console):
            cli_tools_skills._cmd_permissions(ctx, "")

        self.assertTrue(console.print.called)
        printed = " ".join(str(c) for c in console.print.call_args_list)
        self.assertIn("Ask for approval", printed)
        self.assertIn("Approve for me", printed)
        self.assertIn("Full Access", printed)
        self.assertNotIn("only read-only tools", printed)
        self.assertNotIn("request_path_access", printed)

    def test_yolo_command_removed_from_registry(self):
        self.assertNotIn("/yolo", COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
