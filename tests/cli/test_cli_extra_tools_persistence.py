# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: `/tools add` / `/tools remove` remember their set for the next CLI
in the same work_dir; `add-from` deliberately does not.
"""

import os
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.agent import Agent
from agentica.cli.commands import tools_skills as cli_tools_skills
from agentica.cli.commands.context import CommandContext
from agentica.memory.session_log import SessionLog
from agentica.project_store import project_base_dir, read_project_file
from test_cli_prefs import _WorkDir


class _CtxBuilder:
    """A CommandContext whose extra-tool state mirrors the real dispatch loop."""

    def __init__(self, work_dir, session_id="s-tools"):
        self.work_dir = work_dir
        self.tmp_log = SessionLog(session_id, work_dir=work_dir, user_id="default")
        self.agent = Agent()
        self.agent._session_log = self.tmp_log
        self.agent.work_dir = work_dir
        self.ctx = CommandContext(
            agent_config={"work_dir": work_dir, "model_provider": "openai", "model_name": "gpt-4o"},
            current_agent=self.agent,
            extra_tools=[],
            extra_tool_names=[],
            tui_state={},
        )

    def saved(self):
        return read_project_file(project_base_dir(self.work_dir)).get("cli", {}).get("extra_tools")

    def sidecar(self):
        return self.tmp_log.get_cli_prefs()


class TestExtraToolsPersistence(unittest.TestCase):
    def setUp(self):
        self.work = _WorkDir(self)

    def _ctx(self):
        return _CtxBuilder(self.work.path)

    def test_tools_add_records_the_registry_name(self):
        builder = self._ctx()

        with (
            patch.object(cli_tools_skills, "get_console", return_value=MagicMock()),
            patch.object(cli_tools_skills, "configure_tools", return_value=[MagicMock()]),
        ):
            result = cli_tools_skills._cmd_tools(builder.ctx, "add search_serper")

        self.assertEqual(builder.saved(), ["search_serper"])
        self.assertEqual(builder.sidecar(), {"extra_tools": ["search_serper"]})
        self.assertEqual(result["extra_tool_names"], ["search_serper"])

    def test_unknown_tool_name_is_not_recorded(self):
        builder = self._ctx()

        with patch.object(cli_tools_skills, "get_console", return_value=MagicMock()):
            cli_tools_skills._cmd_tools(builder.ctx, "add not_a_real_tool")

        self.assertIsNone(builder.saved())

    def test_remove_drops_the_name_from_the_saved_set(self):
        """`/tools remove` writes the new set, so the next launch does not
        resurrect the tool the user just dropped."""
        builder = self._ctx()
        builder.ctx.extra_tool_names = ["search_serper"]
        builder.agent.tools = []

        with patch.object(cli_tools_skills, "get_console", return_value=MagicMock()):
            cli_tools_skills._cmd_tools(builder.ctx, "add search_serper")
            cli_tools_skills._cmd_tools(builder.ctx, "remove search_serper")

        self.assertIsNone(builder.saved())
        self.assertIsNone(builder.sidecar().get("extra_tools"))

    def test_add_from_is_never_recorded(self):
        """Its module runs arbitrary top-level code and needs a human confirm —
        replaying it at startup would remove the human."""
        builder = self._ctx()
        tool_dir = os.path.join(self.work.path, ".agentica", "tools")
        os.makedirs(tool_dir, exist_ok=True)
        with open(os.path.join(tool_dir, "mine.py"), "w", encoding="utf-8") as fh:
            fh.write("tool = None\n")

        loaded = MagicMock()
        with (
            patch.object(cli_tools_skills, "get_console", return_value=MagicMock()),
            patch.object(cli_tools_skills, "_confirm_via_tui", return_value=True),
            patch.object(cli_tools_skills, "_load_custom_tool_module", return_value=loaded),
        ):
            cli_tools_skills._cmd_tools(builder.ctx, "add-from mine")

        self.assertIsNone(builder.saved())
        self.assertIsNone(builder.sidecar().get("extra_tools"))


class TestStartupRestoresExtraTools(unittest.TestCase):
    def setUp(self):
        self.work = _WorkDir(self)

    def _run_main_with(self, argv):
        import importlib

        # ``agentica.cli.main`` resolves to the *function* re-exported by
        # ``agentica/cli/__init__.py``, so the module must be fetched by name.
        cli_main = importlib.import_module("agentica.cli.main")

        console = MagicMock()
        with (
            patch.object(sys, "argv", argv),
            patch.object(cli_main, "_enable_cli_file_logging"),
            patch.object(cli_main, "refresh_model_catalog_in_background"),
            patch.object(
                cli_main,
                "resolve_model_config",
                return_value={"model_provider": "openai", "model_name": "gpt-4o", "base_url": None},
            ),
            patch.object(cli_main, "get_console", return_value=console),
            # `main()` imports this locally in the interactive branch, so the
            # patch has to land on the module that defines it.
            patch("agentica.cli.interactive.run_interactive") as run_interactive,
        ):
            cli_main.main()
        return run_interactive

    def test_saved_tools_are_loaded_without_a_flag(self):
        from agentica.cli.prefs import write_project_prefs

        write_project_prefs(self.work.path, {"extra_tools": ["search_serper", "weather"]})

        run_interactive = self._run_main_with(
            ["agentica", "--work_dir", self.work.path, "--no-workspace", "--no-experience"]
        )

        names = run_interactive.call_args.args[1]
        self.assertEqual(names, ["search_serper", "weather"])

    def test_flag_and_saved_set_are_unioned_without_duplicates(self):
        from agentica.cli.prefs import write_project_prefs

        write_project_prefs(self.work.path, {"extra_tools": ["weather"]})

        run_interactive = self._run_main_with(
            [
                "agentica",
                "--work_dir", self.work.path,
                "--no-workspace",
                "--no-experience",
                "--tools", "weather", "search_serper",
            ]
        )

        names = run_interactive.call_args.args[1]
        self.assertEqual(names, ["weather", "search_serper"])


if __name__ == "__main__":
    unittest.main()
