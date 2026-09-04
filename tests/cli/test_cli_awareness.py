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
from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.registry import COMMAND_REGISTRY
from agentica.cli.commands import model_config as cli_model_config
from agentica.cli.commands import runtime as cli_runtime_commands
from agentica.cli.commands import tools_skills as cli_tools_skills
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



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
                self._session_log = None

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
        self.assertIn("Subagent types: code, explore, research", ctx)
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
        self.assertIn("/status", COMMAND_REGISTRY)
        self.assertIs(COMMAND_REGISTRY["/status"][0], cli_model_config._cmd_status)

    def test_cmd_agents_registered(self):
        self.assertIn("/agents", COMMAND_REGISTRY)
        self.assertIn("/agent", COMMAND_REGISTRY)
        self.assertIs(COMMAND_REGISTRY["/agents"][0], cli_tools_skills._cmd_agents)
        self.assertIs(COMMAND_REGISTRY["/agent"][0], cli_tools_skills._cmd_agents)

    def test_cmd_worktree_registered(self):
        from agentica.cli.commands.worktree_cmd import _cmd_worktree

        self.assertIn("/worktree", COMMAND_REGISTRY)
        self.assertIs(COMMAND_REGISTRY["/worktree"][0], _cmd_worktree)

    def test_cmd_ps_registered(self):
        self.assertIn("/ps", COMMAND_REGISTRY)
        self.assertIs(COMMAND_REGISTRY["/ps"][0], cli_runtime_commands._cmd_ps)

    def test_cmd_ps_and_stop_control_background_terminals(self):
        import shlex

        from agentica.tools.background_processes import BackgroundProcessRegistry

        with tempfile.TemporaryDirectory() as td:
            agentica_home = Path(td) / "agentica-home"
            with patch.dict(os.environ, {"AGENTICA_HOME": str(agentica_home)}, clear=False):
                registry = BackgroundProcessRegistry(user_id="alice@example.com")
                command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
                item = registry.start(command, cwd=td)
                ctx = CommandContext(
                    agent_config={},
                    current_agent=None,
                    background_processes=registry,
                )
                fake_console = MagicMock()
                try:
                    projects_dir = Path(
                        os.environ.get(
                            "AGENTICA_PROJECTS_DIR",
                            str(agentica_home / "projects"),
                        )
                    )
                    self.assertNotIn(str(Path(td) / ".agentica"), item.log_path)
                    self.assertIn(str(projects_dir / "alice@example.com"), item.log_path)
                    self.assertNotIn(str(projects_dir / "default"), item.log_path)
                    with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
                        cli_runtime_commands._cmd_ps(ctx, "")
                        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list)
                        self.assertIn("Background terminals (1)", rendered)
                        self.assertIn(f"pid={item.pid}", rendered)

                        cli_runtime_commands._cmd_stop(ctx, str(item.pid))
                        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list)
                        self.assertIn("Stopped 1 terminal(s).", rendered)
                    self.assertEqual(registry.running_count(), 0)
                finally:
                    registry.stop()

    def test_cmd_ps_shows_full_background_command(self):
        """/ps must show the complete command, not a 90-char preview."""
        import shlex

        from agentica.tools.background_processes import BackgroundProcessRegistry

        long_tail = "QIDS=$(paste -sd, qids.txt) && python -m personamem.run " + " ".join(
            f"--flag-{i} value-{i}" for i in range(20)
        )
        with tempfile.TemporaryDirectory() as td:
            agentica_home = Path(td) / "agentica-home"
            with patch.dict(os.environ, {"AGENTICA_HOME": str(agentica_home)}, clear=False):
                registry = BackgroundProcessRegistry(user_id="ps-full-cmd")
                command = (
                    f"cd {shlex.quote(td)} && {long_tail} && "
                    f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
                )
                item = registry.start(command, cwd=td)
                ctx = CommandContext(
                    agent_config={},
                    current_agent=None,
                    background_processes=registry,
                )
                fake_console = MagicMock()
                try:
                    with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
                        cli_runtime_commands._cmd_ps(ctx, "")
                    rendered = "\n".join(
                        str(call.args[0]) for call in fake_console.print.call_args_list if call.args
                    )
                    self.assertIn(command, rendered)
                    self.assertNotIn("pas...", rendered)
                    self.assertIn("--flag-19 value-19", rendered)
                finally:
                    registry.stop()

    def test_cmd_ps_shows_a_delegated_session_by_its_task(self):
        """A delegated worker's command line is a `--query <whole task>`; the
        user needs to see which task is running, not that shell line."""
        import shlex

        from agentica.tools.background_processes import BackgroundProcessRegistry

        with tempfile.TemporaryDirectory() as td:
            registry = BackgroundProcessRegistry(user_id="ps-delegate")
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
            registry.start(command, cwd=td, kind="delegate", label="upgrade service-a")
            ctx = CommandContext(
                agent_config={},
                current_agent=None,
                background_processes=registry,
            )
            fake_console = MagicMock()
            try:
                with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
                    cli_runtime_commands._cmd_ps(ctx, "")
                rendered = "\n".join(
                    str(call.args[0]) for call in fake_console.print.call_args_list if call.args
                )
                self.assertIn("delegated session", rendered)
                self.assertIn("upgrade service-a", rendered)
                self.assertNotIn("time.sleep(30)", rendered)
            finally:
                registry.stop()

    def _stop_ctx(self, registry):
        """CommandContext with one background terminal and one background agent."""
        bg_agent = MagicMock()
        ctx = CommandContext(
            agent_config={},
            current_agent=None,
            background_processes=registry,
            bg_tasks={"bg_1": {"thread": None, "agent": bg_agent, "prompt": "audit deps", "num": 1}},
        )
        return ctx, bg_agent

    def test_cmd_stop_without_target_stops_nothing(self):
        """A bare /stop is usage output, never a bulk kill.

        It is one keystroke away from `/stop <id>` and arrives while unrelated
        work is running, so "stop everything" must be typed on purpose.
        """
        import shlex

        from agentica.tools.background_processes import BackgroundProcessRegistry

        with tempfile.TemporaryDirectory() as td:
            registry = BackgroundProcessRegistry(user_id="stop-needs-target")
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
            item = registry.start(command, cwd=td)
            ctx, bg_agent = self._stop_ctx(registry)
            fake_console = MagicMock()
            try:
                with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
                    cli_runtime_commands._cmd_stop(ctx, "")
                rendered = "\n".join(
                    str(call.args[0]) for call in fake_console.print.call_args_list if call.args
                )
                self.assertIn("needs a target", rendered)
                self.assertNotIn("Stopped", rendered)
                # The running targets are listed so the user can copy an id,
                # and Ctrl+C is named as the way to stop the current run.
                self.assertIn(item.id, rendered)
                self.assertIn("Ctrl+C", rendered)
                self.assertEqual(registry.running_count(), 1)
                bg_agent.cancel.assert_not_called()
            finally:
                registry.stop()

    def test_cmd_stop_all_stops_terminals_and_agent_tasks(self):
        """`/stop all` is the explicit bulk form the bare command used to be."""
        import shlex

        from agentica.tools.background_processes import BackgroundProcessRegistry

        with tempfile.TemporaryDirectory() as td:
            registry = BackgroundProcessRegistry(user_id="stop-all")
            command = f"{shlex.quote(sys.executable)} -c {shlex.quote('import time; time.sleep(30)')}"
            registry.start(command, cwd=td)
            ctx, bg_agent = self._stop_ctx(registry)
            fake_console = MagicMock()
            try:
                with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
                    cli_runtime_commands._cmd_stop(ctx, "all")
                rendered = "\n".join(
                    str(call.args[0]) for call in fake_console.print.call_args_list if call.args
                )
                self.assertIn("Stopped 1 terminal(s), 1 agent task(s).", rendered)
                self.assertEqual(registry.running_count(), 0)
                bg_agent.cancel.assert_called_once_with()
            finally:
                registry.stop()

    def test_cmd_stop_targets_one_agent_task_only(self):
        """`/stop #n` must not spill over to the other background agent."""
        first, second = MagicMock(), MagicMock()
        ctx = CommandContext(
            agent_config={},
            current_agent=None,
            bg_tasks={
                "bg_1": {"thread": None, "agent": first, "prompt": "a", "num": 1},
                "bg_2": {"thread": None, "agent": second, "prompt": "b", "num": 2},
            },
        )
        fake_console = MagicMock()
        with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
            cli_runtime_commands._cmd_stop(ctx, "#2")
        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list if call.args)
        self.assertIn("Stopped 1 agent task(s).", rendered)
        first.cancel.assert_not_called()
        second.cancel.assert_called_once_with()

    def test_cmd_stop_with_nothing_running_points_at_ctrl_c(self):
        """With no background work, /stop explains it is not the run's stop key."""
        ctx = CommandContext(agent_config={}, current_agent=None)
        fake_console = MagicMock()
        with patch.object(cli_runtime_commands, "get_console", return_value=fake_console):
            cli_runtime_commands._cmd_stop(ctx, "")
        rendered = "\n".join(str(call.args[0]) for call in fake_console.print.call_args_list if call.args)
        self.assertIn("No active background tasks.", rendered)
        self.assertIn("Ctrl+C interrupts the current run", rendered)

    def _make_apply_profile_ctx(self):
        """Build a CommandContext whose mock agent survives a profile switch.

        Pre-state carries a zhipuai auxiliary so a profile switch to a different auxiliary
        (or to no auxiliary) is observable in agent_config and environment_context.
        """
        mock_agent = MagicMock()
        mock_agent.tools = []
        mock_agent.working_memory.runs = []
        return CommandContext(
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
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model", return_value=MagicMock()),
                patch.object(cli_model_config, "_build_sibling_model", return_value=MagicMock()) as mock_auxiliary,
                # _apply_profile persists a project-scoped override via
                # set_project_profile(work_dir, name); work_dir=None here falls
                # back to os.getcwd(), which would otherwise leak a real
                # ~/.agentica/projects/<repo>/project.json on every test run.
                patch.object(cli_model_config, "set_project_profile"),
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
                cli_model_config._apply_profile(ctx, "withaux")

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
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model") as mock_get_model,
                patch.object(cli_model_config, "_build_sibling_model", return_value=MagicMock()),
                patch.object(cli_model_config, "set_project_profile"),
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
                cli_model_config._apply_profile(ctx, "hy3")

        self.assertEqual(ctx.agent_config["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "high"}})
        self.assertEqual(ctx.agent_config["auxiliary_extra_body"], {"aux": True})
        # get_model (main model rebuild) received the profile's extra_body.
        _args, kw = mock_get_model.call_args
        self.assertEqual(kw["extra_body"], {"chat_template_kwargs": {"reasoning_effort": "high"}})

    def test_apply_profile_switches_wire_api(self):
        from agentica import global_config as gc

        ctx = self._make_apply_profile_ctx()
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model") as mock_get_model,
                patch.object(cli_model_config, "set_project_profile"),
            ):
                gc.upsert_profile(
                    "a",
                    {
                        "model_provider": "openai",
                        "model_name": "gpt-5.6-sol",
                        "base_url": "https://example/v1",
                        "api_key": "sk-main",
                        "wire_api": "responses",
                        "reasoning": "high",
                    },
                    make_active=True,
                )
                cli_model_config._apply_profile(ctx, "a")

        self.assertEqual(ctx.agent_config["wire_api"], "responses")
        self.assertEqual(ctx.agent_config["reasoning"], "high")
        _args, kwargs = mock_get_model.call_args
        self.assertEqual(kwargs["wire_api"], "responses")
        self.assertEqual(kwargs["reasoning"], "high")

    def test_apply_profile_without_auxiliary_clears(self):
        """Switching to a profile without an auxiliary_model block clears the auxiliary fields."""
        from agentica import global_config as gc

        ctx = self._make_apply_profile_ctx()
        self.assertIsNotNone(ctx.agent_config["auxiliary_model_name"])

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = os.path.join(tmp, "config.yaml")
            with (
                patch("agentica.global_config.global_config_path", return_value=cfg_path),
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model", return_value=MagicMock()),
                patch.object(cli_model_config, "_build_sibling_model") as mock_sibling,
                # See test_apply_profile_switches_auxiliary_model above: avoid
                # leaking a real project-profile file to ~/.agentica/projects/.
                patch.object(cli_model_config, "set_project_profile"),
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
                cli_model_config._apply_profile(ctx, "noaux")

        self.assertIsNone(ctx.agent_config["auxiliary_model_name"])
        self.assertIsNone(ctx.agent_config["auxiliary_model_provider"])
        self.assertIsNone(ctx.agent_config["auxiliary_model"])
        mock_sibling.assert_not_called()
        self.assertNotIn("Auxiliary model:", ctx.current_agent.environment_context)

    def test_cmd_tools_add_from_rejects_path_traversal(self):
        """/tools add-from ../evil is rejected before any module is loaded."""
        ctx = CommandContext(
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
            patch.object(cli_tools_skills, "get_console", return_value=mock_console),
            patch.object(cli_tools_skills, "_load_custom_tool_module") as mock_load,
        ):
            cli_tools_skills._cmd_tools(ctx, "add-from ../evil")

        printed = "\n".join(str(c) for c in mock_console.print.call_args_list)
        self.assertIn("Invalid tool name", printed)
        self.assertIn("../evil", printed)
        mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
