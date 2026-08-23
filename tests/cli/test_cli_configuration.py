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
from agentica.cli.commands.context import CommandContext, PendingQueue
from agentica.cli.commands import model_config as cli_model_config
from agentica.cli.commands import runtime as cli_runtime_commands
from agentica.cli.commands import helpers as cli_helpers
from agentica.cli.commands import tools_skills as cli_tools_skills
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



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

        ctx = CommandContext(
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
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model", return_value=MagicMock()),
                patch.object(cli_model_config, "create_agent", return_value=MagicMock()),
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
                cli_model_config._cmd_model(ctx, "openai/gpt-5")
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

        ctx = CommandContext(
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
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model", return_value=MagicMock()) as mock_get_model,
                patch.object(cli_model_config, "create_agent", return_value=MagicMock()),
                # /model <profile> -> _apply_profile -> set_project_profile(work_dir, name);
                # work_dir=None falls back to os.getcwd(), which would otherwise leak a
                # real ~/.agentica/projects/<repo>/project.json on every test run.
                patch.object(cli_model_config, "set_project_profile"),
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
                cli_model_config._cmd_model(ctx, "opus")
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

        ctx = CommandContext(
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
                patch.object(cli_model_config, "get_console", return_value=MagicMock()),
                patch.object(cli_model_config, "get_model", return_value=MagicMock()),
                patch.object(cli_model_config, "create_agent", return_value=MagicMock()),
            ):
                gc.upsert_profile(
                    "proxy",
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

                cli_model_config._cmd_model(ctx, "ark")

                # Mirrors interactive.py's _apply_command_result exactly.
                fixed_name, fixed_source = gc.resolve_active_profile_name(
                    work_dir=ctx.agent_config.get("work_dir") or os.getcwd()
                )
                # Sanity: the pre-fix call (no os.getcwd() fallback) misses the
                # override and silently shows the stale global default instead.
                stale_name, stale_source = gc.resolve_active_profile_name(work_dir=ctx.agent_config.get("work_dir"))

        self.assertEqual((fixed_name, fixed_source), ("ark", "project"))
        self.assertEqual((stale_name, stale_source), ("proxy", "global"))

    def test_parse_goal_budget_flags(self):
        from agentica.cli.commands.goal import _parse_goal_set_args

        objective, budgets, err = _parse_goal_set_args("--turns 5 --tokens 80000 --wall 1800 修复 API")

        self.assertIsNone(err)
        self.assertEqual(objective, "修复 API")
        self.assertEqual(budgets["turn_budget"], 5)
        self.assertEqual(budgets["token_budget"], 80000)
        self.assertEqual(budgets["wall_clock_budget_sec"], 1800)

    def test_parse_goal_budget_minus_one_means_unlimited(self):
        from agentica.cli.commands.goal import _parse_goal_set_args

        objective, budgets, err = _parse_goal_set_args("--tokens=-1 长任务")
        self.assertIsNone(err)
        self.assertEqual(objective, "长任务")
        self.assertEqual(budgets["token_budget"], -1)

        _, budgets, err = _parse_goal_set_args("--turns -1 --wall -1 长任务")
        self.assertIsNone(err)
        self.assertEqual(budgets["turn_budget"], -1)
        self.assertEqual(budgets["wall_clock_budget_sec"], -1)

        for raw in ("--tokens 0 长任务", "--tokens -5 坏值"):
            _, _, err = _parse_goal_set_args(raw)
            self.assertIsNotNone(err, raw)

    def _steer_ctx(self, *, agent_running, steer_accepts, queue_items=()):
        agent = MagicMock()
        agent.steer.return_value = steer_accepts
        pending_queue = PendingQueue()
        for item in queue_items:
            pending_queue.put(item)
        ctx = CommandContext(
            agent_config={},
            current_agent=agent,
            agent_running=agent_running,
            pending_queue=pending_queue,
        )
        return ctx, agent, pending_queue

    def test_steer_injects_into_live_run(self):
        """Mid-run steering goes to the agent, not the queue."""
        ctx, agent, pending_queue = self._steer_ctx(agent_running=True, steer_accepts=True)

        with patch.object(cli_runtime_commands, "get_console", return_value=MagicMock()):
            cli_runtime_commands._cmd_steer(ctx, "keep the API compatible")

        agent.steer.assert_called_once_with("keep the API compatible")
        self.assertEqual(pending_queue.peek_all(), [])

    def test_steer_queues_when_run_ended_mid_dispatch(self):
        """TOCTOU: the run ends between the UI check and steer(); never drop the text.

        Under a standing goal this is the common case — the loop spends seconds
        judging a finished turn with the agent idle.
        """
        ctx, agent, pending_queue = self._steer_ctx(agent_running=True, steer_accepts=False)

        with patch.object(cli_runtime_commands, "get_console", return_value=MagicMock()):
            cli_runtime_commands._cmd_steer(ctx, "stop rewriting the tests")

        agent.steer.assert_called_once()
        self.assertEqual(pending_queue.peek_all(), ["stop rewriting the tests"])

    def test_steer_preempts_pending_goal_continuation(self):
        """A queued continuation is machine-generated — the correction goes first."""
        continuation = f"{CONTINUATION_PROMPT_PREFIX}\nGoal: ship it"
        ctx, _agent, pending_queue = self._steer_ctx(
            agent_running=False,
            steer_accepts=False,
            queue_items=[continuation],
        )

        with patch.object(cli_runtime_commands, "get_console", return_value=MagicMock()):
            cli_runtime_commands._cmd_steer(ctx, "don't touch the public API")

        self.assertEqual(pending_queue.peek_all(), ["don't touch the public API", continuation])

    def test_goal_loop_does_not_double_queue_continuation(self):
        """A /steer that jumped the queue must not spawn a second continuation.

        The interjected turn is still judged and still charged to the budget,
        but the continuation it cut in front of already covers the next step.
        """
        from agentica.cli.interactive import goal_hook as cli_goal_hook
        from agentica.cli.interactive.session_state import SessionState
        from agentica.goals import GoalDecision

        continuation = f"{CONTINUATION_PROMPT_PREFIX}\nGoal: ship it"
        pending_queue = PendingQueue()
        pending_queue.put(continuation)

        mgr = MagicMock()
        mgr.is_active.return_value = True
        mgr.extract_turn_signals.return_value = ("did the thing", 100, 100, [])
        decision = GoalDecision(
            status="active",
            should_continue=True,
            continuation_prompt=continuation,
            verdict="continue",
            reason="more to do",
            message="",
        )

        agent = MagicMock()
        agent._cancelled = False
        state = SessionState(current_agent=agent, goal_manager=mgr)

        with (
            patch.object(cli_goal_hook, "_run_async_safe", return_value=decision),
            patch.object(cli_goal_hook, "_cprint"),
        ):
            cli_goal_hook._maybe_continue_goal(state, pending_queue, {})

        self.assertEqual(pending_queue.peek_all(), [continuation])

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
        """CLI exposes DeepAgent self-evolution controls (no AGENTS.md compile)."""
        from agentica.cli.runtime import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "agentica",
                "--no-experience",
                "--enable-skill-upgrade",
                "--skill-upgrade-mode",
                "draft",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.no_experience)
        self.assertTrue(args.enable_skill_upgrade)
        self.assertEqual(args.skill_upgrade_mode, "draft")

    def test_parse_compression_flags(self):
        from agentica.cli.runtime import parse_args

        with patch.object(sys, "argv", ["agentica"]):
            args = parse_args()
        self.assertIsNone(args.evict)
        self.assertIsNone(args.auto_compact)

        with patch.object(sys, "argv", ["agentica", "--no-evict", "--no-auto-compact"]):
            args = parse_args()
        self.assertFalse(args.evict)
        self.assertFalse(args.auto_compact)

        with patch.object(sys, "argv", ["agentica", "--evict", "--auto-compact"]):
            args = parse_args()
        self.assertTrue(args.evict)
        self.assertTrue(args.auto_compact)

    def test_interactive_extensions_install_reports_replaced_symlinked_skill(self):
        """Interactive install prints when it replaces a symlinked skill."""
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
            patch.object(cli_tools_skills, "install_skills", side_effect=fake_install_skills),
            patch.object(cli_helpers, "reset_skill_registry"),
            patch.object(cli_helpers, "load_system_skills"),
            patch.object(cli_helpers, "get_skill_registry", return_value=refreshed_registry),
            patch.object(cli_helpers, "create_agent", return_value=MagicMock()),
            patch("agentica.cli.commands.tools_skills.Path") as MockPath,
            patch("agentica.cli.commands.tools_skills.get_console") as mock_get_console,
        ):
            # Make Path(source).expanduser().exists() return True so the local
            # install branch is taken instead of falling through to hub_install.
            mock_path_inst = MagicMock()
            mock_path_inst.expanduser.return_value.exists.return_value = True
            MockPath.return_value = mock_path_inst
            mock_console = MagicMock()
            mock_console.print = mock_print
            mock_get_console.return_value = mock_console
            cli_tools_skills._cmd_skills(ctx, cmd_args="install /tmp/mock-skill-repo --force")

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
        self.assertIn("learn-from-experience", agent.session_guidance[0])
        self.assertIn("Available skills", agent.session_guidance[0])

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
        self.assertIsNotNone(captured["experience_config"].skill_upgrade)
        self.assertEqual(captured["experience_config"].skill_upgrade.mode, "draft")

    def test_create_agent_applies_compression_switches(self):
        from agentica.cli.runtime import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
        ):
            create_agent(
                {
                    "model_provider": "zhipuai",
                    "model_name": "glm-5",
                    "debug": False,
                    "work_dir": None,
                    "enable_evict": False,
                    "enable_auto_compact": False,
                },
                extra_tools=[],
                workspace=None,
                skills_registry=None,
            )

        tc = captured["tool_config"]
        self.assertFalse(tc.enable_evict)
        self.assertFalse(tc.enable_auto_compact)
        self.assertTrue(tc.auto_load_mcp)

    def test_resolve_compression_flags_reads_settings_when_unset(self):
        from agentica.cli.runtime import _resolve_compression_flags

        with patch("agentica.cli.runtime.get_setting", side_effect=lambda key, default=None: {
            "enable_evict": False,
            "enable_auto_compact": False,
        }.get(key, default)):
            evict, auto = _resolve_compression_flags({})
        self.assertFalse(evict)
        self.assertFalse(auto)

        evict, auto = _resolve_compression_flags(
            {"enable_evict": True, "enable_auto_compact": False}
        )
        self.assertTrue(evict)
        self.assertFalse(auto)

    def test_resolve_compact_token_limit_profile_then_settings(self):
        from agentica.cli.runtime import _resolve_compact_token_limit

        self.assertIsNone(_resolve_compact_token_limit({}))
        self.assertEqual(_resolve_compact_token_limit({"compact_token_limit": 300_000}), 300_000)
        with patch("agentica.cli.runtime.get_setting", return_value=8000):
            self.assertEqual(_resolve_compact_token_limit({}), 8000)
            self.assertEqual(_resolve_compact_token_limit({"compact_token_limit": 300_000}), 300_000)

    def test_create_agent_wires_compact_token_limit(self):
        from agentica.cli.runtime import create_agent

        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
        ):
            create_agent(
                {
                    "model_provider": "zhipuai",
                    "model_name": "glm-5",
                    "debug": False,
                    "work_dir": None,
                    "compact_token_limit": 300_000,
                },
                extra_tools=[],
                workspace=None,
                skills_registry=None,
            )

        self.assertEqual(captured["tool_config"].compact_token_limit, 300_000)

    def test_create_agent_sets_background_registry_user_id(self):
        """Background command logs should use the current workspace user segment."""
        from agentica.cli.runtime import create_agent
        from agentica.tools.background_processes import BackgroundProcessRegistry

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                self.tools = []

        registry = BackgroundProcessRegistry()
        workspace = MagicMock()
        workspace.user_id = "alice@example.com"

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
                },
                extra_tools=[],
                workspace=workspace,
                skills_registry=None,
                background_process_registry=registry,
            )

        self.assertEqual(registry.user_id, "alice@example.com")

    def _tool_names_from_create_agent(self, **create_agent_kwargs):
        from agentica.cli.runtime import create_agent

        captured = {}

        from agentica.tools.builtin.delegate_tool import BuiltinDelegateTool
        from agentica.tools.background_processes import BackgroundProcessRegistry

        real_delegate = BuiltinDelegateTool(
            background_process_registry=BackgroundProcessRegistry(),
            permission_mode=lambda: "allow-all",
        )

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = [real_delegate]

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
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
                skills_registry=None,
                **create_agent_kwargs,
            )
        tools = captured["tools"] + list(getattr(agent, "tools", []) or [])
        return [tool.name for tool in tools]

    def test_an_interactive_session_can_delegate(self):
        from agentica.tools.background_processes import BackgroundProcessRegistry

        names = self._tool_names_from_create_agent(
            background_process_registry=BackgroundProcessRegistry()
        )

        self.assertIn("builtin_delegate_tool", names)

    def test_a_session_without_a_process_registry_cannot_delegate(self):
        """A one-shot `--query` run and a cron-spawned agent have no registry,
        so a worker they started could never be waited on or reported back."""
        names = self._tool_names_from_create_agent()

        self.assertNotIn("builtin_delegate_tool", names)

    def test_a_delegated_worker_cannot_delegate_further(self):
        from agentica.tools.background_processes import BackgroundProcessRegistry
        from agentica.tools.builtin.delegate_tool import DEPTH_ENV_VAR

        with patch.dict(os.environ, {DEPTH_ENV_VAR: "1"}, clear=False):
            names = self._tool_names_from_create_agent(
                background_process_registry=BackgroundProcessRegistry()
            )

        self.assertNotIn("builtin_delegate_tool", names)


class TestDebugToggle(unittest.TestCase):
    """`/debug` is a runtime switch for verbose logging, not a status dump.

    The logging helpers are patched in the command module's namespace so the
    test never mutates the process-wide handler list other tests rely on.
    """

    def _ctx(self, debug: bool):
        agent = Mock()
        agent.debug = debug
        return CommandContext(
            agent_config={"model_provider": "openai", "model_name": "gpt-4o", "debug": debug},
            current_agent=agent,
            tui_state={"debug": debug},
        )

    def _run(self, ctx, cmd_args=""):
        with patch.object(cli_model_config, "restore_console_logging") as restore, patch.object(
            cli_model_config, "suppress_console_logging"
        ) as suppress, patch.object(cli_model_config, "set_log_level_to_debug") as to_debug, patch.object(
            cli_model_config, "set_log_level_to_info"
        ) as to_info:
            cli_model_config._cmd_debug(ctx, cmd_args)
        return restore, suppress, to_debug, to_info

    def test_bare_debug_turns_verbose_logging_on(self):
        ctx = self._ctx(debug=False)
        restore, suppress, to_debug, to_info = self._run(ctx)

        self.assertTrue(ctx.agent_config["debug"])
        self.assertTrue(ctx.tui_state["debug"])
        self.assertTrue(ctx.current_agent.debug)
        restore.assert_called_once_with("DEBUG", color=False)
        to_debug.assert_called_once()
        suppress.assert_not_called()
        to_info.assert_not_called()

    def test_bare_debug_flips_back_off(self):
        ctx = self._ctx(debug=True)
        restore, suppress, to_debug, to_info = self._run(ctx)

        self.assertFalse(ctx.agent_config["debug"])
        self.assertFalse(ctx.tui_state["debug"])
        self.assertFalse(ctx.current_agent.debug)
        suppress.assert_called_once()
        to_info.assert_called_once()
        restore.assert_not_called()
        to_debug.assert_not_called()

    def test_explicit_on_is_idempotent(self):
        ctx = self._ctx(debug=True)
        restore, _suppress, to_debug, _to_info = self._run(ctx, "on")

        self.assertTrue(ctx.agent_config["debug"])
        restore.assert_called_once_with("DEBUG", color=False)
        to_debug.assert_called_once()

    def test_explicit_off(self):
        ctx = self._ctx(debug=True)
        _restore, suppress, _to_debug, to_info = self._run(ctx, "off")

        self.assertFalse(ctx.agent_config["debug"])
        suppress.assert_called_once()
        to_info.assert_called_once()

    def test_unknown_argument_changes_nothing(self):
        ctx = self._ctx(debug=False)
        restore, suppress, to_debug, to_info = self._run(ctx, "verbose")

        self.assertFalse(ctx.agent_config["debug"])
        self.assertFalse(ctx.tui_state["debug"])
        for mock in (restore, suppress, to_debug, to_info):
            mock.assert_not_called()

    def test_registered_as_a_toggle(self):
        from agentica.cli.commands.registry import COMMAND_REGISTRY

        handler, description = COMMAND_REGISTRY["/debug"]
        self.assertIs(handler, cli_model_config._cmd_debug)
        self.assertIn("Toggle", description)


if __name__ == "__main__":
    unittest.main()
