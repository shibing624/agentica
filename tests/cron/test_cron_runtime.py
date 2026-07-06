# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the CLI cron runtime: settings, scheduler loop, runner,
the /cron slash command, and the standalone daemon entry.
"""
import os
import tempfile
import threading
import time
import unittest


def _isolate():
    home = tempfile.mkdtemp()
    os.environ["AGENTICA_HOME"] = home
    os.environ["AGENTICA_CRON_DIR"] = os.path.join(home, "cron")
    os.environ["AGENTICA_DOTENV_PATH"] = os.path.join(home, ".env")
    import importlib
    import agentica.config as cfg
    importlib.reload(cfg)
    import agentica.global_config as gc
    importlib.reload(gc)
    import agentica.cron.jobs as cronjobs
    importlib.reload(cronjobs)
    return cfg, gc, cronjobs


class TestCronSettings(unittest.TestCase):
    def test_setting_round_trip(self):
        _, gc, _ = _isolate()
        self.assertFalse(bool(gc.get_setting("cron.enabled", False)))
        gc.set_setting("cron.enabled", True)
        self.assertTrue(gc.get_setting("cron.enabled"))
        gc.set_setting("cron.interval", 30)
        self.assertEqual(gc.get_setting("cron.interval"), 30)
        gc.set_setting("cron.enabled", None)
        self.assertEqual(gc.get_setting("cron.enabled", "x"), "x")


class TestSchedulerLoop(unittest.TestCase):
    def test_loop_ticks_and_stops(self):
        import agentica.cron.cli_runner as m

        calls = {"n": 0}

        async def fake_tick(agent_runner, verbose=False):
            calls["n"] += 1

        orig = m.tick
        m.tick = fake_tick
        try:
            class StubRunner:
                async def run(self, prompt, context=None):
                    return "ok"

            t, ev = m.start_cron_thread(StubRunner(), interval=1)
            time.sleep(2.2)
            ev.set()
            t.join(timeout=3)
            self.assertGreaterEqual(calls["n"], 1)
            self.assertFalse(t.is_alive())
        finally:
            m.tick = orig


class TestCronCommand(unittest.TestCase):
    def setUp(self):
        _, _, self.cronjobs = _isolate()
        self._running = False

        class Ctx:
            agent_config = {"model_provider": "openai", "model_name": "gpt-4o",
                            "api_key": "sk-test", "base_url": None}
            extra_tools = []
            workspace = None
            skills_registry = None
            # No interactive channel / agent in tests -> /cron add skips the
            # LLM refine + confirm flow and creates the job with the raw prompt.
            current_agent = None
            ask_user_question_callback = None

        self.ctx = Ctx()
        self.ctx.tui_state = {
            "cron_is_running": lambda: self._running,
            "cron_start": lambda: True,
            "cron_stop": lambda: None,
        }

    def test_add_list_pause_resume(self):
        from agentica.cli import commands
        commands._cmd_cron(self.ctx, 'add "do a thing" 0 9 * * *')
        jobs = self.cronjobs.list_jobs()
        self.assertEqual(len(jobs), 1)
        jid = jobs[0].id
        commands._cmd_cron(self.ctx, f"pause {jid}")
        commands._cmd_cron(self.ctx, f"resume {jid}")
        # list + daemon status + unknown should not raise
        commands._cmd_cron(self.ctx, "")
        commands._cmd_cron(self.ctx, "daemon status")
        commands._cmd_cron(self.ctx, "bogus")

    def test_registered(self):
        from agentica.cli import commands
        self.assertIn("/cron", commands.COMMAND_REGISTRY)

    def test_daemon_status_reports_persisted_config(self):
        """`/cron daemon status` must surface the persisted cron.enabled config,
        not only the current session's thread — otherwise a daemon enabled in
        config (or run by a separate process) looks like it has no status."""
        import io
        from unittest.mock import patch
        from rich.console import Console
        from agentica.cli import commands
        from agentica.global_config import set_setting

        set_setting("cron.enabled", True)
        set_setting("cron.interval", 45)
        self._running = False  # no live thread in this session

        buf = io.StringIO()
        with patch.object(commands, "get_console", return_value=Console(file=buf, width=200)):
            commands._cmd_cron(self.ctx, "daemon status")
        out = buf.getvalue()
        self.assertIn("config", out)
        self.assertIn("cron.enabled=", out)
        self.assertIn("45s", out)
        # Enabled in config but no live thread -> must explain (proves the
        # persisted True was read), not stay silent.
        self.assertIn("Enabled in config but no scheduler thread", out)

    def test_edit_prompt_and_schedule(self):
        from agentica.cli import commands
        commands._cmd_cron(self.ctx, 'add "do a thing" 0 9 * * *')
        jid = self.cronjobs.list_jobs()[0].id

        commands._cmd_cron(self.ctx, f'edit {jid} prompt "a much better thing"')
        job = self.cronjobs.get_job(jid)
        self.assertEqual(job.prompt, "a much better thing")
        self.assertEqual(job.name, "a much better thing")

        commands._cmd_cron(self.ctx, f"edit {jid} schedule every 5m")
        job = self.cronjobs.get_job(jid)
        from agentica.cron.types import EverySchedule
        self.assertIsInstance(job.schedule, EverySchedule)
        self.assertEqual(job.schedule.interval_ms, 5 * 60 * 1000)

    def test_edit_bad_usage_does_not_raise(self):
        from agentica.cli import commands
        commands._cmd_cron(self.ctx, 'add "do a thing" 0 9 * * *')
        jid = self.cronjobs.list_jobs()[0].id
        # Missing field/value, and unknown job id — must print, not raise.
        commands._cmd_cron(self.ctx, f"edit {jid}")
        commands._cmd_cron(self.ctx, "edit nope prompt hi")

    def test_remove_uses_tui_confirm_not_pt_prompt(self):
        """Regression: /cron remove must confirm via the ask_user_question_callback
        (TUI-safe), never a nested pt_prompt that deadlocks the bg thread."""
        from agentica.cli import commands
        commands._cmd_cron(self.ctx, 'add "do a thing" 0 9 * * *')
        jid = self.cronjobs.list_jobs()[0].id

        # 'no' keeps the job.
        self.ctx.ask_user_question_callback = lambda prompt, options=None: "no"
        commands._cmd_cron(self.ctx, f"remove {jid}")
        self.assertEqual(len(self.cronjobs.list_jobs()), 1)

        # 'yes' removes it.
        self.ctx.ask_user_question_callback = lambda prompt, options=None: "yes"
        commands._cmd_cron(self.ctx, f"remove {jid}")
        self.assertEqual(len(self.cronjobs.list_jobs()), 0)

    def test_confirm_via_tui_safe_default_without_callback(self):
        from agentica.cli import commands
        self.ctx.ask_user_question_callback = None
        self.assertFalse(commands._confirm_via_tui(self.ctx, "Delete?"))

    def test_add_refines_prompt_and_stores_recommended(self):
        """Add-time flow: refine the rough prompt with the model and, when the
        user picks the recommended option, store the refined prompt while
        keeping their original words as the display name."""
        from agentica.cli import commands

        class _FakeResp:
            content = "Touch tmp/<ISO-timestamp>.txt in one run."

        class _FakeModel:
            async def response(self, messages):
                return _FakeResp()

        class _FakeAgent:
            def resolve_auxiliary_model(self, task="default"):
                return _FakeModel()

        self.ctx.current_agent = _FakeAgent()
        # Callback picks the first (recommended = refined) option.
        self.ctx.ask_user_question_callback = lambda prompt, options=None: (
            options[0] if options else "")

        commands._cmd_cron(self.ctx, 'add "每分钟在tmp下touch时间戳txt" "every 1m"')
        jobs = self.cronjobs.list_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].prompt, "Touch tmp/<ISO-timestamp>.txt in one run.")
        self.assertEqual(jobs[0].name, "每分钟在tmp下touch时间戳txt")

    def test_add_keeps_original_when_user_declines_refine(self):
        from agentica.cli import commands

        class _FakeResp:
            content = "some refined text"

        class _FakeModel:
            async def response(self, messages):
                return _FakeResp()

        class _FakeAgent:
            def resolve_auxiliary_model(self, task="default"):
                return _FakeModel()

        self.ctx.current_agent = _FakeAgent()
        # Pick the "Keep the original prompt" option (2nd).
        self.ctx.ask_user_question_callback = lambda prompt, options=None: (
            options[1] if options and len(options) > 1 else "")

        commands._cmd_cron(self.ctx, 'add "raw words" "every 1m"')
        jobs = self.cronjobs.list_jobs()
        self.assertEqual(jobs[0].prompt, "raw words")

    def test_execute_job_signature(self):
        """Regression: _execute_job requires positional `verbose`; ensure calling
        it the way /cron run does works end-to-end with a stub runner (offline)."""
        import asyncio
        from agentica.cron.scheduler import _execute_job

        commands_jid = self.cronjobs.create_job(prompt="say hi", schedule="0 9 * * *").id

        class StubRunner:
            async def run(self, prompt, context=None):
                return "hi"

        job = self.cronjobs.get_job(commands_jid)
        # Must not raise (TypeError on missing verbose would mean a real bug).
        result = asyncio.run(_execute_job(job, agent_runner=StubRunner(), verbose=False))
        self.assertIsInstance(result, dict)

    def test_runs_reads_real_fields(self):
        """Regression: /cron runs must read task_id/started_at_ms/RunStatus.OK,
        not the wrong job_id/started_at/'success' names it once used."""
        from agentica.cli import commands
        from agentica.cron.types import RunStatus
        commands._cmd_cron(self.ctx, 'add "do a thing" 0 9 * * *')
        jid = self.cronjobs.list_jobs()[0].id
        # Record a successful run, then list runs.
        self.cronjobs.mark_job_run(jid, status=RunStatus.OK, result="done")
        runs = self.cronjobs.list_task_runs(job_id=jid)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].task_id, jid)
        self.assertEqual(runs[0].status, RunStatus.OK)
        self.assertTrue(runs[0].started_at_ms > 0)
        # The command itself must not raise on real records.
        commands._cmd_cron(self.ctx, "runs")
        commands._cmd_cron(self.ctx, f"runs {jid}")


class TestCronInCliTools(unittest.TestCase):
    def test_cli_agent_has_cron_tool(self):
        import sys
        sys.argv = ["agentica"]
        from agentica.cli.config import create_agent, parse_args
        cfg = vars(parse_args())
        cfg.update({"model_provider": "openai", "model_name": "gpt-4o",
                    "api_key": "sk-test", "base_url": None})
        agent = create_agent(cfg, extra_tools=[], workspace=None, skills_registry=None)
        names = [type(t).__name__ for t in agent.tools]
        self.assertIn("CronTool", names)
        self.assertIn("SelfManageTool", names)

    def test_cli_agent_exposes_cron_functions_to_model(self):
        """Regression: CronTool/SelfManageTool must be real Tool subclasses so
        ``model.add_tool`` registers their functions. Previously they were plain
        objects with a ``functions`` list, so ``cronjob``/``self_manage`` were
        silently dropped and the agent could not manage cron via natural language."""
        import sys
        sys.argv = ["agentica"]
        from agentica.cli.config import create_agent, parse_args
        cfg = vars(parse_args())
        cfg.update({"model_provider": "openai", "model_name": "gpt-4o",
                    "api_key": "sk-test", "base_url": None})
        agent = create_agent(cfg, extra_tools=[], workspace=None, skills_registry=None)
        agent.update_model()
        fns = agent.model.functions or {}
        self.assertIn("cronjob", fns)
        self.assertIn("self_manage", fns)

    def test_cron_agent_ask_tool_is_noninteractive(self):
        """Regression: a cron job runs unattended on a background scheduler
        thread, so its ``ask_user_question`` tool must NOT fall back to a bare
        ``input()`` (which blocks forever / deadlocks prompt_toolkit). The cron
        factory wires a non-interactive callback that returns immediately."""
        import sys
        sys.argv = ["agentica"]
        from agentica.cli.config import parse_args
        from agentica.cron.cli_runner import build_cli_agent_factory
        from agentica.tools.ask_user_question_tool import AskUserQuestionTool

        cfg = vars(parse_args())
        cfg.update({"model_provider": "openai", "model_name": "gpt-4o",
                    "api_key": "sk-test", "base_url": None})
        factory = build_cli_agent_factory(
            cfg, extra_tools=[], workspace=None, skills_registry=None)
        agent = factory()

        ask_tools = [t for t in agent.tools if isinstance(t, AskUserQuestionTool)]
        self.assertTrue(ask_tools, "cron agent must include the ask_user_question tool")
        cb = ask_tools[0].input_callback
        self.assertIsNotNone(
            cb, "cron agent's ask tool must have a non-interactive callback, not bare input()")
        # The callback must return immediately (never block on stdin).
        result = cb("choose an implementation approach", ["A", "B", "C"])
        self.assertIn("non-interactive", result.lower())


if __name__ == "__main__":
    unittest.main()