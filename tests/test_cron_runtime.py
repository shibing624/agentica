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


if __name__ == "__main__":
    unittest.main()