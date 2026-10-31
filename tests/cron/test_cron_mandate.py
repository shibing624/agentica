# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: A cron job runs under the tier and directory it was created
with (its mandate), not whatever the host session is in when it fires; a job
whose mandate cannot be honored, or that cannot go on unattended, pauses with
a reason instead of retrying.
"""
import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_cron_dir(tmp_path):
    cron_dir = tmp_path / "cron"
    (cron_dir / "output").mkdir(parents=True)
    with patch("agentica.cron.jobs.CRON_DIR", cron_dir), \
         patch("agentica.cron.jobs.JOBS_FILE", cron_dir / "jobs.json"), \
         patch("agentica.cron.jobs.OUTPUT_DIR", cron_dir / "output"), \
         patch("agentica.cron.jobs.RUNS_FILE", cron_dir / "runs.jsonl"):
        yield cron_dir


def _caller(mode: str, work_dir: str):
    return SimpleNamespace(tool_config=SimpleNamespace(permission_mode=mode), work_dir=work_dir)


class _RecordingRunner:
    def __init__(self):
        self.contexts = []

    async def run(self, prompt, context=None):
        self.contexts.append(context)
        return "done"


# ============== Capture at create ==============

class TestMandateCapture:
    def test_create_records_the_calling_sessions_tier_and_dir(self, tmp_cron_dir, tmp_path):
        from agentica.cron.jobs import get_job
        from agentica.tools.cron_tool import CronTool

        out = json.loads(CronTool().cronjob(
            action="create", prompt="p", schedule="1h", agent=_caller("auto", str(tmp_path))))
        job = get_job(out["job"]["job_id"])
        assert (job.permission_mode, job.work_dir) == ("auto", str(tmp_path))

    def test_model_cannot_ask_for_a_broader_tier_than_its_session(self, tmp_cron_dir, tmp_path):
        from agentica.cron.jobs import list_jobs
        from agentica.tools.cron_tool import CronTool

        out = json.loads(CronTool().cronjob(
            action="create", prompt="p", schedule="1h", permission_mode="allow-all",
            agent=_caller("ask", str(tmp_path))))
        assert out["success"] is False
        assert "broader" in out["error"]
        assert list_jobs(include_disabled=True) == []

    def test_model_may_make_a_job_stricter(self, tmp_cron_dir, tmp_path):
        from agentica.cron.jobs import get_job
        from agentica.tools.cron_tool import CronTool

        tool = CronTool()
        out = json.loads(tool.cronjob(
            action="create", prompt="p", schedule="1h", permission_mode="ask",
            agent=_caller("allow-all", str(tmp_path))))
        job_id = out["job"]["job_id"]
        assert get_job(job_id).permission_mode == "ask"

        # Re-tiering on update is held to the editor's ceiling too.
        up = json.loads(tool.cronjob(
            action="update", job_id=job_id, permission_mode="allow-all",
            agent=_caller("auto", "/elsewhere")))
        assert up["success"] is False
        assert get_job(job_id).permission_mode == "ask"
        # And an edit from another directory does not move the job.
        json.loads(tool.cronjob(
            action="update", job_id=job_id, permission_mode="auto",
            agent=_caller("auto", "/elsewhere")))
        assert get_job(job_id).work_dir == str(tmp_path)

    def test_mandate_is_not_in_the_llm_schema(self):
        from agentica.tools.cron_tool import CronTool

        fn = CronTool().functions["cronjob"]
        fn.process_entrypoint()
        props = fn.to_dict()["parameters"]["properties"]
        assert "agent" not in props and "mandate" not in props and "work_dir" not in props

    def test_invalid_tier_fails_loud(self, tmp_cron_dir):
        from agentica.cron.jobs import create_job

        with pytest.raises(ValueError):
            create_job(prompt="p", schedule="1h", permission_mode="yolo")


# ============== Run under the mandate ==============

class TestRunUnderMandate:
    def test_scheduler_passes_the_mandate_to_the_runner(self, tmp_cron_dir, tmp_path):
        from agentica.cron.jobs import create_job, get_job
        from agentica.cron.scheduler import _execute_job

        job = create_job(prompt="p", schedule="1h", permission_mode="ask", work_dir=str(tmp_path))
        runner = _RecordingRunner()
        asyncio.run(_execute_job(get_job(job.id), runner, verbose=False))
        ctx = runner.contexts[0]
        assert (ctx["permission_mode"], ctx["work_dir"]) == ("ask", str(tmp_path))

    def test_cli_job_agent_ignores_the_hosts_current_tier(self, tmp_path):
        """The factory closes over the host's agent_config, which `/permissions`
        mutates in place. A job created under "ask" must still run under "ask"
        after the host switched to allow-all, in the job's own directory."""
        from agentica.cli.runtime import parse_args
        from agentica.cron.cli_runner import build_cli_agent_factory

        sys.argv = ["agentica"]
        cfg = vars(parse_args())
        cfg.update({"model_provider": "openai", "model_name": "gpt-4o",
                    "api_key": "sk-test", "base_url": None, "work_dir": "/host/dir"})
        factory = build_cli_agent_factory(cfg)
        cfg["permissions"] = "allow-all"  # host flips tier after the daemon started

        agent = factory(permission_mode="ask", work_dir=str(tmp_path))
        assert agent.tool_config.permission_mode == "ask"
        assert agent.work_dir == str(tmp_path)
        assert cfg["work_dir"] == "/host/dir"

        # A job with no mandate (created before this field existed) keeps the
        # old behavior: the host's tier.
        legacy = factory()
        assert legacy.tool_config.permission_mode == "allow-all"

    def test_cli_runner_hands_context_mandate_to_factory_and_tells_job_its_id(self):
        from agentica.cron.cli_runner import CliAgentRunner

        seen = {}

        class _Agent:
            guidance = []

            def add_session_guidance(self, text):
                self.guidance.append(text)

            async def run(self, message):
                return SimpleNamespace(content="ok", cost_tracker=None)

        agent = _Agent()

        def factory(**kw):
            seen.update(kw)
            return agent

        asyncio.run(CliAgentRunner(factory).run(
            "p", {"job_id": "j42", "permission_mode": "ask", "work_dir": "/w"}))
        assert seen["permission_mode"] == "ask" and seen["work_dir"] == "/w"
        assert any("j42" in g and "cronjob(action='pause'" in g for g in agent.guidance)

    def test_gateway_run_cron_applies_the_mandate(self, tmp_path):
        pytest.importorskip("fastapi", reason="requires agentica[gateway]")
        from unittest.mock import AsyncMock, MagicMock
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        agent = MagicMock()
        agent.run = AsyncMock(return_value=MagicMock(content="done", tools=[]))
        seen = {}

        def build(session_id, owner=None):
            seen["mode"] = svc.get_session_approval_mode(session_id, owner)
            seen["dir"] = svc.get_session_work_dir(session_id, owner)
            return agent

        svc._build_agent = build
        asyncio.run(svc.run_cron("x", job_id="j1", user_id="u1",
                                 permission_mode="ask", work_dir=str(tmp_path)))
        assert seen == {"mode": "ask", "dir": str(tmp_path)}


# ============== Pause instead of retry ==============

class TestPausedWithReason:
    def test_missing_work_dir_pauses_without_running(self, tmp_cron_dir, tmp_path):
        from agentica.cron.jobs import create_job, get_job, list_task_runs
        from agentica.cron.scheduler import _execute_job
        from agentica.cron.types import JobStatus

        gone = tmp_path / "gone"
        job = create_job(prompt="p", schedule="1h", work_dir=str(gone), max_retries=3)
        runner = _RecordingRunner()
        result = asyncio.run(_execute_job(get_job(job.id), runner, verbose=False))

        assert result["status"] == "paused"
        assert runner.contexts == []
        after = get_job(job.id)
        assert after.status == JobStatus.PAUSED and not after.enabled
        assert str(gone) in after.paused_reason
        assert after.retry_count == 0  # waits for a person, not the retry delay
        assert list_task_runs(job.id)[0].error_type == "MandateError"

    def test_job_pausing_itself_stays_paused_with_reason(self, tmp_cron_dir):
        """A one-shot job that pauses itself must not be stamped completed, and
        a failing one must not be retried past its own pause."""
        from datetime import datetime, timedelta
        from agentica.cron.jobs import create_job, get_job
        from agentica.cron.scheduler import _execute_job
        from agentica.cron.types import JobStatus
        from agentica.tools.cron_tool import CronTool

        at = (datetime.now() + timedelta(hours=1)).isoformat(timespec="seconds")
        job = create_job(prompt="p", schedule=at, max_retries=2)

        class _SelfPausing:
            async def run(self, prompt, context=None):
                CronTool().cronjob(action="pause", job_id=context["job_id"],
                                   reason="needs GITHUB_TOKEN")
                raise RuntimeError("gave up")

        asyncio.run(_execute_job(get_job(job.id), _SelfPausing(), verbose=False))
        after = get_job(job.id)
        assert after.status == JobStatus.PAUSED
        assert after.paused_reason == "needs GITHUB_TOKEN"
        assert after.retry_count == 0

    def test_resume_clears_the_reason_and_manual_pause_sets_none(self, tmp_cron_dir):
        from agentica.cron.jobs import create_job, pause_job, resume_job

        job = create_job(prompt="p", schedule="1h")
        assert pause_job(job.id, reason="x").paused_reason == "x"
        assert resume_job(job.id).paused_reason == ""
        assert pause_job(job.id).paused_reason == ""

    def test_cron_list_shows_the_reason(self, tmp_cron_dir):
        from agentica.cli.commands import cron_cmd
        from agentica.cron.jobs import create_job, pause_job

        job = create_job(prompt="p", schedule="1h")
        pause_job(job.id, reason="work_dir /x no longer exists")
        printed = []
        with patch.object(cron_cmd, "get_console",
                          return_value=SimpleNamespace(print=lambda *a, **k: printed.append(a[0] if a else ""))):
            cron_cmd._cmd_cron(SimpleNamespace(), "list")
        assert any("work_dir /x no longer exists" in line and f"/cron resume {job.id}" in line
                   for line in printed)
