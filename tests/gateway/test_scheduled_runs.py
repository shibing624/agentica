# -*- coding: utf-8 -*-
"""The gateway delivers each scheduled run into a conversation.

Pins: where a run lands (the job's conversation, IM first when bound, a
terminal that has it open), what is recorded (session id, job on the turn),
that runs wait for an idle conversation instead of steering into it, and that
due jobs are claimed once and run independently.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")


@pytest.fixture
def cron_store(tmp_path):
    cron_dir = tmp_path / "cron"
    with patch("agentica.cron.jobs.CRON_DIR", cron_dir), \
         patch("agentica.cron.jobs.JOBS_FILE", cron_dir / "jobs.json"), \
         patch("agentica.cron.jobs.OUTPUT_DIR", cron_dir / "output"), \
         patch("agentica.cron.jobs.RUNS_FILE", cron_dir / "runs.jsonl"), \
         patch("agentica.cron.scheduler.LOCK_FILE", cron_dir / ".tick.lock"):
        yield cron_dir


def _svc(tmp_path):
    svc = MagicMock()
    log = MagicMock()
    log.get_name.return_value = None
    svc.session_log_for.return_value = log
    return svc


def _result(content="answer", error=""):
    from agentica_gateway.services.agent_service import ChatResult

    return ChatResult(content=content, session_id="x", error=error,
                      turn_usage={"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 0,
                                  "cache_write_tokens": 0, "api_calls": 1, "cost_usd": 0.0,
                                  "cache_hit_percent": 0})


class TestGatewayCronRunner:
    def _run(self, runner, ctx):
        with patch("agentica_gateway.services.scheduler_service.live_cli_peer", return_value=None):
            return asyncio.run(runner.run("msg", ctx))

    def test_delivers_into_the_jobs_conversation_with_the_job_on_the_turn(self, tmp_path):
        from agentica_gateway.services.scheduler_service import GatewayCronRunner

        svc = _svc(tmp_path)
        turn = AsyncMock(return_value=_result())
        with patch("agentica_gateway.routes.chat.run_scheduled_turn", turn):
            out = self._run(GatewayCronRunner(svc), {
                "job_id": "j1", "job_name": "Daily digest", "user_id": "u1",
                "session_id": "conv-1", "work_dir": str(tmp_path), "permission_mode": "ask",
            })
        turn.assert_awaited_once_with(svc, "conv-1", "u1", "msg", {"cron_job": {"id": "j1", "name": "Daily digest"}})
        assert out.session_id == "conv-1" and out.content == "answer"
        assert out.usage["input_tokens"] == 10 and "cache_hit_percent" not in out.usage
        svc.set_session_work_dir.assert_called_once_with("conv-1", str(tmp_path), "u1")
        svc.set_session_approval_mode.assert_called_once_with("conv-1", "ask", "u1")
        # A conversation the user already has keeps its own title.
        svc.session_log_for.return_value.set_name.assert_not_called()

    def test_no_conversation_yet_starts_one(self, tmp_path):
        from agentica_gateway.services.scheduler_service import GatewayCronRunner

        turn = AsyncMock(return_value=_result())
        with patch("agentica_gateway.routes.chat.run_scheduled_turn", turn):
            svc = _svc(tmp_path)
            out = self._run(GatewayCronRunner(svc), {"job_id": "j1", "job_name": "Daily digest", "session_id": ""})
        assert out.session_id and turn.await_args.args[1] == out.session_id
        # A conversation the job creates is named after it.
        svc.session_log_for.return_value.set_name.assert_called_once_with("Daily digest")

    def test_an_im_bound_conversation_runs_through_the_im_bot(self, tmp_path):
        from agentica_gateway.services.scheduler_service import GatewayCronRunner

        deliver_im = AsyncMock(return_value=_result("from phone"))
        turn = AsyncMock()
        with patch("agentica_gateway.routes.chat.run_scheduled_turn", turn):
            out = self._run(GatewayCronRunner(_svc(tmp_path), deliver_im=deliver_im),
                            {"job_id": "j1", "user_id": "u1", "session_id": "conv-1"})
        assert out.content == "from phone"
        turn.assert_not_awaited()

    def test_a_failed_turn_is_a_failed_run(self, tmp_path):
        from agentica_gateway.services.scheduler_service import GatewayCronRunner

        with patch("agentica_gateway.routes.chat.run_scheduled_turn", AsyncMock(return_value=_result(error="boom"))):
            with pytest.raises(RuntimeError, match="boom"):
                self._run(GatewayCronRunner(_svc(tmp_path)), {"job_id": "j1", "session_id": "c"})

    def test_a_terminal_with_the_conversation_open_gets_the_prompt(self, tmp_path):
        from agentica_gateway.services.scheduler_service import GatewayCronRunner

        peer = SimpleNamespace(name="term", session_id="conv-1")
        sent = []
        turn = AsyncMock()
        with patch("agentica_gateway.services.scheduler_service.live_cli_peer", return_value=peer), \
             patch("agentica.peers.send_message", side_effect=lambda p, **kw: sent.append(kw["text"])), \
             patch("agentica_gateway.routes.chat.run_scheduled_turn", turn):
            out = asyncio.run(GatewayCronRunner(_svc(tmp_path)).run("msg", {"job_id": "j1", "session_id": "conv-1"}))
        assert sent == ["msg"] and out.session_id == "conv-1"
        turn.assert_not_awaited()


class TestRunScheduledTurn:
    def test_waits_for_an_idle_conversation_and_logs_an_automated_turn(self, tmp_path):
        from agentica_gateway.routes import chat
        from agentica_gateway.services import live_turn

        live_turn.reset()
        busy = {"n": 2}
        svc = MagicMock()
        svc.is_session_active.side_effect = lambda *a, **k: busy.__setitem__("n", busy["n"] - 1) or busy["n"] >= 0
        log = MagicMock()
        log.path = tmp_path / "c.jsonl"
        svc.session_log_for.return_value = log
        svc.chat_stream = AsyncMock(return_value=_result())
        svc.get_context_window.return_value = 1000

        async def go():
            with patch.object(chat.asyncio, "sleep", AsyncMock()) as slept:
                result = await chat.run_scheduled_turn(svc, "conv-1", "u1", "msg", {"cron_job": {"id": "j1"}})
            return result, slept

        result, slept = asyncio.run(go())
        assert result.content == "answer"
        assert slept.await_count >= 1
        kwargs = svc.chat_stream.await_args.kwargs
        assert kwargs["automated"] is True and kwargs["input_meta"] == {"cron_job": {"id": "j1"}}
        assert kwargs["session_id"] == "conv-1" and kwargs["owner"] == "u1"
        live_turn.reset()

    def test_the_live_run_names_its_job_for_an_open_page(self, tmp_path):
        """A page attaching mid-run reads the active run: it must say which job
        sent the message, not present the prompt as something the user typed."""
        from agentica_gateway.routes import chat
        from agentica_gateway.services import live_turn

        live_turn.reset()
        svc = MagicMock()
        svc.is_session_active.return_value = False
        log = MagicMock()
        log.path = tmp_path / "c.jsonl"
        svc.session_log_for.return_value = log
        seen = {}

        async def stream(**kw):
            seen.update(live_turn.active("conv-1", "u1").public())
            return _result()

        svc.chat_stream = stream
        svc.get_context_window.return_value = 1000
        job = {"id": "j1", "name": "Daily"}
        asyncio.run(chat.run_scheduled_turn(svc, "conv-1", "u1", "do it", {"cron_job": job}))
        assert seen["cron_job"] == job
        assert seen["automated_message"] == "do it"
        assert "user_message" not in seen
        live_turn.reset()


class TestSchedulerService:
    def test_claim_takes_a_due_job_once(self, cron_store):
        from agentica.cron.jobs import create_job, get_job, update_job
        from agentica.cron.scheduler import claim_due_jobs

        job = create_job(prompt="p", schedule="1h")
        update_job(job.id, {"next_run_at_ms": 1})
        assert [j.id for j in claim_due_jobs()] == [job.id]
        assert claim_due_jobs() == []
        assert get_job(job.id).next_run_at_ms > 1

    def test_a_job_does_not_run_twice_at_once(self, cron_store):
        from agentica.cron.jobs import create_job, get_job, list_task_runs
        from agentica.cron.scheduler import execute_job

        job = create_job(prompt="p", schedule="1h")
        gate = asyncio.Event()

        class Slow:
            async def run(self, prompt, context=None):
                await gate.wait()
                return "ok"

        async def go():
            first = asyncio.create_task(execute_job(get_job(job.id), Slow(), False))
            await asyncio.sleep(0)
            second = await execute_job(get_job(job.id), Slow(), False)
            gate.set()
            return await first, second

        first, second = asyncio.run(go())
        assert first["status"] == "ok" and second["status"] == "busy"
        assert len(list_task_runs(job.id)) == 1

    def test_a_new_once_job_reuses_its_first_conversation(self, cron_store):
        from agentica.cron.jobs import create_job, get_job
        from agentica.cron.scheduler import execute_job
        from agentica.cron.types import CronRunOutput

        job = create_job(prompt="p", schedule="1h", target="new_once")
        seen = []

        class R:
            async def run(self, prompt, context=None):
                seen.append(context["session_id"])
                return CronRunOutput(content="ok", session_id=context["session_id"] or "conv-made")

        asyncio.run(execute_job(get_job(job.id), R(), False))
        asyncio.run(execute_job(get_job(job.id), R(), False))
        assert seen == ["", "conv-made"]

    def test_a_new_job_starts_a_fresh_conversation_every_run(self, cron_store):
        from agentica.cron.jobs import create_job, get_job
        from agentica.cron.scheduler import execute_job
        from agentica.cron.types import CronRunOutput

        job = create_job(prompt="p", schedule="1h", target="new")
        seen = []

        class R:
            async def run(self, prompt, context=None):
                seen.append(context["session_id"])
                return CronRunOutput(content="ok", session_id=f"conv-{len(seen)}")

        asyncio.run(execute_job(get_job(job.id), R(), False))
        asyncio.run(execute_job(get_job(job.id), R(), False))
        assert seen == ["", ""]
        assert get_job(job.id).session_id == "conv-2"

    def test_run_now_names_the_conversation_before_the_run_starts(self, cron_store):
        from agentica.cron.jobs import create_job, get_job, list_task_runs
        from agentica.cron.types import CronRunOutput
        from agentica_gateway.services.scheduler_service import GatewayScheduler

        job = create_job(prompt="p", schedule="1h")

        class R:
            async def run(self, prompt, context=None):
                return CronRunOutput(content="ok", session_id=context["session_id"])

        async def go():
            sched = GatewayScheduler(R())
            started = sched.run_now(get_job(job.id))
            busy = sched.run_now(get_job(job.id))
            await asyncio.gather(*sched._runs)
            return started, busy

        started, busy = asyncio.run(go())
        assert started["status"] == "queued" and busy["status"] == "busy"
        assert list_task_runs(job.id)[0].session_id == started["session_id"]

    def test_run_now_titles_fresh_session_synchronously_and_preserves_later_user_rename(self, cron_store, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOME", str(tmp_path / "home"))
        from agentica.cron.jobs import create_job, get_job
        from agentica_gateway.services.agent_service import AgentService
        from agentica_gateway.services.scheduler_service import GatewayCronRunner, GatewayScheduler

        work_dir = tmp_path / "proj"
        work_dir.mkdir()
        svc = AgentService(workspace_path=str(tmp_path / "ws"))
        job = create_job(
            name="每日 CI 巡检（agentica + memhub）",
            prompt="运行巡检脚本",
            schedule="1h",
            target="new_once",
            work_dir=str(work_dir),
            user_id="default",
        )
        gate = asyncio.Event()
        turn = AsyncMock(side_effect=lambda *a, **kw: gate.wait())

        async def go():
            with patch("agentica_gateway.services.scheduler_service.live_cli_peer", return_value=None), \
                 patch("agentica_gateway.routes.chat.run_scheduled_turn", turn):
                sched = GatewayScheduler(GatewayCronRunner(svc))
                started = sched.run_now(get_job(job.id))
                sid = started["session_id"]
                # Before the background task even finishes (right when AutomationsPage
                # calls GET /api/sessions), the session is already listed with the job title!
                rows_before = {r["session_id"]: r for r in svc.list_sessions(owner="default", scope="all")}
                assert rows_before[sid]["name"] == "每日 CI 巡检（agentica + memhub）"

                turn.side_effect = None
                turn.return_value = _result("ok")
                gate.set()
                await asyncio.gather(*sched._runs)

                # User manually renames the new_once conversation; run 2 must keep the user's title.
                svc.rename_session(sid, "我的 CI 巡检看板", owner="default")
                second = sched.run_now(get_job(job.id))
                assert second["session_id"] == sid
                await asyncio.gather(*sched._runs)
                rows_after = {r["session_id"]: r for r in svc.list_sessions(owner="default", scope="all")}
                assert rows_after[sid]["name"] == "我的 CI 巡检看板"

        asyncio.run(go())

    def test_unnamed_historical_and_inflight_cron_sessions_fallback_to_cron_job_name(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOME", str(tmp_path / "home"))
        from agentica.memory.session_log import SessionLog
        from agentica_gateway.services import live_turn
        from agentica_gateway.services.agent_service import AgentService

        live_turn.reset()
        work_dir = tmp_path / "proj"
        work_dir.mkdir()
        svc = AgentService(workspace_path=str(tmp_path / "ws"))

        # 1. Historical session created before set_name ran on run_now (no `name` in meta.json)
        log = SessionLog("hist-cron-1", work_dir=str(work_dir), user_id="default")
        log.set_source("gateway")
        log.append(
            "user",
            "[Scheduled task “每日 CI 巡检” (j1)]\ncheck ci",
            automated=True,
            cron_job={"id": "j1", "name": "每日 CI 巡检"},
        )
        log.append("assistant", "[NO_UPDATE]")

        # 2. In-flight LiveTurn whose JSONL has not been written yet
        inflight = live_turn.start("live-cron-2", owner="default")
        inflight.cron_job = {"id": "j2", "name": "每小时服务健康检查"}
        inflight.automated_message = "check health"

        rows = {r["session_id"]: r for r in svc.list_sessions(owner="default", scope="all")}
        assert rows["hist-cron-1"]["name"] == "每日 CI 巡检"
        assert rows["live-cron-2"]["name"] == "每小时服务健康检查"
        live_turn.reset()
