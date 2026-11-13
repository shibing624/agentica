# -*- coding: utf-8 -*-
"""Gateway tool-approval: POST, owner auth, SSE replay, no-LiveTurn deny."""
import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agentica.agent.approvals import PendingApproval
from agentica_gateway.models import ApprovalDecisionRequest
from agentica_gateway.services import live_turn
from agentica.tools.base import Function, FunctionCall


@pytest.fixture(autouse=True)
def _reset_live_turn():
    live_turn.reset()
    yield
    live_turn.reset()


def _fc(name="execute", arguments=None, call_id="c1"):
    fn = Function(name=name)
    fn.entrypoint = lambda **kwargs: "ran"
    fn.is_destructive = True
    return FunctionCall(function=fn, arguments=arguments or {"command": "rm -f x"}, call_id=call_id)


def _request(owner: str):
    return SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(user_id=owner)))


def _pending(tool_call_id="t1"):
    return PendingApproval(
        tool_call_id=tool_call_id,
        name="execute",
        arguments={"command": "rm -f /tmp/x"},
        question="Allow running the following command?",
        preview="rm -f /tmp/x",
    )


@pytest.fixture()
def client():
    from agentica_gateway.main import app
    from agentica_gateway import deps

    with TestClient(app, raise_server_exceptions=False) as c:
        original = deps.agent_service
        deps.agent_service = MagicMock()
        yield c
        deps.agent_service = original


class TestApprovalPost:
    def test_http_unknown_id_is_404(self, client):
        resp = client.post(
            "/api/sessions/s1/approvals/missing",
            json={"decision": "allow"},
        )
        assert resp.status_code == 404

    def test_http_invalid_decision_is_422(self, client):
        resp = client.post(
            "/api/sessions/s1/approvals/t1",
            json={"decision": "maybe"},
        )
        assert resp.status_code == 422

    def test_allow_resolves_wait(self):
        from agentica_gateway.routes.chat import decide_session_approval

        async def _run():
            turn = live_turn.start("s1", owner="default")
            waiter = turn.approvals.wait(_pending())
            result = await decide_session_approval(
                "s1", "t1", ApprovalDecisionRequest(decision="allow"),
                _request("default"),
            )
            assert result == {"status": "ok", "tool_call_id": "t1", "decision": "allow"}
            assert await waiter == "allow"
            assert turn.approvals.size == 0

        asyncio.run(_run())

    def test_allow_prefix_and_deny_round_trip(self):
        from agentica_gateway.routes.chat import decide_session_approval

        async def _run():
            turn = live_turn.start("s1", owner="default")
            w1 = turn.approvals.wait(_pending("a"))
            w2 = turn.approvals.wait(_pending("b"))
            await decide_session_approval(
                "s1", "a", ApprovalDecisionRequest(decision="allow_prefix"),
                _request("default"),
            )
            await decide_session_approval(
                "s1", "b", ApprovalDecisionRequest(decision="deny"),
                _request("default"),
            )
            assert await w1 == "allow_prefix"
            assert await w2 == "deny"

        asyncio.run(_run())

    def test_unknown_id_on_live_turn_is_404(self):
        from agentica_gateway.routes.chat import decide_session_approval
        from fastapi import HTTPException

        async def _run():
            live_turn.start("s1", owner="default")
            with pytest.raises(HTTPException) as ei:
                await decide_session_approval(
                    "s1", "nope", ApprovalDecisionRequest(decision="deny"),
                    _request("default"),
                )
            assert ei.value.status_code == 404

        asyncio.run(_run())

    def test_other_owner_cannot_decide(self):
        from agentica_gateway.routes.chat import decide_session_approval
        from fastapi import HTTPException

        async def _run():
            turn = live_turn.start("s1", owner="alice")
            waiter = turn.approvals.wait(_pending())
            with pytest.raises(HTTPException) as ei:
                await decide_session_approval(
                    "s1", "t1", ApprovalDecisionRequest(decision="allow"),
                    _request("bob"),
                )
            assert ei.value.status_code == 404
            assert not waiter.done()
            assert turn.approvals.size == 1

        asyncio.run(_run())


class TestApprovalReplay:
    def test_subscribe_republishes_still_pending(self):
        from agentica_gateway.routes import chat as chat_mod

        async def _run():
            turn = live_turn.start("s1", owner="default")
            turn.publish({
                "event": "tool_call",
                "data": {"name": "execute", "args": {"command": "rm"}, "tool_call_id": "t1"},
            })
            turn.approvals.wait(_pending("t1"))
            gen = chat_mod._sse_from_turn(turn, after=1)
            found = None
            try:
                async for raw in gen:
                    if not raw.startswith("data: "):
                        continue
                    payload = raw[len("data: "):].strip()
                    if payload == "[DONE]":
                        break
                    ev = json.loads(payload)
                    if ev.get("event") == "approval_request":
                        found = ev["data"]
                        break
            finally:
                await gen.aclose()
            assert found is not None
            assert found["tool_call_id"] == "t1"
            assert found["name"] == "execute"
            assert found["preview"] == "rm -f /tmp/x"
            assert found["options"] == ["allow", "allow_prefix", "deny", "deny_prefix"]
            assert found["similar_label"] == ""
            assert "question" in found
            assert found["args"]["command"] == "rm -f /tmp/x"

        asyncio.run(_run())

    def test_tool_call_sse_includes_tool_call_id(self):
        from agentica_gateway.routes.chat import _sse_stream_hooks

        published = []
        _, on_tool_call, _, _ = _sse_stream_hooks(published.append)
        on_tool_call("execute", {"command": "ls"}, "call_9")
        assert published[0]["event"] == "tool_call"
        assert published[0]["data"]["tool_call_id"] == "call_9"
        assert published[0]["data"]["name"] == "execute"


class TestNoLiveTurnDeny:
    def test_injected_approve_denies_without_live_turn(self, tmp_path):
        from agentica_gateway.services.agent_service import AgentService

        async def _run():
            svc = AgentService(workspace_path=str(tmp_path))
            svc.set_session_approval_mode("s1", "ask", owner="default")
            approve = svc._make_session_approve("s1", "default")
            assert await approve(_fc()) == "deny"

        asyncio.run(_run())

    def test_injected_approve_parks_when_live_turn_exists(self, tmp_path):
        from agentica_gateway.services.agent_service import AgentService

        async def _run():
            svc = AgentService(workspace_path=str(tmp_path))
            svc.set_session_approval_mode("s1", "ask", owner="default")
            turn = live_turn.start("s1", owner="default")
            approve = svc._make_session_approve("s1", "default")
            task = asyncio.create_task(approve(_fc(call_id="park1")))
            await asyncio.sleep(0)
            assert turn.approvals.size == 1
            events = [e for e in turn.events if e.get("event") == "approval_request"]
            assert events
            assert events[-1]["data"]["tool_call_id"] == "park1"
            assert turn.approvals.decide("park1", "allow")
            assert await task == "allow"

        asyncio.run(_run())


class TestDenyAllOnFinish:
    def test_finish_denies_pending(self):
        async def _run():
            turn = live_turn.start("s1", owner="default")
            waiter = turn.approvals.wait(_pending())
            turn.finish("cancelled")
            assert await waiter == "deny"
            assert turn.approvals.size == 0

        asyncio.run(_run())

    def test_cancel_denies_pending_before_abort(self):
        async def _run():
            turn = live_turn.start("s1", owner="default")
            waiter = turn.approvals.wait(_pending())
            svc = MagicMock()
            svc.has_cached_session.return_value = False
            svc.is_session_active.return_value = False

            async def runner():
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    turn.finish("cancelled")
                    raise

            turn.task = asyncio.create_task(runner())
            result = await live_turn.cancel_and_wait(
                svc, run_id=turn.run_id, owner="default", timeout=2,
            )
            assert await waiter == "deny"
            assert result["status"] == "cancelled"

        asyncio.run(_run())


class TestMidTurnApprovalModeSwitch:
    def test_switching_to_allow_all_unparks_current_and_skips_next_in_same_turn(self, tmp_path):
        from agentica_gateway.services.agent_service import AgentService

        async def _run():
            svc = AgentService(workspace_path=str(tmp_path))
            svc.set_session_work_dir("s1", str(tmp_path), owner="default")
            svc.set_session_approval_mode("s1", "ask", owner="default")
            turn = live_turn.start("s1", owner="default")
            approve = svc._make_session_approve("s1", "default")

            fc1 = _fc("write_file", {"file_path": "/tmp/outside_mode_switch.txt"}, call_id="c1")
            wait1 = asyncio.create_task(approve(fc1))
            await asyncio.sleep(0)
            assert turn.approvals.size == 1

            # User gets tired of approving and flips permission mode to allow-all mid-turn.
            svc.set_session_approval_mode("s1", "allow-all", owner="default")

            assert await wait1 == "allow"
            assert turn.approvals.size == 0
            assert fc1.approval_trace["reason"] == "mode_switch"
            assert fc1.approval_trace["mode"] == "allow-all"
            assert fc1.approval_trace["grant"] is None
            # Switching back to ask must NOT leave /tmp/outside_mode_switch.txt whitelisted.
            grants = svc._grants_for("s1", "default")
            assert not grants.path_exact

            resolved = [e for e in turn.events if e.get("event") == "approval_resolved"]
            assert len(resolved) == 1
            assert resolved[0]["data"] == {
                "tool_call_id": "c1",
                "decision": "allow",
                "by": "mode",
            }

            # Next mutating execute in the same turn runs immediately without parking.
            fc2 = _fc("execute", {"command": "python run_step.py"}, call_id="c2")
            assert await approve(fc2) == "allow"
            assert turn.approvals.size == 0

        asyncio.run(_run())

    def test_switching_ask_to_auto_unparks_execute_but_keeps_outside_write_parked(self, tmp_path):
        from agentica_gateway.services.agent_service import AgentService

        async def _run():
            work = tmp_path / "proj"
            outside = tmp_path / "outside.txt"
            work.mkdir()
            svc = AgentService(workspace_path=str(tmp_path))
            svc.set_session_work_dir("s1", str(work), owner="default")
            svc.set_session_approval_mode("s1", "ask", owner="default")
            turn = live_turn.start("s1", owner="default")
            approve = svc._make_session_approve("s1", "default")

            fc_exec = _fc("execute", {"command": "python run_step.py"}, call_id="exec1")
            fc_out = _fc("write_file", {"file_path": str(outside)}, call_id="out1")
            t_exec = asyncio.create_task(approve(fc_exec))
            t_out = asyncio.create_task(approve(fc_out))
            await asyncio.sleep(0)
            assert turn.approvals.size == 2

            svc.set_session_approval_mode("s1", "auto", owner="default")
            assert await t_exec == "allow"
            assert not t_out.done()
            assert [p.tool_call_id for p in turn.approvals.list()] == ["out1"]

            turn.resolve_approval("out1", "deny", by="Web")
            assert await t_out == "deny"

        asyncio.run(_run())

    def test_route_and_prefs_update_live_non_cron_turn_only(self, tmp_path, monkeypatch):
        from agentica_gateway import deps
        from agentica_gateway.models import SessionApprovalModeRequest
        from agentica_gateway.routes import chat as chat_routes
        from agentica_gateway.routes import settings as settings_routes
        from agentica_gateway.services.agent_service import AgentService

        monkeypatch.setattr(settings_routes, "AGENTICA_HOME", str(tmp_path))

        async def _run():
            svc = AgentService(workspace_path=str(tmp_path))
            original = deps.agent_service
            deps.agent_service = svc
            try:
                svc.set_session_approval_mode("chat1", "ask", owner="default")
                svc.set_session_approval_mode("cron1", "ask", owner="default")
                t_chat = live_turn.start("chat1", owner="default")
                t_cron = live_turn.start("cron1", owner="default")
                t_cron.cron_job = {"id": "j1", "name": "nightly"}

                resp = await chat_routes.set_session_approval_mode(
                    "chat1",
                    SessionApprovalModeRequest(approval_mode="auto"),
                    _request("default"),
                    svc=svc,
                )
                assert resp == {"session_id": "chat1", "approval_mode": "auto"}

                w_chat = t_chat.approvals.wait(_pending("tc_chat"))
                w_cron = t_cron.approvals.wait(_pending("tc_cron"))

                req = SimpleNamespace(
                    state=SimpleNamespace(principal=SimpleNamespace(user_id="default")),
                    json=lambda: asyncio.sleep(0, result={"approval_mode": "allow-all"}),
                )
                await settings_routes.put_prefs(req)

                assert svc.get_session_approval_mode("chat1", owner="default") == "allow-all"
                assert await w_chat == "allow"
                # Cron mandate stays untouched.
                assert svc.get_session_approval_mode("cron1", owner="default") == "ask"
                assert not w_cron.done()
            finally:
                deps.agent_service = original

        asyncio.run(_run())
