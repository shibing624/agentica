# -*- coding: utf-8 -*-
"""Gateway tool-approval: POST, owner auth, SSE replay, no-LiveTurn deny."""
import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agentica.agent.approvals import PendingApproval
from agentica.gateway.models import ApprovalDecisionRequest
from agentica.gateway.services import live_turn
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
    from agentica.gateway.main import app
    from agentica.gateway import deps

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
        from agentica.gateway.routes.chat import decide_session_approval

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
        from agentica.gateway.routes.chat import decide_session_approval

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
        from agentica.gateway.routes.chat import decide_session_approval
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
        from agentica.gateway.routes.chat import decide_session_approval
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
        from agentica.gateway.routes import chat as chat_mod

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
        from agentica.gateway.routes.chat import _sse_stream_hooks

        published = []
        _, on_tool_call, _, _ = _sse_stream_hooks(published.append)
        on_tool_call("execute", {"command": "ls"}, "call_9")
        assert published[0]["event"] == "tool_call"
        assert published[0]["data"]["tool_call_id"] == "call_9"
        assert published[0]["data"]["name"] == "execute"


class TestNoLiveTurnDeny:
    def test_injected_approve_denies_without_live_turn(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        async def _run():
            svc = AgentService(workspace_path=str(tmp_path))
            svc.set_session_approval_mode("s1", "ask", owner="default")
            approve = svc._make_session_approve("s1", "default")
            assert await approve(_fc()) == "deny"

        asyncio.run(_run())

    def test_injected_approve_parks_when_live_turn_exists(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

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
