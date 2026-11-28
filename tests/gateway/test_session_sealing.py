# -*- coding: utf-8 -*-
"""A turn a dead process left open is closed when the session is next built.

The gateway can be SIGKILLed mid-turn; the log then ends on a tool result with
no assistant reply. ``seal_incomplete_turn`` appends the marker that keeps the
replay valid and tells the model the turn was interrupted. It runs when the
session's agent is built. A run builds under the session lock it holds itself,
and that lock must not read as another live turn: before this, the turn was
sealed only when the session was opened first, and a message sent straight after
the crash put two user turns in a row on the wire. Whether the model then picks
the dead turn's work up is not what the seal decides.
"""
import asyncio
import json

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica.session.log import SessionLog
from agentica_gateway.services import agent_service as svc_mod
from agentica_gateway.services.agent_service import AgentService

SEALED = "[Session ended before the assistant replied]"


def _service(tmp_path, monkeypatch):
    monkeypatch.setattr(svc_mod, "get_profiles", lambda: {})
    monkeypatch.setattr(svc_mod, "get_project_profile", lambda wd: None)
    monkeypatch.setattr(svc_mod, "get_active_profile_name", lambda: "")
    monkeypatch.setattr(svc_mod, "get_profile", lambda name: None)
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    svc.model_provider = "openai"
    svc.model_name = "gpt-4o-mini"
    svc.api_key = "sk-test"
    asyncio.run(svc._ensure_initialized())
    return svc


def _interrupted_log(tmp_path, session_id):
    """A log as a SIGKILL leaves it: a tool result, then nothing."""
    work = tmp_path / session_id
    work.mkdir()
    log = SessionLog(session_id=session_id, work_dir=str(work), user_id="default")
    log.append("user", "create three files, one write_file call each")
    log.append("assistant", "", finish_reason="tool_calls", tool_calls=[{
        "id": "call_1", "type": "function",
        "function": {"name": "write_file", "arguments": '{"file_path": "step1.txt", "content": "step1.txt"}'},
    }])
    log.append("tool", "Created file step1.txt", tool_call_id="call_1", tool_name="write_file", is_error=False)
    return log


def _tail(log):
    return json.loads(log.path.read_text(encoding="utf-8").splitlines()[-1])


def test_get_agent_for_run_seals_under_its_own_lock(tmp_path, monkeypatch):
    """Exercises the full _get_agent(for_run=True) -> _build_and_cache_agent -> _build_agent wiring."""
    svc = _service(tmp_path, monkeypatch)
    log = _interrupted_log(tmp_path, "crashed-via-get")
    svc.set_session_work_dir("crashed-via-get", str(tmp_path / "crashed-via-get"))
    lock = svc._get_session_lock("crashed-via-get")

    async def scenario():
        await lock.acquire()
        try:
            await svc._get_agent("crashed-via-get", for_run=True)
        finally:
            lock.release()

    asyncio.run(scenario())
    tail = _tail(log)
    assert (tail.get("content"), tail.get("finish_reason")) == (SEALED, "interrupted"), tail


def test_a_run_building_its_own_session_seals_the_turn_a_dead_process_left(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    log = _interrupted_log(tmp_path, "crashed")
    lock = svc._get_session_lock("crashed")

    async def scenario():
        await lock.acquire()  # the run holds its own session lock while it builds
        try:
            svc._build_agent("crashed", under_own_lock=True)
        finally:
            lock.release()

    asyncio.run(scenario())
    tail = _tail(log)
    assert (tail.get("content"), tail.get("finish_reason")) == (SEALED, "interrupted"), tail


def test_a_build_that_does_not_hold_the_lock_leaves_a_live_turn_alone(tmp_path, monkeypatch):
    """A usage poll may build while a turn is in flight; that turn is not dead."""
    svc = _service(tmp_path, monkeypatch)
    log = _interrupted_log(tmp_path, "live")
    lock = svc._get_session_lock("live")

    async def scenario():
        await lock.acquire()
        try:
            svc._build_agent("live")
        finally:
            lock.release()

    asyncio.run(scenario())
    assert _tail(log)["type"] == "tool"


def test_an_idle_session_is_sealed_whatever_asked_for_the_build(tmp_path, monkeypatch):
    svc = _service(tmp_path, monkeypatch)
    log = _interrupted_log(tmp_path, "opened")
    svc._build_agent("opened")
    assert _tail(log)["finish_reason"] == "interrupted"
