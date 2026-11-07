# -*- coding: utf-8 -*-
"""Switching model/profile while another session runs.

The switch used to 409 whenever *any* session had a run in flight, so a new
chat on another model failed with "Cannot switch profile while a run is
active". Now the running session keeps its agent until the run ends and every
other session rebuilds on the new settings.
"""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica_gateway.services.agent_service import AgentService


def _service(tmp_path, monkeypatch):
    svc = AgentService(workspace_path=str(tmp_path))
    built = []

    def build(session_id, owner=None):
        agent = SimpleNamespace(user_id=svc._owner(owner), session_id=session_id, n=len(built))
        built.append(agent)
        return agent

    monkeypatch.setattr(svc, "_build_agent", build)
    return svc, built


def test_switch_mid_run_keeps_the_running_agent_and_rebuilds_the_rest(tmp_path, monkeypatch):
    svc, built = _service(tmp_path, monkeypatch)

    async def scenario():
        running = await svc._get_agent("a")
        idle = await svc._get_agent("b")
        lock = svc._get_session_lock("a")
        await lock.acquire()  # session a is mid-run

        await svc._invalidate_cache()

        # The run in flight is still reachable (cancel / steer find it).
        assert svc._cached_agent_for_owner("a") is running
        # Another session builds on the new settings right away.
        assert await svc._get_agent("b") is not idle

        lock.release()  # run a ends
        rebuilt = await svc._get_agent("a")
        assert rebuilt is not running
        # ...and only once: the next request reuses the rebuilt agent.
        assert await svc._get_agent("a") is rebuilt

    asyncio.run(scenario())
    assert len(built) == 4


def test_profile_switch_route_no_longer_refuses_during_a_run(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentica_gateway import deps
    from agentica_gateway.routes import settings as settings_routes

    svc, _ = _service(tmp_path, monkeypatch)
    reloaded = []

    async def reload_profile(name=None):
        reloaded.append(name)

    monkeypatch.setattr(svc, "reload_profile", reload_profile)
    monkeypatch.setattr(settings_routes, "get_profiles", lambda: {"fast": {}})
    asyncio.run(svc._get_session_lock("busy").acquire())

    app = FastAPI()
    app.include_router(settings_routes.router)
    app.dependency_overrides[deps.get_agent_service] = lambda: svc
    resp = TestClient(app).post("/api/profile/switch", json={"name": "fast"})
    assert resp.status_code == 200, resp.text
    assert reloaded == ["fast"]
