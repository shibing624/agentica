# -*- coding: utf-8 -*-
"""Each Web/Desktop chat runs on its own model.

Picking a model in one chat used to switch config.yaml's active profile, so
every other chat in the gateway changed model with it. A pick is now stored on
the chat (``<id>.meta.json`` ``profile_name``), a chat without one follows its
project's default, and the gateway default stays where it was.
"""
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

import yaml

from agentica.config.profiles import get_active_profile_name, set_project_profile
from agentica_gateway.services import agent_service as svc_mod
from agentica_gateway.services.agent_service import AgentService

PROFILES = {
    "active_profile": "base",
    "profiles": {
        "base": {"model_provider": "openai", "model_name": "gpt-4o", "api_key": "k-base"},
        "fast": {
            "model_provider": "deepseek", "model_name": "deepseek-chat", "api_key": "k-fast",
            "reasoning_effort": "high",
            "auxiliary_model": {"model_name": "deepseek-lite"},
        },
        "proj": {"model_provider": "openai", "model_name": "gpt-4o-mini"},
    },
}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(PROFILES), encoding="utf-8")
    monkeypatch.setenv("AGENTICA_HOME", str(home))
    monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(tmp_path / "projects"))
    service = AgentService(workspace_path=str(tmp_path / "ws"))
    for sid, d in (("a", "proj_a"), ("b", "proj_a"), ("c", "proj_c")):
        work = tmp_path / d
        work.mkdir(exist_ok=True)
        service.set_session_work_dir(sid, str(work))
    return service


def _built(service, monkeypatch):
    """Record which (provider, model) each built agent got."""
    calls = []

    def fake_create(provider, name, **kw):
        calls.append((provider, name, kw))
        return SimpleNamespace(provider=provider, id=name)

    monkeypatch.setattr(svc_mod, "create_model", fake_create)
    return calls


def test_a_chat_without_a_pick_runs_on_the_gateway_default(svc):
    cfg = svc.session_model_config("a")
    assert svc.session_profile_source("a") == "default"
    assert (cfg["model_provider"], cfg["model_name"]) == (svc.model_provider, svc.model_name)


def test_picking_a_model_changes_only_that_chat(svc):
    asyncio.run(svc.set_session_profile("a", "fast"))

    a = svc.session_model_config("a")
    assert (a["profile"], a["model_provider"], a["model_name"]) == ("fast", "deepseek", "deepseek-chat")
    assert a["api_key"] == "k-fast" and a["reasoning_effort"] == "high"
    # Same provider: the auxiliary model reuses the main endpoint and key.
    assert (a["auxiliary_model_name"], a["auxiliary_api_key"]) == ("deepseek-lite", "k-fast")
    assert svc.session_profile_source("a") == "session"
    # The other chat in the same project, and the gateway default, did not move.
    assert svc.session_profile_source("b") == "default"
    assert svc.session_model_config("b")["model_name"] == svc.model_name
    assert get_active_profile_name() == "base"


def test_the_pick_survives_a_restart(svc, tmp_path):
    asyncio.run(svc.set_session_profile("a", "fast"))
    fresh = AgentService(workspace_path=str(tmp_path / "ws"))
    fresh.set_session_work_dir("a", svc.get_session_work_dir("a"))
    assert fresh.session_model_config("a")["model_name"] == "deepseek-chat"


def test_a_chat_without_a_pick_follows_its_project_default(svc):
    set_project_profile(svc.get_session_work_dir("c"), "proj")
    assert svc.session_profile_source("c") == "project"
    assert svc.session_model_config("c")["model_name"] == "gpt-4o-mini"
    # A pick still wins over the project's default.
    asyncio.run(svc.set_session_profile("c", "fast"))
    assert svc.session_model_config("c")["model_name"] == "deepseek-chat"


def test_a_deleted_profile_falls_back_instead_of_breaking_the_chat(svc):
    svc.session_log_for("a").set_profile("gone", "session")
    assert svc.session_profile_source("a") == "default"
    assert svc.session_model_config("a")["model_name"] == svc.model_name


def test_a_pick_rebuilds_only_that_chats_agent(svc, monkeypatch):
    built = []
    monkeypatch.setattr(
        svc, "_build_agent",
        lambda sid, owner=None: built.append(sid) or SimpleNamespace(user_id=svc._owner(owner)),
    )

    async def scenario():
        a, b = await svc._get_agent("a"), await svc._get_agent("b")
        await svc.set_session_profile("a", "fast")
        assert await svc._get_agent("a") is not a
        assert await svc._get_agent("b") is b

    asyncio.run(scenario())
    assert built == ["a", "b", "a"]


def test_build_agent_creates_the_main_model_from_the_session_profile(svc, monkeypatch):
    calls = _built(svc, monkeypatch)
    asyncio.run(svc.set_session_profile("a", "fast"))
    monkeypatch.setattr(
        svc_mod, "DeepAgent",
        lambda **kw: SimpleNamespace(**{**kw, "tools": [], "_session_log": None}),
    )
    svc._workspace = None
    svc._build_agent("a")
    provider, name, kw = calls[0]
    assert (provider, name) == ("deepseek", "deepseek-chat")
    assert kw["api_key"] == "k-fast" and kw["reasoning_effort"] == "high"
    assert calls[1][:2] == ("deepseek", "deepseek-lite")  # auxiliary from the same profile


def test_a_pick_mid_run_waits_for_the_run(svc, monkeypatch):
    built = []
    monkeypatch.setattr(svc, "_build_agent", lambda sid, owner=None: built.append(sid) or SimpleNamespace(user_id=svc._owner(owner)))

    async def scenario():
        running = await svc._get_agent("a")
        lock = svc._get_session_lock("a")
        await lock.acquire()
        await svc.set_session_profile("a", "fast")
        assert svc._cached_agent_for_owner("a") is running  # cancel/steer still reach it
        lock.release()
        assert await svc._get_agent("a") is not running

    asyncio.run(scenario())


def test_session_model_routes(svc, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from agentica_gateway import deps
    from agentica_gateway.routes import chat as chat_routes

    app = FastAPI()

    @app.middleware("http")
    async def principal(request, call_next):
        request.state.principal = SimpleNamespace(user_id=svc._owner(None))
        return await call_next(request)

    app.include_router(chat_routes.router)
    app.dependency_overrides[deps.get_agent_service] = lambda: svc
    client = TestClient(app)

    got = client.get("/api/sessions/a/model").json()
    assert got["source"] == "default" and got["model_name"] == svc.model_name

    picked = client.post("/api/sessions/a/model", json={"profile": "fast"})
    assert picked.status_code == 200
    assert (picked.json()["profile"], picked.json()["source"]) == ("fast", "session")
    assert client.get("/api/sessions/b/model").json()["source"] == "default"

    assert client.post("/api/sessions/a/model", json={"profile": "nope"}).status_code == 404
