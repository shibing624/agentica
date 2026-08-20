# -*- coding: utf-8 -*-
"""Profile CRUD through Gateway HTTP — the Web Settings → Profiles surface."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from agentica.gateway.main import app
    from agentica.gateway import deps

    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    svc.has_active_runs = MagicMock(return_value=False)
    svc.reload_profile = AsyncMock()
    svc.get_context_window = MagicMock(return_value=128000)
    svc.model_provider = "openai"
    svc.model_name = "gpt-4o"
    svc.max_tokens = 0
    svc.temperature = 0.0
    svc.top_p = 0.0
    svc.model_wire_api = ""
    svc.model_reasoning = ""
    svc.model_reasoning_effort = ""

    with TestClient(app, raise_server_exceptions=False) as c:
        original = deps.agent_service
        deps.agent_service = svc
        yield c, svc
        deps.agent_service = original


def test_list_profiles_shape(client):
    c, _ = client
    profiles = {
        "default": {"model_provider": "openai", "model_name": "gpt-4o", "api_key": "sk-testkey-xxxx"},
        "cheap": {"model_provider": "deepseek", "model_name": "deepseek-v4-flash", "base_url": "https://api.deepseek.com"},
    }
    with patch("agentica.gateway.routes.settings.get_profiles", return_value=profiles), \
         patch("agentica.gateway.routes.settings.get_active_profile_name", return_value="default"):
        resp = c.get("/api/profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] == "default"
    names = [p["name"] for p in data["profiles"]]
    assert names == ["default", "cheap"]
    cheap = next(p for p in data["profiles"] if p["name"] == "cheap")
    assert cheap["model_provider"] == "deepseek"
    assert cheap["base_url"] == "https://api.deepseek.com"


def test_create_switch_delete_profile(client):
    c, svc = client
    with patch("agentica.gateway.routes.settings.upsert_profile") as upsert:
        created = c.post("/api/profile", json={
            "name": "web-v2-smoke",
            "model_provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "sk-test-not-real",
            "reasoning_effort": "low",
        })
    assert created.status_code == 200, created.text
    assert created.json()["name"] == "web-v2-smoke"
    assert upsert.call_args.args[0] == "web-v2-smoke"
    assert upsert.call_args.kwargs["make_active"] is False

    with patch("agentica.gateway.routes.settings.get_profiles", return_value={"web-v2-smoke": {}}), \
         patch("agentica.gateway.routes.settings.get_active_profile_name", return_value="default"):
        switched = c.post("/api/profile/switch", json={"name": "web-v2-smoke"})
    assert switched.status_code == 200, switched.text
    assert switched.json()["active_profile"] == "web-v2-smoke"
    svc.reload_profile.assert_awaited_with("web-v2-smoke")

    with patch("agentica.gateway.routes.settings.get_profile", return_value={"model_name": "gpt-4o-mini"}), \
         patch("agentica.gateway.routes.settings.delete_profile", return_value=True) as delete:
        removed = c.delete("/api/profile/web-v2-smoke")
    assert removed.status_code == 200
    delete.assert_called_once_with("web-v2-smoke")
