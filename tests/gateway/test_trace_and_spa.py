# -*- coding: utf-8 -*-
"""Gateway Trace routes and SPA hosting."""
import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from agentica.memory.session_log import SessionLog


@pytest.fixture()
def client_and_svc(tmp_path):
    from agentica.gateway.main import app
    from agentica.gateway import deps
    from agentica.gateway.services.agent_service import AgentService

    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    log = SessionLog(session_id="sess1", base_dir=str(tmp_path))
    log.append_trace_prelude(
        model="gpt-4o-mini", provider="OpenAI", context_window=128000,
        tools=["read_file", "execute"], system_prompt="You are agentica.",
    )
    log.append("user", "hello")
    log.append_event("request_begin")
    log.append_event("text")
    log.append_event("token_usage", request={"input": 10, "cache_read": 0, "cache_write": 0, "output": 4, "total": 14})
    log.append_event("request_end", status="completed")
    log.append("assistant", "hi")
    # `owner` is the signed-in account, which the route reads off the request:
    # the trace of somebody else's session is not one this account may open.
    svc.session_log_for = MagicMock(
        side_effect=lambda sid, owner=None: SessionLog(session_id=sid, base_dir=str(tmp_path))
    )
    svc.list_sessions = MagicMock(return_value=[])

    with TestClient(app, raise_server_exceptions=False) as client:
        original = deps.agent_service
        deps.agent_service = svc
        yield client, svc
        deps.agent_service = original


def test_trace_routes_404_when_session_missing(client_and_svc):
    client, svc = client_and_svc
    assert client.get("/api/sessions/missing/trace/analysis").status_code == 404
    assert client.get("/api/sessions/missing/trace/events").status_code == 404


def test_trace_analysis_and_events(client_and_svc):
    client, _ = client_and_svc
    events = client.get("/api/sessions/sess1/trace/events").json()
    assert events["total"] >= 5
    analysis = client.get("/api/sessions/sess1/trace/analysis").json()
    assert analysis["hasTimeline"] is True
    assert analysis["modelSegments"][0]["kind"] == "text"
    assert analysis["meta"]["model"] == "gpt-4o-mini"
    assert analysis["meta"]["tools"] == ["read_file", "execute"]
    assert analysis["totals"]["rounds"] == 1
    assert analysis["file"]["sizeBytes"] > 0
    round0 = analysis["rounds"][0]
    assert round0["title"] == "hello"
    assert round0["tokens"]["prompt"] == 10
    assert round0["tokens"]["input"] == 10
    assert round0["tokens"]["output"] == 4
    assert round0["tokens"]["cacheHitPercent"] is None
    prompt = next(e for e in round0["entries"] if e["kind"] == "system_prompt")
    assert prompt["detail"] == "You are agentica."
    text = next(e for e in round0["entries"] if e["kind"] == "text")
    assert text["detail"] == "hi"


def test_chat_and_traces_serve_html(client_and_svc):
    client, _ = client_and_svc
    for path in ("/chat", "/traces", "/users"):
        resp = client.get(path)
        assert resp.status_code in (200, 503)
        assert "text/html" in resp.headers["content-type"]


def test_spa_index_explains_how_to_build_when_dist_is_missing(tmp_path, monkeypatch):
    import agentica.gateway.main as gateway_main

    monkeypatch.setattr(gateway_main, "_UI_DIR", tmp_path)
    resp = gateway_main._spa_index()
    assert resp.status_code == 503
    assert b"npm run build" in resp.body
