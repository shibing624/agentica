# -*- coding: utf-8 -*-
"""Gateway Trace routes and SPA hosting."""
import json

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from agentica.memory.session_log import SessionLog


@pytest.fixture()
def client_and_svc(tmp_path):
    from agentica_gateway.main import app
    from agentica_gateway import deps
    from agentica_gateway.services.agent_service import AgentService

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
    assert client.get("/api/sessions/missing/export").status_code == 404


def test_export_downloads_the_session_log_byte_for_byte(client_and_svc, tmp_path):
    client, svc = client_and_svc
    resp = client.get("/api/sessions/sess1/export")
    assert resp.status_code == 200
    assert resp.content == SessionLog(session_id="sess1", base_dir=str(tmp_path)).path.read_bytes()
    assert 'filename="sess1.jsonl"' in resp.headers["content-disposition"]
    # The log comes from the signed-in account's partition, as for the trace.
    assert svc.session_log_for.call_args.kwargs["owner"]


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


def test_trace_lite_analysis_and_call_body(client_and_svc):
    client, _ = client_and_svc
    lite = client.get("/api/sessions/sess1/trace/analysis?lite=true").json()
    model = next(c for c in lite["rounds"][0]["calls"] if c["kind"] == "model")
    assert "body" not in model and model["tokens"]["prompt"] == 10
    body = client.get(f"/api/sessions/sess1/trace/calls/{model['index']}").json()
    assert body["text"] == "hi"
    assert client.get("/api/sessions/sess1/trace/calls/9999").status_code == 404
    found = client.get("/api/sessions/sess1/trace/search", params={"q": "HI"}).json()
    assert found["indices"] == [model["index"]]
    assert client.get("/api/sessions/sess1/trace/search", params={"q": "nope"}).json()["indices"] == []


def test_analysis_can_leave_the_call_lists_out_for_the_chat_footer(client_and_svc):
    client, _ = client_and_svc
    slim = client.get("/api/sessions/sess1/trace/analysis?lite=true&calls=false").json()
    assert all("calls" not in r and "entries" not in r for r in slim["rounds"])
    # What a reply's footer is stamped from is still there.
    assert slim["rounds"][0]["tokens"]["prompt"] == 10 and "durationMs" in slim["rounds"][0]
    assert slim["totals"]["rounds"] == 1 and slim["session_id"] == "sess1"
    # The default keeps the call lists: the Trace page reads them.
    page = client.get("/api/sessions/sess1/trace/analysis?lite=true").json()
    assert page["rounds"][0]["calls"]


def test_events_route_passes_include_turn_through(tmp_path):
    from agentica_gateway.main import app
    from agentica_gateway import deps

    log = SessionLog(session_id="long", base_dir=str(tmp_path))
    for i in range(1, 31):
        log.append("user", f"q{i}")
        log.append("assistant", f"a{i}")
    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    svc.session_log_for = MagicMock(side_effect=lambda sid, owner=None: SessionLog(session_id=sid, base_dir=str(tmp_path)))
    svc.list_sessions = MagicMock(return_value=[])

    def turns(page):
        return [e["content"] for e in page["events"] if e["type"] == "user"]

    with TestClient(app, raise_server_exceptions=False) as client:
        original = deps.agent_service
        deps.agent_service = svc
        try:
            plain = client.get("/api/sessions/long/trace/events", params={"turns": 6}).json()
            assert turns(plain)[0] == "q25"
            deep = client.get("/api/sessions/long/trace/events", params={"turns": 6, "include_turn": 4}).json()
            assert turns(deep)[0] == "q3" and turns(deep)[-1] == "q30" and deep["older_turns"] == 2
            assert client.get("/api/sessions/long/trace/events", params={"turns": 6, "include_turn": 0}).status_code == 422
        finally:
            deps.agent_service = original


def test_session_search_can_be_scoped_to_one_session_with_a_higher_cap(tmp_path):
    from agentica_gateway.main import app
    from agentica_gateway import deps

    for sid in ("a", "b"):
        log = SessionLog(session_id=sid, base_dir=str(tmp_path))
        for i in range(12):
            log.append("user", f"needle question {i}")
            log.append("assistant", f"needle answer {i}")
    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    svc.session_log_for = MagicMock(side_effect=lambda sid, owner=None: SessionLog(session_id=sid, base_dir=str(tmp_path)))
    svc.list_sessions = MagicMock(return_value=[{"session_id": "a"}, {"session_id": "b", "archived": True}])

    with TestClient(app, raise_server_exceptions=False) as client:
        original = deps.agent_service
        deps.agent_service = svc
        try:
            everywhere = client.get("/api/sessions/search", params={"q": "needle"}).json()["results"]
            # Archived sessions are not searched; the rest keep the small default cap.
            assert set(everywhere) == {"a"} and len(everywhere["a"]) == 8
            one = client.get("/api/sessions/search", params={"q": "needle", "session_id": "b", "per_session": 100}).json()["results"]
            # Naming a session searches exactly that one, and lifts the cap.
            assert set(one) == {"b"} and len(one["b"]) == 24
            assert svc.session_log_for.call_args_list[-1].args[0] == "b"
        finally:
            deps.agent_service = original


def test_trace_header_uses_session_work_dir_not_stamped_process_cwd(tmp_path, monkeypatch):
    """A session in /Temp with no git must not show git:main …/Codes/agentica."""
    from agentica_gateway.main import app
    from agentica_gateway import deps

    gateway = tmp_path / "Codes" / "agentica"
    gateway.mkdir(parents=True)
    import subprocess
    subprocess.run(["git", "init", "-b", "main"], cwd=gateway, check=True, capture_output=True)
    work = tmp_path / "Temp"
    work.mkdir()
    monkeypatch.chdir(gateway)

    stale = SessionLog(session_id="sess-temp", base_dir=str(tmp_path / "sessions"))
    stale.append("user", "hello")
    stamped = json.loads(stale.path.read_text(encoding="utf-8").splitlines()[0])
    assert stamped["cwd"] == str(gateway)
    assert stamped["git_branch"] == "main"

    live = SessionLog(
        session_id="sess-temp", base_dir=str(tmp_path / "sessions"), work_dir=str(work),
    )
    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    svc.session_log_for = MagicMock(side_effect=lambda sid, owner=None: live)
    svc.list_sessions = MagicMock(return_value=[])

    with TestClient(app, raise_server_exceptions=False) as client:
        original = deps.agent_service
        deps.agent_service = svc
        try:
            analysis = client.get("/api/sessions/sess-temp/trace/analysis").json()
        finally:
            deps.agent_service = original
    assert analysis["meta"]["cwd"] == str(work.resolve())
    assert analysis["meta"]["gitBranch"] is None


def test_chat_and_traces_serve_html(client_and_svc):
    client, _ = client_and_svc
    for path in ("/chat", "/traces", "/users", "/assistant", "/plugins"):
        resp = client.get(path)
        assert resp.status_code in (200, 503)
        assert "text/html" in resp.headers["content-type"]


def test_spa_index_explains_how_to_build_when_dist_is_missing(tmp_path, monkeypatch):
    import agentica_gateway.main as gateway_main

    monkeypatch.setattr(gateway_main, "_UI_DIR", tmp_path)
    resp = gateway_main._spa_index()
    assert resp.status_code == 503
    assert b"npm run build" in resp.body


def test_trace_events_turn_window_pages_backwards_and_keeps_compact_boundary(client_and_svc, tmp_path):
    client, _ = client_and_svc
    log = SessionLog(session_id="paged", base_dir=str(tmp_path))
    log.append("user", "cron notice", automated=True)
    for i in range(1, 6):
        if i == 4:
            log.append_compact_boundary("", trigger="auto")
        log.append("user", f"q{i}")
        log.append_event("request_begin")
        log.append("assistant", f"a{i}")
        log.append_event("request_end", status="completed")

    p1 = client.get("/api/sessions/paged/trace/events?turns=2").json()
    assert p1["has_more"] is True
    assert p1["older_turns"] == 3
    assert all(e["type"] != "event" for e in p1["events"])
    assert [(e["type"], e.get("content")) for e in p1["events"]] == [
        ("compact_boundary", None),
        ("user", "q4"),
        ("assistant", "a4"),
        ("user", "q5"),
        ("assistant", "a5"),
    ]

    p2 = client.get(f"/api/sessions/paged/trace/events?turns=2&before={p1['next_before']}").json()
    assert p2["has_more"] is True
    assert p2["older_turns"] == 1
    assert [(e["type"], e.get("content")) for e in p2["events"]] == [
        ("user", "q2"),
        ("assistant", "a2"),
        ("user", "q3"),
        ("assistant", "a3"),
    ]

    p3 = client.get(f"/api/sessions/paged/trace/events?turns=2&before={p2['next_before']}").json()
    assert p3["has_more"] is False
    assert p3["older_turns"] == 0
    assert p3["next_before"] == 0
    assert [(e["type"], e.get("content")) for e in p3["events"]] == [
        ("user", "cron notice"),
        ("user", "q1"),
        ("assistant", "a1"),
    ]
