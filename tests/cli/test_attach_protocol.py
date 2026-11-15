# -*- coding: utf-8 -*-
"""Attach protocol and prompt hand-off, as logic — no sockets, no TUI.

The socket-server end-to-end suite was a development acceptance check: it
started a real server per scenario and waited out real settle windows. What
regression needs is the decision logic underneath — token gating, request
validation, where a prompt went and which turn it waits for, and the
pending/answer mapping the cron and IM callers depend on.
"""
import json
import os
import stat
import threading
import time
from types import SimpleNamespace

import pytest

from agentica.cli import attach as attach_mod
from agentica.cli.attach import (
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
    ERR_REFUSED,
    AttachPromptTimeout,
    AttachServer,
    _load_or_create_token,
    prompt_live_session,
    socket_path,
)


def _server(**kwargs):
    defaults = dict(
        inject=lambda text, automated=False, ticket_id=None: "queued",
        session_id="sess-1",
    )
    defaults.update(kwargs)
    server = AttachServer("peer-1", **defaults)
    server._token = "tok-1"
    return server


def _request(server, method, params=None, *, authenticated=True, token="tok-1"):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    if token:
        payload["params"].setdefault("authToken", token)
    response, authed = server._handle_line(json.dumps(payload).encode(), authenticated)
    return response, authed


# --------------------------------------------------------------- token gate


def test_a_request_without_the_token_learns_nothing():
    server = _server()
    response, authed = _request(server, "initialize", token="wrong", authenticated=False)
    assert response["error"]["code"] == ERR_REFUSED
    assert authed is False
    assert "result" not in response


def test_a_valid_token_authenticates_once_and_persists():
    server = _server()
    response, authed = _request(server, "initialize", authenticated=False)
    assert authed is True
    assert "result" in response
    # Second request needs no token: the connection is authenticated.
    response, authed = _request(server, "ping", token="", authenticated=True)
    assert response["result"] == {"status": "ok"}


def test_initialize_advertises_attach_without_claiming_acp_auth_methods():
    server = _server()
    response, _ = _request(server, "initialize")
    result = response["result"]
    assert result["agenticaAuth"] == "token"
    assert result["agentCapabilities"]["agenticaAttach"] is True
    # Advertising an empty authMethods list would say "no auth needed" about a
    # channel that refuses without a token.
    assert "authMethods" not in result


def test_the_token_file_is_private_and_reused(tmp_path):
    path = tmp_path / "token"
    first = _load_or_create_token(path)
    assert first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert _load_or_create_token(path) == first


# --------------------------------------------------------------- validation


def test_unparseable_json_is_reported_not_raised():
    server = _server()
    response, _ = server._handle_line(b"{not json", True)
    assert response["error"]["code"] == ERR_PARSE


def test_an_unknown_method_is_method_not_found():
    server = _server()
    response, _ = _request(server, "session/nope")
    assert response["error"]["code"] == ERR_METHOD_NOT_FOUND


def test_a_prompt_without_text_is_invalid():
    server = _server()
    response, _ = _request(server, "session/prompt", {"prompt": ""})
    assert response["error"]["code"] == ERR_INVALID_PARAMS


def test_session_load_refuses_a_foreign_session_id():
    server = _server()
    response, _ = _request(server, "session/load", {"sessionId": "other"})
    assert response["error"]["code"] == ERR_REFUSED


def test_session_load_returns_the_live_session_and_snapshot():
    server = _server(snapshot=lambda: {"busy": False, "cwd": "/proj"})
    response, _ = _request(server, "session/load", {"sessionId": "sess-1"})
    assert response["result"] == {"sessionId": "sess-1", "busy": False, "cwd": "/proj"}


def test_session_answer_requires_a_non_empty_answer():
    server = _server()
    response, _ = _request(server, "session/answer", {"answer": "  "})
    assert response["error"]["code"] == ERR_INVALID_PARAMS


# ------------------------------------------------------------- prompt logic


def _drive_turn(state, *, start_after=0.02, end_after=0.04):
    """Flip ``state['running']`` the way a turn would: start, then end."""

    def run():
        time.sleep(start_after)
        state["running"] = True
        time.sleep(end_after)
        state["running"] = False

    threading.Thread(target=run, daemon=True).start()


def test_a_queued_prompt_waits_for_its_own_turn_and_reports_the_answer():
    state = {"running": False}
    server = _server(
        is_running=lambda: state["running"],
        answer=lambda: "done",
        grace=2.0,
    )
    _drive_turn(state)

    response, _ = _request(server, "session/prompt", {"prompt": "hi"})

    assert response["result"] == {"stopReason": "end_turn", "agenticaAnswer": "done"}


def test_a_prompt_that_never_gets_a_turn_is_pending_not_an_answer():
    server = _server(is_running=lambda: False, grace=0.05)
    response, _ = _request(server, "session/prompt", {"prompt": "hi"})
    assert response["result"] == {"stopReason": "agentica_pending", "agenticaPending": True}


def test_a_steered_prompt_reports_the_run_it_was_steered_into():
    state = {"running": False}
    server = _server(
        inject=lambda text, automated=False, ticket_id=None: "steered",
        is_running=lambda: state["running"],
        grace=2.0,
        settle=0.0,
    )
    response, _ = _request(server, "session/prompt", {"prompt": "go"})
    assert response["result"]["stopReason"] == "end_turn"


def test_an_injector_that_reports_no_disposition_is_pending():
    server = _server(inject=lambda text, automated=False, ticket_id=None: "")
    response, _ = _request(server, "session/prompt", {"prompt": "hi"})
    assert response["result"] == {"stopReason": "agentica_pending", "agenticaPending": True}


def test_automated_and_ticket_provenance_reach_the_injector():
    seen = {}

    def inject(text, automated=False, ticket_id=None):
        seen.update(text=text, automated=automated, ticket_id=ticket_id)
        return "steered"

    server = _server(inject=inject, settle=0.0)
    _request(
        server,
        "session/prompt",
        {"prompt": "report", "automated": True, "ticketId": "tk-1"},
    )
    assert seen == {"text": "report", "automated": True, "ticket_id": "tk-1"}


# ------------------------------------------------------- client hand-off


class _FakeClient:
    calls = []
    reply: object = {"stopReason": "end_turn"}
    timeout = False

    def __init__(self, path, token, timeout=None):
        self.path, self.token, self.timeout = path, token, timeout

    def call(self, method, params):
        _FakeClient.calls.append((method, params, self.timeout))
        if _FakeClient.timeout:
            raise AttachPromptTimeout("still running")
        return _FakeClient.reply


def _peer(**overrides):
    defaults = dict(attach_socket="/tmp/s.sock", peer_id="peer-1", session_id="sess-1")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def fake_client(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.timeout = False
    _FakeClient.reply = {"stopReason": "end_turn", "agenticaAnswer": "ok"}
    monkeypatch.setattr(attach_mod, "AttachClient", _FakeClient)
    monkeypatch.setattr(attach_mod, "read_attach_token", lambda sock, peer: "tok")
    return _FakeClient


def test_prompt_after_a_write_but_before_an_answer_is_pending(fake_client):
    fake_client.timeout = True
    result = prompt_live_session(_peer(), "hello")
    assert result == {"stopReason": "agentica_pending", "agenticaPending": True}


def test_prompt_without_a_socket_or_token_is_skipped(monkeypatch):
    monkeypatch.setattr(attach_mod, "read_attach_token", lambda sock, peer: None)
    assert prompt_live_session(_peer(), "hello") is None
    assert prompt_live_session(_peer(attach_socket=None), "hello") is None


def test_prompt_carries_provenance_and_session_id(fake_client):
    prompt_live_session(
        _peer(), "nightly", timeout=12.0, automated=True, ticket_id="tk-9", session_id="sess-9"
    )
    method, params, timeout = fake_client.calls[0]
    assert method == "session/prompt"
    assert params == {"prompt": "nightly", "sessionId": "sess-9", "automated": True, "ticketId": "tk-9"}
    assert timeout == 12.0


def test_prompt_answer_is_passed_through(fake_client):
    assert prompt_live_session(_peer(), "hello")["agenticaAnswer"] == "ok"


def test_socket_path_lives_under_the_attach_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTICA_ATTACH_DIR", str(tmp_path))
    path = socket_path("peer-1")
    assert path == tmp_path / "peer-1.sock"
    assert attach_mod.attach_dir() == tmp_path
