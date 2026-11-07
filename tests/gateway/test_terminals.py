# -*- coding: utf-8 -*-
"""Right-panel terminals: a real shell on a pty, replay on re-attach, admin only."""
import os
import time

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")
if os.name != "posix":
    pytest.skip("terminals need a POSIX pty", allow_module_level=True)

from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from agentica_gateway.main import app
    from agentica_gateway.terminals import manager

    with TestClient(app) as c:
        yield c
        manager.close_all()


def _read_until(ws, needle: str, timeout: float = 10.0) -> str:
    seen = ""
    deadline = time.time() + timeout
    while needle not in seen:
        assert time.time() < deadline, f"never saw {needle!r} in {seen[-400:]!r}"
        event = ws.receive_json()
        if event["type"] in ("snapshot", "output"):
            seen += event["data"]
    return seen


def test_shell_runs_in_the_cwd_and_replays_on_reattach(client, tmp_path):
    made = client.post("/api/terminals", json={"cwd": str(tmp_path), "cols": 100, "rows": 30})
    assert made.status_code == 200, made.text
    term = made.json()
    assert term["cwd"] == str(tmp_path)
    assert term["cols"] == 100 and term["rows"] == 30

    with client.websocket_connect(f"/api/terminals/{term['id']}/stream") as ws:
        ws.send_json({"type": "input", "data": "echo marker-$((40+2)) \"cwd=$(pwd -P)\"\n"})
        # The arithmetic only expands in a real shell, so the echoed command
        # line ("marker-$((40+2))") cannot satisfy this.
        _read_until(ws, f"marker-42 cwd={tmp_path.resolve()}")

    # Detaching does not end the shell: the next attach gets the screen back.
    with client.websocket_connect(f"/api/terminals/{term['id']}/stream") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        assert "marker-42" in first["data"]

    assert [t["id"] for t in client.get("/api/terminals").json()["terminals"]] == [term["id"]]
    assert client.delete(f"/api/terminals/{term['id']}").status_code == 200
    assert client.get("/api/terminals").json()["terminals"] == []


def test_exit_is_reported(client, tmp_path):
    term = client.post("/api/terminals", json={"cwd": str(tmp_path)}).json()
    with client.websocket_connect(f"/api/terminals/{term['id']}/stream") as ws:
        ws.send_json({"type": "input", "data": "exit 3\n"})
        deadline = time.time() + 10
        while True:
            assert time.time() < deadline
            event = ws.receive_json()
            if event["type"] == "exit":
                assert event["code"] == 3
                break


def test_cwd_must_be_a_directory(client, tmp_path):
    resp = client.post("/api/terminals", json={"cwd": str(tmp_path / "missing")})
    assert resp.status_code == 400


def test_cross_site_socket_is_refused(client, tmp_path):
    from starlette.websockets import WebSocketDisconnect

    term = client.post("/api/terminals", json={"cwd": str(tmp_path)}).json()
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(f"/api/terminals/{term['id']}/stream",
                                      headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
    assert excinfo.value.code == 4401


def test_non_admin_is_refused():
    from fastapi import HTTPException
    from agentica_gateway.auth import Principal
    from agentica_gateway.routes.terminals import _admin

    class _Req:
        class state:
            principal = Principal("alice", "session", "user")

    with pytest.raises(HTTPException) as excinfo:
        _admin(_Req())
    assert excinfo.value.status_code == 403
    _Req.state.principal = Principal("default", "session", "admin")
    assert _admin(_Req()) == "default"
