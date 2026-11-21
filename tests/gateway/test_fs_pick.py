# -*- coding: utf-8 -*-
"""POST /api/fs/pick: the native "Open folder" dialog for adding a workspace."""
import pytest

pytest.importorskip("agentica_gateway")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentica_gateway.auth import Principal
from agentica_gateway.routes import settings as settings_routes


def _client(host: str = "127.0.0.1", headers: dict | None = None, role: str = "admin") -> TestClient:
    app = FastAPI()
    app.include_router(settings_routes.router)

    @app.middleware("http")
    async def authenticate(request, call_next):
        request.state.principal = Principal("fixture", "session", role)
        return await call_next(request)

    return TestClient(app, client=(host, 50000), headers=headers or {})


def test_returns_the_folder_the_dialog_chose(monkeypatch, tmp_path):
    monkeypatch.setattr(settings_routes, "_pick_command", lambda platform=None: ["chooser"])
    seen = []
    monkeypatch.setattr(settings_routes, "_pick_directory_os", lambda cmd: seen.append(cmd) or str(tmp_path))
    resp = _client().post("/api/fs/pick")
    assert resp.status_code == 200
    assert resp.json() == {"path": str(tmp_path)}
    assert seen == [["chooser"]]


def test_cancel_is_a_null_path_not_an_error(monkeypatch):
    monkeypatch.setattr(settings_routes, "_pick_command", lambda platform=None: ["chooser"])
    monkeypatch.setattr(settings_routes, "_pick_directory_os", lambda cmd: None)
    resp = _client().post("/api/fs/pick")
    assert resp.status_code == 200
    assert resp.json() == {"path": None}


@pytest.mark.parametrize("host,headers", [
    ("192.168.1.20", None),
    ("127.0.0.1", {"X-Forwarded-For": "203.0.113.9"}),
])
def test_remote_client_never_opens_a_dialog(monkeypatch, host, headers):
    monkeypatch.setattr(settings_routes, "_pick_command", lambda platform=None: ["chooser"])
    opened = []
    monkeypatch.setattr(settings_routes, "_pick_directory_os", lambda cmd: opened.append(cmd))
    resp = _client(host, headers).post("/api/fs/pick")
    assert resp.status_code == 409
    assert opened == []


def test_no_chooser_on_this_system_is_501(monkeypatch):
    monkeypatch.setattr(settings_routes, "_pick_command", lambda platform=None: None)
    assert _client().post("/api/fs/pick").status_code == 501


def test_local_non_admin_does_not_open_a_dialog(monkeypatch):
    opened = []
    monkeypatch.setattr(settings_routes, "_pick_command", lambda: ["chooser"])
    monkeypatch.setattr(settings_routes, "_pick_directory_os", lambda cmd: opened.append(cmd))
    assert _client(role="user").post("/api/fs/pick").status_code == 403
    assert opened == []


def test_pick_command_per_platform(monkeypatch):
    mac = settings_routes._pick_command("darwin")
    assert mac[0] == "osascript" and "choose folder" in mac[-1]
    monkeypatch.setattr(settings_routes.shutil, "which", lambda name: "/usr/bin/zenity" if name == "zenity" else None)
    assert settings_routes._pick_command("linux")[0] == "zenity"
    monkeypatch.setattr(settings_routes.shutil, "which", lambda name: None)
    assert settings_routes._pick_command("linux") is None
    assert settings_routes._pick_command("win32")[0] == "powershell"


class _Done:
    def __init__(self, returncode, stdout):
        self.returncode, self.stdout = returncode, stdout


def test_dialog_output_is_a_path_and_cancel_or_failure_is_none(monkeypatch, opened_externally):
    run = settings_routes.subprocess
    monkeypatch.setattr(settings_routes, "_pick_directory_os", settings_routes._pick_directory_os.real)
    monkeypatch.setattr(run, "run", lambda *a, **k: _Done(0, "/Users/me/proj/\n"))
    assert settings_routes._pick_directory_os(["x"]) == "/Users/me/proj"
    # osascript exits 1 with "User canceled" on stderr; zenity exits 1 too.
    monkeypatch.setattr(run, "run", lambda *a, **k: _Done(1, "/ignored\n"))
    assert settings_routes._pick_directory_os(["x"]) is None
    monkeypatch.setattr(run, "run", lambda *a, **k: _Done(0, "\n"))
    assert settings_routes._pick_directory_os(["x"]) is None
