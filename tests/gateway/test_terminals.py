# -*- coding: utf-8 -*-
"""Right-panel terminals as logic: the cwd/size the shell is started with, the
bytes routed between socket and shell, replay on re-attach, exit reporting,
and the admin/origin gates.

The old suite allocated a real pseudo-terminal and ran a real login shell for
every scenario. That is environment risk (a pty may not exist at all under a
sandbox) and end-to-end acceptance more than logic, so the manager's spawn
seam — the pty pair plus the process factory — is replaced here, test-side
only, with an in-memory shell. The real shell is exercised by using the
product.
"""
import os
import socket
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")


def test_gateway_imports_without_posix_tty_modules():
    """Windows has no fcntl/pty/termios. Importing the gateway must still succeed."""
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as home:
        env = os.environ.copy()
        env["AGENTICA_HOME"] = home
        env["AGENTICA_CACHE_DIR"] = os.path.join(home, "cache")
        script = (
            "import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name in {'fcntl', 'pty', 'termios'}:\n"
            "            raise ModuleNotFoundError(name)\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            "import agentica_gateway.main\n"
            "import agentica_gateway.terminals as terminals\n"
            "assert terminals.pty is None\n"
        )
        proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


if os.name != "posix":
    pytest.skip("terminals need a POSIX pty", allow_module_level=True)

from fastapi.testclient import TestClient


class FakeShell:
    """One in-memory shell attached to the fake terminal pair.

    ``feed`` writes bytes the manager's reader will see; ``finish`` closes the
    peer so the manager observes EOF and reports the exit code.
    """

    def __init__(self, argv, cwd, peer_fd):
        self.argv = list(argv)
        self.cwd = cwd
        self.env = None
        self.pid = 4242
        self.code = None
        self._fd = os.dup(peer_fd)
        os.set_blocking(self._fd, False)

    # -- what the routes may observe --------------------------------------
    def feed(self, text):
        os.write(self._fd, text.encode())

    def finish(self, code):
        self.code = code
        self.close()

    def received(self, timeout=2.0):
        """Bytes the manager wrote to the shell, or ``""`` if none arrived."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                data = os.read(self._fd, 65536)
            except BlockingIOError:
                data = b""
            if data:
                return data.decode()
            time.sleep(0.005)
        return ""

    def close(self):
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    # -- the process interface the manager uses ---------------------------
    def wait(self, timeout=None):
        return self.code if self.code is not None else -1

    def kill(self):
        self.finish(-9)


class FakeSpawner:
    """The spawn seam: a socket pair for the pty, a fake shell for the child."""

    def __init__(self):
        self.shell = None
        self.size = None
        self.signals = []

    def openpty(self):
        left, right = socket.socketpair()
        try:
            return os.dup(left.fileno()), os.dup(right.fileno())
        finally:
            left.close()
            right.close()

    def set_size(self, fd, cols, rows):
        if self.size is None:
            self.size = (cols, rows)

    def popen(self, argv, stdin=None, stdout=None, stderr=None, cwd=None, env=None, **kwargs):
        self.shell = FakeShell(argv, cwd, stdin)
        self.shell.env = env
        return self.shell

    def close(self):
        if self.shell is not None:
            self.shell.close()


class _FakeOs:
    """``terminals.os`` with process-group signals recorded, not delivered."""

    name = os.name

    def __init__(self, spawner):
        self._spawner = spawner

    def __getattr__(self, name):
        return getattr(os, name)

    def getpgid(self, pid):
        return pid

    def killpg(self, pgid, sig):
        self._spawner.signals.append((pgid, sig))


@pytest.fixture()
def fake_spawn(monkeypatch):
    from agentica_gateway import terminals

    spawner = FakeSpawner()
    monkeypatch.setattr(terminals, "pty", SimpleNamespace(openpty=spawner.openpty))
    monkeypatch.setattr(terminals, "subprocess", SimpleNamespace(Popen=spawner.popen))
    monkeypatch.setattr(terminals, "_set_size", spawner.set_size)
    monkeypatch.setattr(terminals, "os", _FakeOs(spawner))
    yield spawner
    spawner.close()


@pytest.fixture()
def client(fake_spawn):
    from agentica_gateway.main import app
    from agentica_gateway.terminals import manager

    with TestClient(app) as c:
        yield c
        manager.close_all()


def _receive(ws, kind: str, timeout: float = 2.0) -> dict:
    """Next websocket event of ``kind``, skipping the attach snapshot."""
    deadline = time.time() + timeout
    while True:
        assert time.time() < deadline, f"no {kind!r} event"
        event = ws.receive_json()
        if event["type"] == kind:
            return event


def test_shell_starts_in_the_cwd_and_replays_on_reattach(client, fake_spawn, tmp_path):
    made = client.post("/api/terminals", json={"cwd": str(tmp_path), "cols": 100, "rows": 30})
    assert made.status_code == 200, made.text
    term = made.json()
    shell = fake_spawn.shell
    assert term["cwd"] == str(tmp_path)
    assert (term["cols"], term["rows"]) == (100, 30)
    # The shell is started in the requested directory, as a login shell.
    assert shell.cwd == str(tmp_path)
    assert shell.argv[0].startswith("/") and shell.argv[1:] == ["-l"]
    assert fake_spawn.size == (100, 30)
    assert shell.env["TERM"] == "xterm-256color"

    with client.websocket_connect(f"/api/terminals/{term['id']}/stream") as ws:
        ws.send_json({"type": "input", "data": "echo marker-42\n"})
        assert shell.received() == "echo marker-42\n"
        shell.feed("marker-42\r\n")
        assert "marker-42" in _receive(ws, "output")["data"]

    # Detaching does not end the shell: the next attach gets the screen back.
    with client.websocket_connect(f"/api/terminals/{term['id']}/stream") as ws:
        first = ws.receive_json()
        assert first["type"] == "snapshot"
        assert "marker-42" in first["data"]

    assert [t["id"] for t in client.get("/api/terminals").json()["terminals"]] == [term["id"]]
    assert client.delete(f"/api/terminals/{term['id']}").status_code == 200
    assert client.get("/api/terminals").json()["terminals"] == []


def test_exit_is_reported(client, fake_spawn, tmp_path):
    term = client.post("/api/terminals", json={"cwd": str(tmp_path)}).json()
    with client.websocket_connect(f"/api/terminals/{term['id']}/stream") as ws:
        ws.send_json({"type": "input", "data": "exit 3\n"})
        assert fake_spawn.shell.received() == "exit 3\n"
        fake_spawn.shell.finish(3)
        assert _receive(ws, "exit")["code"] == 3

    listed = client.get("/api/terminals").json()["terminals"]
    assert listed[0]["exited"] is True and listed[0]["exit_code"] == 3


def test_cwd_must_be_a_directory(client, fake_spawn, tmp_path):
    resp = client.post("/api/terminals", json={"cwd": str(tmp_path / "missing")})
    assert resp.status_code == 400
    # Rejected before the shell was spawned.
    assert fake_spawn.shell is None


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
