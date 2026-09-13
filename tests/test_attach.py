# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The attach server — an external program sending a user message into
a running session over a unix socket, JSON-RPC / ACP shaped.

A real socket and a real client every time: the framing, the auth gate and the
"one socket per session" addressing are the whole contract, and a fake transport
would test the fake.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import pytest

from agentica import attach, peers
from agentica.attach import (
    ERR_METHOD_NOT_FOUND,
    ERR_PARSE,
    ERR_REFUSED,
    AttachServer,
)


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """Keep sockets and tokens out of the developer's real attach dir.

    Via the documented override rather than by patching a module global: the
    path has to stay short (``AF_UNIX``), so tests get their own directory
    instead of the deep pytest tmp dir.
    """
    import shutil
    import tempfile

    # Not tmp_path: pytest's per-test directory is long enough that the socket
    # path exceeds AF_UNIX's sun_path limit, which is the very trap the override
    # exists for. mkdtemp in the short system temp dir keeps it honest.
    short = tempfile.mkdtemp(prefix="ag-attach-")
    monkeypatch.setenv("AGENTICA_ATTACH_DIR", short)
    monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path))
    yield tmp_path
    shutil.rmtree(short, ignore_errors=True)


class _Client:
    """A minimal line-delimited JSON-RPC client, i.e. what a desktop app is."""

    def __init__(self, path, token=None):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(10)
        self._sock.connect(str(path))
        self._buf = b""
        if token is not None:
            self.call("initialize", {"authToken": token, "clientCapabilities": {}})

    def call(self, method, params=None, request_id=1):
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._sock.sendall(json.dumps(payload).encode() + b"\n")
        return self._read()

    def send_raw(self, raw: bytes):
        self._sock.sendall(raw)
        return self._read()

    def _read(self):
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise AssertionError("server closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line.decode())

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


def _server(tmp_path, **kw):
    """A server with a recording injector; defaults to an idle session."""
    injected = kw.pop("injected", None)
    if injected is None:
        injected = []
    running = kw.pop("running", None)
    if running is None:
        running = _FlipFlop()

    def inject(text):
        injected.append(text)
        running.poke()

    server = AttachServer(
        kw.pop("peer_id", "abcd1234"),
        inject=inject,
        is_running=running.is_running,
        session_id=kw.pop("session_id", "sess-1234"),
        **kw,
    )
    assert server.start() is True, "the server must be listening"
    return server, injected, running


class _FlipFlop:
    """Stands in for the session's run state: poke() means "a turn is happening"."""

    def __init__(self, start_running=False):
        self._running = start_running
        self._lock = threading.Lock()

    def poke(self, seconds=0.15):
        with self._lock:
            self._running = True

        def _end():
            time.sleep(seconds)
            with self._lock:
                self._running = False

        threading.Thread(target=_end, daemon=True).start()

    def is_running(self):
        with self._lock:
            return self._running


class TestHandshake:
    def test_a_valid_token_is_accepted(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call("ping")
            assert result["result"] == {"status": "ok"}
            client.close()
        finally:
            server.stop()

    def test_no_token_is_refused(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path)
            reply = client.call("initialize", {"clientCapabilities": {}})
            assert reply["error"]["code"] == ERR_REFUSED
            assert "authToken" in reply["error"]["message"]
            client.close()
        finally:
            server.stop()

    def test_a_wrong_token_is_refused(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path)
            reply = client.call("initialize", {"authToken": "not-the-token"})
            assert reply["error"]["code"] == ERR_REFUSED
            client.close()
        finally:
            server.stop()

    def test_an_unauthenticated_client_learns_nothing(self, tmp_path):
        """Even the method list is gated: no token, no information."""
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path)
            reply = client.call("session/load", {"sessionId": "sess-1234"})
            assert reply["error"]["code"] == ERR_REFUSED
            client.close()
        finally:
            server.stop()

    def test_initialize_reports_the_protocol(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call("initialize", {"clientCapabilities": {}})["result"]
            assert result["protocolVersion"] == attach.PROTOCOL_VERSION
            assert result["agentInfo"]["name"] == "agentica"
            client.close()
        finally:
            server.stop()


class TestAddressing:
    """One socket per session, so connecting *is* naming the session."""

    def test_the_socket_is_under_a_private_dir(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            assert server.path.exists()
            assert (attach.attach_dir().stat().st_mode & 0o777) == 0o700
            assert (server.path.stat().st_mode & 0o777) == 0o600
            assert (server.token_file.stat().st_mode & 0o777) == 0o600
        finally:
            server.stop()

    def test_two_sessions_get_two_sockets(self, tmp_path):
        a = AttachServer("aaaaaaaa", inject=lambda t: None, is_running=lambda: False)
        b = AttachServer("bbbbbbbb", inject=lambda t: None, is_running=lambda: False)
        assert a.start() and b.start()
        try:
            assert a.path != b.path
            assert a.path.exists() and b.path.exists()
        finally:
            a.stop()
            b.stop()

    def test_a_token_left_by_a_crashed_session_is_reused_not_replaced(self, tmp_path):
        """A client holding the token must not be locked out by a restart.

        ``stop()`` removes the token because the session is gone; a crash does
        not get to run ``stop()``, so the next start finds the file and must
        adopt it rather than minting a new one behind the client's back.
        """
        server, _, _ = _server(tmp_path)
        token_file = server.token_file
        original = token_file.read_text().strip()
        # Simulate a crash: drop the listener without the cleanup stop() does.
        server._sock.close()
        server._sock = None
        assert token_file.exists()

        server2, _, _ = _server(tmp_path)
        try:
            assert server2.token_file.read_text().strip() == original
            client = _Client(server2.path, token=original)
            assert client.call("ping")["result"] == {"status": "ok"}
            client.close()
        finally:
            server2.stop()

    def test_session_load_rejects_a_mismatched_session_id(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call("session/load", {"sessionId": "some-other-session"})
            assert reply["error"]["code"] == ERR_REFUSED
            assert "sess-1234" in reply["error"]["message"]
            client.close()
        finally:
            server.stop()

    def test_session_load_returns_the_attached_session(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call("session/load", {"sessionId": "sess-1234"})["result"]
            assert result["sessionId"] == "sess-1234"
            client.close()
        finally:
            server.stop()


class TestPrompt:
    def test_the_text_reaches_the_agent(self, tmp_path):
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "run the tests"}]},
            )
            assert "error" not in reply, reply
            assert injected == ["run the tests"]
            client.close()
        finally:
            server.stop()

    def test_the_stop_reason_is_reported(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "hi"}]},
            )["result"]
            assert result["stopReason"] == "end_turn"
            client.close()
        finally:
            server.stop()

    def test_an_idle_session_waits_for_the_turn_it_started(self, tmp_path):
        """The injected line becomes the next turn; the reply must not come back
        before that turn has run, or the client would report an answer that has
        not been produced yet."""
        server, _, running = _server(tmp_path, running=_FlipFlop(start_running=False))
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            started = time.monotonic()
            client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "hi"}]},
            )
            assert time.monotonic() - started >= 0.1
            client.close()
        finally:
            server.stop()

    def test_a_running_session_is_steered_not_queued(self, tmp_path):
        running = _FlipFlop(start_running=True)
        server, injected, _ = _server(tmp_path, running=running)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())

            def _finish():
                time.sleep(0.2)
                running.poke(0.05)

            threading.Thread(target=_finish, daemon=True).start()
            reply = client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "stop"}]},
            )
            assert "error" not in reply, reply
            assert injected == ["stop"]
            client.close()
        finally:
            server.stop()

    def test_an_empty_prompt_is_refused(self, tmp_path):
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "  "}]}
            )
            assert reply["error"]["code"] == -32602
            assert injected == []
            client.close()
        finally:
            server.stop()

    def test_a_non_text_block_is_not_silently_dropped(self, tmp_path):
        """An image the user attached must not turn into a different request."""
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "image", "uri": "file:///x.png"}]},
            )
            assert reply["error"]["code"] == -32602
            assert injected == []
            client.close()
        finally:
            server.stop()

    def test_a_mismatched_session_id_cannot_prompt_this_session(self, tmp_path):
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt",
                {"sessionId": "other", "prompt": [{"type": "text", "text": "hi"}]},
            )
            assert reply["error"]["code"] == ERR_REFUSED
            assert injected == []
            client.close()
        finally:
            server.stop()

    def test_the_answer_can_be_reported_back(self, tmp_path):
        """Optional: a host that can read the last answer may report it."""
        server, _, _ = _server(tmp_path, answer=lambda: "the final answer")
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "hi"}]},
            )["result"]
            assert result["agenticaAnswer"] == "the final answer"
            client.close()
        finally:
            server.stop()


class TestRobustness:
    """A client must never be able to take the session down or wedge it."""

    def test_malformed_json_gets_a_parse_error(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.send_raw(b"this is not json\n")
            assert reply["error"]["code"] == ERR_PARSE
            # The connection stays usable.
            assert client.call("ping")["result"] == {"status": "ok"}
            client.close()
        finally:
            server.stop()

    def test_an_unknown_method_is_reported(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call("session/new", {"cwd": "/tmp"})
            assert reply["error"]["code"] == ERR_METHOD_NOT_FOUND
            client.close()
        finally:
            server.stop()

    def test_a_missing_method_field_is_reported(self, tmp_path):
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.send_raw(json.dumps({"jsonrpc": "2.0", "id": 3}).encode() + b"\n")
            assert reply["error"]["code"] == -32600
            client.close()
        finally:
            server.stop()

    def test_an_injector_that_raises_is_reported_not_fatal(self, tmp_path):
        def boom(text):
            raise RuntimeError("session is gone")

        server = AttachServer("abcd1234", inject=boom, is_running=lambda: False,
                              session_id="sess-1234")
        assert server.start()
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "x"}]}
            )
            assert reply["error"]["code"] == -32603
            assert client.call("ping")["result"] == {"status": "ok"}
            client.close()
        finally:
            server.stop()

    def test_a_client_that_vanishes_does_not_stop_the_server(self, tmp_path):
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            client.close()
            time.sleep(0.2)
            # A second client still works.
            client2 = _Client(server.path, token=server.token_file.read_text().strip())
            client2.call(
                "session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "after"}]}
            )
            assert injected == ["after"]
            client2.close()
        finally:
            server.stop()

    def test_several_messages_on_one_connection(self, tmp_path):
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            for n in range(3):
                client.call(
                    "session/prompt",
                    {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": f"m{n}"}]},
                    request_id=n + 10,
                )
            assert injected == ["m0", "m1", "m2"]
            client.close()
        finally:
            server.stop()

    def test_two_clients_can_both_prompt(self, tmp_path):
        server, injected, _ = _server(tmp_path)
        try:
            token = server.token_file.read_text().strip()
            a = _Client(server.path, token=token)
            b = _Client(server.path, token=token)
            a.call("session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "a"}]})
            b.call("session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "b"}]})
            assert injected == ["a", "b"]
            a.close()
            b.close()
        finally:
            server.stop()


class TestLifecycle:
    def test_disabled_by_default(self, monkeypatch):
        """Off unless switched on: this channel's effect is the user speaking."""
        monkeypatch.delenv("AGENTICA_ATTACH_ENABLED", raising=False)
        assert attach.attach_enabled({}) is False
        assert attach.attach_enabled({"settings": {"attach_enabled": False}}) is False

    def test_enabled_by_config(self, monkeypatch):
        monkeypatch.delenv("AGENTICA_ATTACH_ENABLED", raising=False)
        assert attach.attach_enabled({"settings": {"attach_enabled": True}}) is True

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_ATTACH_ENABLED", "1")
        assert attach.attach_enabled({"settings": {"attach_enabled": False}}) is True
        monkeypatch.setenv("AGENTICA_ATTACH_ENABLED", "0")
        assert attach.attach_enabled({"settings": {"attach_enabled": True}}) is False

    def test_a_broken_settings_block_is_off_not_a_crash(self, monkeypatch):
        monkeypatch.delenv("AGENTICA_ATTACH_ENABLED", raising=False)
        # `settings` itself not being a mapping: get_setting returns the default,
        # which is what the rest of the CLI relies on too.
        assert attach.attach_enabled({"settings": "nonsense"}) is False
        assert attach.attach_enabled({}) is False

    def test_stop_removes_the_socket_and_the_token(self, tmp_path):
        server, _, _ = _server(tmp_path)
        path, token = server.path, server.token_file
        assert path.exists() and token.exists()
        server.stop()
        assert not path.exists()
        # The token goes with the socket: it must not survive to authenticate
        # against a session that is gone.
        assert not token.exists()

    def test_a_stale_socket_file_does_not_block_a_restart(self, tmp_path):
        server, _, _ = _server(tmp_path)
        path = server.path
        server.stop()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale")  # a crashed process left something behind

        server2, _, _ = _server(tmp_path)
        try:
            assert server2.listening
            client = _Client(server2.path, token=server2.token_file.read_text().strip())
            assert client.call("ping")["result"] == {"status": "ok"}
            client.close()
        finally:
            server2.stop()

    def test_no_peer_id_means_no_server(self, tmp_path):
        server = AttachServer("", inject=lambda t: None)
        assert server.start() is False
        assert server.listening is False

    def test_discovery_paths_are_derived_from_the_peer_id(self, tmp_path):
        server, _, _ = _server(tmp_path, peer_id="deadbeef")
        try:
            assert attach.socket_path("deadbeef") == server.path
            assert server.path.name == "deadbeef.sock"
        finally:
            server.stop()


class TestPromptText:
    """ACP sends content blocks; this channel is text-only and says so."""

    def test_text_blocks_are_joined(self):
        assert attach._prompt_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"

    def test_a_bare_string_is_accepted(self):
        assert attach._prompt_text("  hello  ") == "hello"

    def test_nothing_usable_gives_an_empty_string(self):
        assert attach._prompt_text([{"type": "image", "uri": "x"}]) == ""
        assert attach._prompt_text(None) == ""
        assert attach._prompt_text(123) == ""


class TestTheIncomingLineIsEchoed:
    """Relayed input is not echoed by the run itself, so the host must echo it.

    Without this the terminal shows an answer to a question that is nowhere on
    screen — found by scripts/verify_attach_e2e.py, which refused to call the run
    verified until the relayed line was visible in the pane.
    """

    def test_it_renders_the_text_through_the_shared_panel(self, capsys):
        from agentica.cli.display.messages import display_attached_user_message

        display_attached_user_message("run the tests")

        out = capsys.readouterr().out
        assert "run the tests" in out

    def test_it_can_name_where_the_line_came_from(self, capsys):
        from agentica.cli.display.messages import display_attached_user_message

        display_attached_user_message("hi", from_name="vpet-desktop")

        assert "vpet-desktop" in capsys.readouterr().out
