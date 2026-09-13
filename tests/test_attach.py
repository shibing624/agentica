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
    """A server whose injector records the text and reports where it went.

    **The default injector does not start a turn.** That is the whole point: a
    fixture that poked the run state on every injection manufactured a turn right
    where the interesting case has none, so the ``queued``-while-running bug (the
    run in flight will never see this text) passed no matter what the wait did.
    Tests script turns explicitly through ``session``.
    """
    injected = kw.pop("injected", None)
    if injected is None:
        injected = []
    session = kw.pop("session", None)
    if session is None:
        session = _Session()
    disposition = kw.pop("disposition", "queued")

    def inject(text):
        injected.append(text)
        return disposition

    server = AttachServer(
        kw.pop("peer_id", "abcd1234"),
        inject=inject,
        is_running=session.is_running,
        session_id=kw.pop("session_id", "sess-1234"),
        **kw,
    )
    assert server.start() is True, "the server must be listening"
    return server, injected, session


def _run_one_turn_soon(session: "_Session", *, delay: float = 0.1,
                       duration: float = 0.15) -> None:
    """Script the turn a ``session/prompt`` will become.

    Needed by every test that expects a *completed* prompt: the default injector
    no longer invents a turn, so a test that wants one has to say so.
    """
    session.start_after(delay, duration)


class _Session:
    """The CLI's run state, scripted by the test.

    ``start`` / ``end`` are the test driving turns; ``is_running`` is what the
    server polls. Kept explicit so a test can reproduce "a run is finishing and
    this text did *not* go into it".
    """

    def __init__(self, start_running: bool = False):
        self._running = start_running
        self._lock = threading.Lock()
        self.turns_started = 0

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        with self._lock:
            self._running = True
            self.turns_started += 1

    def end(self) -> None:
        with self._lock:
            self._running = False

    def run_for(self, seconds: float) -> None:
        """Start a turn that ends itself after ``seconds``."""
        self.start()

        def _end():
            time.sleep(seconds)
            self.end()

        threading.Thread(target=_end, daemon=True).start()

    def start_after(self, delay: float, seconds: float) -> None:
        """Start a turn after ``delay``, running for ``seconds``."""
        def _go():
            time.sleep(delay)
            self.run_for(seconds)

        threading.Thread(target=_go, daemon=True).start()


class TestHandshake:
    def test_no_auth_methods_are_advertised(self, tmp_path):
        """Advertising ``authMethods: []`` would tell a spec-aware ACP client
        "no auth needed" about a channel that refuses without a token."""
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call("initialize", {"clientCapabilities": {}})["result"]
            assert "authMethods" not in result
            assert result["agenticaAuth"] == "token"
            client.close()
        finally:
            server.stop()

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

    def test_a_resumed_session_is_tracked_not_frozen(self, tmp_path):
        """``/resume`` swaps the session under a running CLI.

        The id must be read per request: a snapshot taken at construction would
        reject the id the client just read from the presence record.
        """
        current = {"session": "before-resume"}
        server = AttachServer(
            "abcd1234",
            inject=lambda t: "queued",
            is_running=lambda: False,
            session_id=lambda: current["session"],
        )
        assert server.start()
        try:
            token = server.token_file.read_text().strip()
            client = _Client(server.path, token=token)

            assert client.call("session/load", {})["result"]["sessionId"] == "before-resume"

            current["session"] = "after-resume"
            assert client.call("session/load", {})["result"]["sessionId"] == "after-resume"
            # And the new id is accepted, the old one is now the mismatch.
            assert "error" not in client.call("session/load", {"sessionId": "after-resume"})
            assert client.call("session/load", {"sessionId": "before-resume"})["error"]["code"] == (
                ERR_REFUSED
            )
            client.close()
        finally:
            server.stop()


class TestPrompt:
    def test_a_steered_prompt_waits_for_the_run_in_flight(self, tmp_path):
        """Steered: the run already going is the one carrying this text."""
        session = _Session(start_running=True)
        server, injected, _ = _server(tmp_path, session=session, disposition="steered")
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())

            def _end():
                time.sleep(0.25)
                session.end()

            threading.Thread(target=_end, daemon=True).start()
            reply = client.call(
                "session/prompt",
                {"prompt": [{"type": "text", "text": "keep going"}]},
            )
            assert "error" not in reply, reply
            assert injected == ["keep going"]
            client.close()
        finally:
            server.stop()

    def test_a_queued_prompt_waits_for_its_own_turn(self, tmp_path):
        """The bug this pins, and why the earlier test could not catch it.

        Text queued while a run is still finishing: that run will never see it, so
        the prompt must outlive it and wait for the *next* turn. The assertion is
        on the script, not on elapsed time — "it took at least N seconds" is
        satisfied by any slow machine, which is how this stayed green while the
        wait still returned on the wrong run's end.
        """
        session = _Session(start_running=True)
        # ``settle`` is set below the gap between the two turns on purpose. At the
        # default settle the window happened to span the gap, so the next turn
        # starting inside it made the old logic look correct — the test passed with
        # the queued phase deleted. A gap wider than the window is what makes the
        # phase load-bearing.
        server, _, _ = _server(tmp_path, session=session, disposition="queued",
                               settle=0.2)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            events = []

            def _script():
                time.sleep(0.2)
                session.end()            # the run that does NOT carry it ends
                events.append("current_ended")
                time.sleep(0.6)          # longer than the settle window
                session.start()          # the queued turn starts
                events.append("next_started")
                time.sleep(0.3)
                session.end()
                events.append("next_ended")

            threading.Thread(target=_script, daemon=True).start()
            reply = client.call(
                "session/prompt",
                {"prompt": [{"type": "text", "text": "after this run"}]},
            )
            assert "error" not in reply, reply
            # The reply must not have come before its own turn ran. This is the
            # assertion the old test was missing: the server returns only after
            # the second turn has started *and* ended.
            assert events == ["current_ended", "next_started", "next_ended"], (
                f"returned with {events!r} — it settled on a run that never carried "
                f"this message"
            )
            client.close()
        finally:
            server.stop()

    def test_a_queued_prompt_reports_pending_when_its_turn_never_starts(self, tmp_path):
        """Queued and the session never runs it (it is shutting down): that is
        ``agenticaPending``, not a completion, and not a 600s silent hang."""
        session = _Session(start_running=True)
        server, _, _ = _server(tmp_path, session=session, disposition="queued",
                               grace=0.6, settle=0.0)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())

            def _script():
                time.sleep(0.15)
                session.end()  # and nothing ever starts

            threading.Thread(target=_script, daemon=True).start()
            started = time.monotonic()
            reply = client.call(
                "session/prompt",
                {"prompt": [{"type": "text", "text": "into the void"}]},
            )
            elapsed = time.monotonic() - started
            assert reply["result"].get("agenticaPending") is True, reply
            assert elapsed < 5, f"waited {elapsed:.1f}s instead of giving up"
            client.close()
        finally:
            server.stop()

    def test_an_injector_reporting_nothing_is_not_claimed_as_complete(self, tmp_path):
        server, _, _ = _server(tmp_path, disposition="")
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call(
                "session/prompt", {"prompt": [{"type": "text", "text": "x"}]}
            )["result"]
            assert result.get("agenticaPending") is True
            client.close()
        finally:
            server.stop()

    def test_a_cancelled_turn_reports_cancelled_not_end_turn(self, tmp_path):
        """The docs promise a caller can tell "interrupted" from "finished".

        Without this, ``session/cancel`` was only ``Agent.cancel()``: the run ended,
        the prompt returned ``end_turn``, and the two outcomes were indistinguish-
        able to a client. This test did not exist, which is why that could ship.
        """
        session = _Session(start_running=True)
        server, _, _ = _server(tmp_path, session=session, disposition="steered",
                               grace=5.0, settle=0.0, cancel=lambda: None)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())

            def _cancel_soon():
                time.sleep(0.3)
                # As ``session/cancel`` does, on its own connection: the server
                # holds one prompt at a time, so a client cannot send it while its
                # prompt is blocked.
                canceller = _Client(server.path, token=server.token_file.read_text().strip())
                canceller.call("session/cancel", {})
                canceller.close()
                time.sleep(0.2)
                session.end()  # the run stops because of the cancel

            threading.Thread(target=_cancel_soon, daemon=True).start()
            result = client.call(
                "session/prompt",
                {"prompt": [{"type": "text", "text": "long job"}]},
                request_id=1,
            )["result"]
            assert result.get("stopReason") == "cancelled", result
            client.close()
        finally:
            server.stop()

    def test_a_cancel_that_ends_the_run_shortens_the_wait(self, tmp_path):
        """A cancel ends the turn it names, so the wait must not sit out the grace.

        The run is left *running* here on purpose: that is the only version of this
        test that can fail. If the script also ends the run, the wait returns on its
        own and the cancel check is never load-bearing — it passed with that check
        disabled before this was changed.
        """
        session = _Session(start_running=True)
        server, _, _ = _server(tmp_path, session=session, disposition="steered",
                               grace=8.0, settle=0.0)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())

            def _cancel():
                time.sleep(0.3)
                server._cancel_current({})  # as session/cancel would
                # deliberately no session.end(): the run keeps going, so only the
                # cancel can end the wait.

            threading.Thread(target=_cancel, daemon=True).start()
            started = time.monotonic()
            result = client.call(
                "session/prompt", {"prompt": [{"type": "text", "text": "job"}]}
            )["result"]
            elapsed = time.monotonic() - started
            assert result.get("stopReason") == "cancelled", result
            assert elapsed < 3, f"waited {elapsed:.1f}s despite the cancel"
            client.close()
        finally:
            server.stop()

        server, _, _ = _server(tmp_path, disposition="")
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            result = client.call(
                "session/prompt", {"prompt": [{"type": "text", "text": "x"}]}
            )["result"]
            assert result.get("agenticaPending") is True
            client.close()
        finally:
            server.stop()

    def test_the_text_reaches_the_agent(self, tmp_path):
        """A turn must actually happen for this to be a round trip at all."""
        session = _Session()
        server, injected, _ = _server(tmp_path, session=session, disposition="queued",
                                      settle=0.0)
        session.start_after(0.1, 0.15)
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
        session = _Session()
        server, _, _ = _server(tmp_path, session=session, settle=0.0)
        session.start_after(0.1, 0.15)
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
        not been produced yet.

        Asserted on the script, not on elapsed time: a duration alone passes on a
        slow machine while the wait still settled on the wrong run.
        """
        session = _Session()
        server, _, _ = _server(tmp_path, session=session, settle=0.0)
        finished = []

        def _script():
            time.sleep(0.25)
            session.start()

            def _end():
                time.sleep(0.25)
                session.end()
                finished.append("turn_done")

            threading.Thread(target=_end, daemon=True).start()

        threading.Thread(target=_script, daemon=True).start()
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            client.call(
                "session/prompt",
                {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "hi"}]},
            )
            assert finished == ["turn_done"], "replied before its own turn finished"
            client.close()
        finally:
            server.stop()

    def test_a_running_session_is_steered_not_queued(self, tmp_path):
        """The disposition the injector reports decides the wait; scripted here as
        the real ``hand_to_agent`` reports it for a run that takes the text."""
        session = _Session(start_running=True)
        server, injected, _ = _server(tmp_path, session=session, disposition="steered",
                                      settle=0.0)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())

            def _finish():
                time.sleep(0.2)
                session.end()

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

    def test_a_lone_non_text_block_is_refused(self, tmp_path):
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

    def test_a_mixed_prompt_is_refused_whole_not_partially_delivered(self, tmp_path):
        """The case that actually bites: text *and* an image.

        Skipping the image would deliver "describe this image" with no image —
        a different question than the user asked, with nothing downstream able
        to notice. The earlier test only covered a lone image, which errored
        incidentally as "prompt is empty" and hid this.
        """
        server, injected, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt",
                {
                    "sessionId": "sess-1234",
                    "prompt": [
                        {"type": "text", "text": "describe this image"},
                        {"type": "image", "data": "AAAA"},
                    ],
                },
            )
            assert reply["error"]["code"] == -32602
            # The text must NOT have gone through on its own.
            assert injected == []
            assert "image" in reply["error"]["message"]
            client.close()
        finally:
            server.stop()

    def test_the_refusal_names_the_block_type(self, tmp_path):
        """The client must learn what was not delivered, not guess."""
        server, _, _ = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            reply = client.call(
                "session/prompt",
                {
                    "sessionId": "sess-1234",
                    "prompt": [
                        {"type": "text", "text": "look"},
                        {"type": "resource", "uri": "file:///a"},
                        {"type": "image", "data": "x"},
                    ],
                },
            )
            message = reply["error"]["message"]
            assert "resource" in message and "image" in message
            assert "NOT delivered" in message
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
        server, _, session = _server(tmp_path, answer=lambda: "the final answer")
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            _run_one_turn_soon(session)
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
        server, injected, session = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            client.close()
            time.sleep(0.2)
            # A second client still works.
            client2 = _Client(server.path, token=server.token_file.read_text().strip())
            _run_one_turn_soon(session)
            client2.call(
                "session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "after"}]}
            )
            assert injected == ["after"]
            client2.close()
        finally:
            server.stop()

    def test_several_messages_on_one_connection(self, tmp_path):
        server, injected, session = _server(tmp_path)
        try:
            client = _Client(server.path, token=server.token_file.read_text().strip())
            for n in range(3):
                _run_one_turn_soon(session)
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
        server, injected, session = _server(tmp_path)
        try:
            token = server.token_file.read_text().strip()
            a = _Client(server.path, token=token)
            b = _Client(server.path, token=token)
            _run_one_turn_soon(session)
            a.call("session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "a"}]})
            _run_one_turn_soon(session)
            b.call("session/prompt", {"sessionId": "sess-1234", "prompt": [{"type": "text", "text": "b"}]})
            assert injected == ["a", "b"]
            a.close()
            b.close()
        finally:
            server.stop()


class TestLifecycle:
    def test_the_socket_is_discoverable_from_the_listing(self, tmp_path, monkeypatch):
        """The discovery entrance must be one clients actually have.

        Without the row, a client can only go spelunking in ``live/*.json`` —
        which is what the first version of the e2e script had to do, and is not a
        supported path. Pinned through ``detail_rows`` because that is the single
        source both ``describe()`` (the model's ``list_agents``) and
        ``/list-agents`` render.
        """
        monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path))
        session = peers.PeerSession(name="tmux-cli", cwd="/tmp/proj")
        session.publish()
        server = AttachServer(
            session.peer_id, inject=lambda t: "queued", is_running=lambda: False
        )
        assert server.start()
        try:
            session.publish(attach_socket=str(server.path))
            info = peers.list_live_peers()[0]
            rows = dict(info.detail_rows())
            assert rows["attach"] == str(server.path)
            assert str(server.path) in info.describe()
        finally:
            server.stop()

    def test_no_attach_row_when_not_serving(self, tmp_path, monkeypatch):
        """A session with attach off must not advertise a socket that is not there."""
        monkeypatch.setattr(peers, "AGENTICA_CACHE_DIR", str(tmp_path))
        session = peers.PeerSession(name="tmux-cli", cwd="/tmp/proj")
        session.publish()
        assert "attach" not in dict(peers.list_live_peers()[0].detail_rows())

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

    def test_an_undeliverable_block_raises_rather_than_being_dropped(self):
        from agentica.attach import AttachError

        with pytest.raises(AttachError) as mixed:
            attach._prompt_text([{"type": "text", "text": "look"}, {"type": "image", "data": "x"}])
        assert "image" in str(mixed.value)
        assert "NOT delivered" in str(mixed.value)

        with pytest.raises(AttachError):
            attach._prompt_text([{"type": "image", "uri": "x"}])

    def test_a_non_list_non_string_is_refused(self):
        from agentica.attach import AttachError

        for bad in (None, 123, {"type": "text", "text": "x"}):
            with pytest.raises(AttachError):
                attach._prompt_text(bad)

    def test_an_empty_prompt_raises(self):
        from agentica.attach import AttachError

        with pytest.raises(AttachError):
            attach._prompt_text("   ")
        with pytest.raises(AttachError):
            attach._prompt_text([{"type": "text", "text": "  "}])


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
