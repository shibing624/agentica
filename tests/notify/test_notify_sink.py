# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the external notify sink.

These are aimed at the silent failures — the ways this can be wrong without
crashing. The headline risk is approving something the user never approved, so
every degradation path gets an explicit test.

A real in-process Unix-socket server stands in for the desktop app. The HTTP
layer is deliberately NOT mocked: mocking it would also mock away the timeout
and connect-failure behaviour that is the point of the ladder.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
import pytest

from agentica.notify import sink as sink_mod
from agentica.notify.config import NotifyConfig, load_notify_config
from agentica.notify.sink import NotifySink, get_sink, install_sink, reset_sink_for_tests


async def _read_one_request(reader) -> tuple:
    """Read exactly one HTTP request, honouring Content-Length.

    A single ``reader.read()`` is not enough: a stream socket may hand back
    only part of the request, so the body can be truncated depending on how
    the kernel split the write. When that happened the JSON failed to parse,
    ``json`` came back ``None``, and a test reading ``body["payload"]`` raised
    KeyError — about one run in five. That is worse than a plain bug: the
    suite goes red at random and a real failure gets waved off as "that flaky
    test". Frame on Content-Length instead.
    """
    raw = b""
    while b"\r\n\r\n" not in raw:
        chunk = await reader.read(65536)
        if not chunk:
            break
        raw += chunk
    head, _, body = raw.partition(b"\r\n\r\n")
    length = None
    for line in head.decode(errors="replace").splitlines()[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip().lower() == "content-length":
                try:
                    length = int(v.strip())
                except ValueError:
                    length = None
    if length is not None:
        while len(body) < length:
            chunk = await reader.read(65536)
            if not chunk:
                break
            body += chunk
        body = body[:length]
    return head, body


class _FakeDesktop:
    """An in-process stand-in for the desktop app's UDS server."""

    def __init__(self, *, decision_body: Optional[Any] = None, status: int = 200,
                 hang: bool = False, require_token: Optional[str] = None):
        self.requests: List[Dict[str, Any]] = []
        self._decision_body = decision_body if decision_body is not None else {"ok": True}
        self._status = status
        self._hang = hang
        self._require_token = require_token
        self._dir = tempfile.mkdtemp()
        self.socket_path = os.path.join(self._dir, "notify.sock")
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._ready = threading.Event()
        self._closing: Optional[asyncio.Event] = None
        self._hang_stop: Optional[asyncio.Event] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        async def handler(reader, writer):
            try:
                head, body = await _read_one_request(reader)
                lines = head.decode(errors="replace").splitlines()
                request_line = lines[0] if lines else ""
                headers = {}
                for line in lines[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        headers[k.strip().lower()] = v.strip()
                try:
                    parsed = json.loads(body.decode() or "{}")
                except Exception:
                    parsed = None
                self.requests.append({
                    "request_line": request_line,
                    "headers": headers,
                    "json": parsed,
                })
                if self._hang:
                    # Connected but silent: exercises the timeout path. Waits on
                    # an event rather than a long sleep so teardown can end it
                    # cleanly instead of destroying a pending task.
                    while not (self._hang_stop is not None and self._hang_stop.is_set()):
                        await asyncio.sleep(0.05)
                    return
                if self._require_token is not None:
                    if headers.get("authorization") != f"Bearer {self._require_token}":
                        payload = json.dumps({"error": "unauthorized"}).encode()
                        writer.write(
                            b"HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\n"
                            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
                        )
                        await writer.drain()
                        writer.close()
                        return
                if isinstance(self._decision_body, str):
                    payload = self._decision_body.encode()
                else:
                    payload = json.dumps(self._decision_body).encode()
                writer.write(
                    f"HTTP/1.1 {self._status} OK\r\nContent-Type: application/json\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n".encode() + payload
                )
                await writer.drain()
            except Exception:
                pass
            finally:
                try:
                    writer.close()
                except Exception:
                    pass

        async def main():
            self._server = await asyncio.start_unix_server(handler, path=self.socket_path)
            self._closing = asyncio.Event()
            self._hang_stop = asyncio.Event()
            self._ready.set()
            # Wait to be told to stop, then close cleanly. Killing the loop while
            # a coroutine is suspended in sleep() is what produces the
            # "coroutine ignored GeneratorExit" noise at teardown.
            await self._closing.wait()
            self._server.close()
            await self._server.wait_closed()

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(main())
        except Exception:
            pass

    def close(self) -> None:
        if self._loop is None:
            return
        if self._closing is not None:
            if self._hang_stop is not None:
                self._loop.call_soon_threadsafe(self._hang_stop.set)
            self._loop.call_soon_threadsafe(self._closing.set)
            self._thread.join(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        try:
            os.unlink(self.socket_path)
        except OSError:
            pass


class _MissingDesktop:
    """A path that nothing listens on — the desktop app is not running."""

    def __init__(self) -> None:
        self.socket_path = os.path.join(tempfile.mkdtemp(), "nope.sock")


def _cfg(socket_path: str, **kw) -> NotifyConfig:
    base = dict(enabled=True, socket=socket_path, approve_from_desktop=True)
    base.update(kw)
    return NotifyConfig(**base)


def _sink(desktop, **kw) -> NotifySink:
    """Sink whose transport points at ``desktop``'s socket."""
    cfg = _cfg(desktop.socket_path, **kw)
    return NotifySink(cfg, transport_factory=lambda p: httpx.HTTPTransport(uds=p))


def _flush(sink: NotifySink, timeout: float = 3.0) -> None:
    """Wait for the worker to drain the queue (bounded, so a bug cannot hang CI)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sink._queue.empty():
            # Give the worker one more beat to finish the in-flight POST.
            time.sleep(0.05)
            if sink._queue.empty():
                return
        time.sleep(0.02)


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


class TestNonBlockingEvents:
    def test_a_run_event_reaches_the_desktop_app(self):
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop)
            sink.emit_event("run.completed", session_id="s-1",
                            payload={"duration_seconds": 1.5, "title": "run completed"})
            _flush(sink)
            sink.stop()

            assert len(desktop.requests) == 1
            req = desktop.requests[0]
            assert req["request_line"].startswith("POST /event")
            assert req["json"]["event"] == "run.completed"
            assert req["json"]["v"] == 1
            assert req["json"]["source"] == "agentica"
            assert req["json"]["session_key"] == "s-1"
            assert req["json"]["payload"]["duration_seconds"] == 1.5
        finally:
            desktop.close()

    def test_a_disabled_sink_sends_nothing_at_all(self):
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop, enabled=False)
            sink.emit_event("run.started")
            _flush(sink, timeout=0.5)
            sink.stop()
            assert desktop.requests == []
        finally:
            desktop.close()

    def test_per_event_switches_are_honoured(self):
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop)
            sink.config.events["run.started"] = False
            sink.emit_event("run.started")
            sink.emit_event("run.completed")
            _flush(sink)
            sink.stop()
            events = [r["json"]["event"] for r in desktop.requests if r["json"]]
            assert events == ["run.completed"]
        finally:
            desktop.close()

    def test_a_missing_desktop_app_does_not_raise_or_stall(self):
        """Level 2: no app running is the common case and must be invisible."""
        sink = _sink(_MissingDesktop())
        started = time.monotonic()
        sink.emit_event("run.started")
        _flush(sink, timeout=2.0)
        sink.stop()
        assert time.monotonic() - started < 3.0
        assert sink._sent == 0

    def test_queue_is_bounded_and_drops_the_oldest(self):
        """A wedged app must not let the queue grow without limit."""
        from agentica.notify.config import QUEUE_MAXSIZE

        sink = _sink(_MissingDesktop())
        sink.stop()  # keep the worker out of the way; we are testing the queue
        total = QUEUE_MAXSIZE + 10
        for i in range(total):
            sink.emit_event("run.started", payload={"i": i})

        assert sink._queue.qsize() == QUEUE_MAXSIZE
        assert sink._dropped == total - QUEUE_MAXSIZE
        queued = []
        while not sink._queue.empty():
            queued.append(sink._queue.get_nowait())
        # Oldest went first, so the newest event is still there.
        assert queued[-1]["payload"]["i"] == total - 1

    def test_emitting_never_blocks_the_caller_even_when_the_app_hangs(self):
        """The caller-side contract: emit is queue-and-return, always.

        A consumer watching "when did I last hear anything" (a desktop app's
        inactivity timer) is only correct if a wedged peer delays *delivery*
        and never *the run*. A full queue drops events; it must not turn into a
        wait. Measured, not asserted by inspection.
        """
        from agentica.notify.config import QUEUE_MAXSIZE

        desktop = _FakeDesktop(hang=True)  # accepts, never answers
        try:
            sink = _sink(desktop, timeout_seconds=55)
            worst = 0.0
            # Well past the queue capacity, so drops definitely happen.
            for i in range(QUEUE_MAXSIZE * 4):
                started = time.monotonic()
                sink.emit_event("run.started", payload={"i": i})
                worst = max(worst, time.monotonic() - started)
            sink.stop()
            # Delivery timeout is 2s; if emit ever waited on it this would blow
            # past this bound.
            assert worst < 0.5, f"emit_event blocked for {worst:.3f}s"
            assert sink._dropped > 0, "expected the queue to have overflowed"
        finally:
            desktop.close()


class TestAwaitDecision:
    def test_a_decision_comes_back_verbatim(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow", "message": "ok"})
        try:
            sink = _sink(desktop)
            out = sink.await_decision(
                "needs.approval",
                payload={"approval_id": "call_1", "kind": "permission", "question": "run it?"},
            )
            sink.stop()
            assert out == {"decision": "allow"}
            assert desktop.requests[0]["request_line"].startswith("POST /await")
            body = desktop.requests[0]["json"]
            assert body["event"] == "needs.approval"
            assert body["payload"]["approval_id"] == "call_1"
        finally:
            desktop.close()

    def test_an_answer_comes_back_for_a_question(self):
        desktop = _FakeDesktop(decision_body={"answer": "the second one"})
        try:
            sink = _sink(desktop)
            out = sink.await_decision("needs.input", payload={"kind": "question", "question": "which?"})
            sink.stop()
            assert out == {"answer": "the second one"}
        finally:
            desktop.close()

    def test_a_timeout_falls_back_and_never_allows(self):
        """The headline risk: no answer must mean 'ask the human', not 'yes'."""
        desktop = _FakeDesktop(hang=True)
        try:
            sink = _sink(desktop, timeout_seconds=0.4)
            out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
            sink.stop()
            assert out is None
        finally:
            desktop.close()

    def test_a_missing_app_falls_back_immediately_not_after_the_timeout(self):
        """Level 2 again, on the blocking path: connect failure is a fast fail."""
        sink = _sink(_MissingDesktop(), timeout_seconds=55)
        started = time.monotonic()
        out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
        elapsed = time.monotonic() - started
        sink.stop()
        assert out is None
        assert elapsed < 5.0, f"waited {elapsed:.1f}s for an app that is not running"

    @pytest.mark.parametrize("body", [
        {"decision": "maybe"},          # not a decision we know
        {"decision": "yes"},            # plausible-looking, still not ours
        {"reject": True},               # explicit rejection
        {"ok": True},                   # an /event-shaped body
        {},                             # empty
        "not json at all",              # unparseable
        [1, 2, 3],                      # wrong shape
    ])
    def test_an_unusable_body_is_never_an_allow(self, body):
        """Level 4: unparseable or unknown means 'no decision', never approval."""
        desktop = _FakeDesktop(decision_body=body)
        try:
            sink = _sink(desktop)
            out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
            sink.stop()
            assert out is None
        finally:
            desktop.close()

    def test_an_http_error_is_not_a_decision(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow"}, status=500)
        try:
            sink = _sink(desktop)
            out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
            sink.stop()
            assert out is None
        finally:
            desktop.close()

    def test_approve_from_desktop_off_means_no_blocking_call_is_made(self):
        """The switch gates deciding. With it off the app is still told (via
        /event), but it can never answer — so nothing waits."""
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            sink = _sink(desktop, approve_from_desktop=False)
            started = time.monotonic()
            out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
            assert out is None
            assert time.monotonic() - started < 1.0
            sink.stop()
            assert desktop.requests == [], "must not even ask when deciding is off"
        finally:
            desktop.close()

    def test_a_disabled_sink_never_awaits(self):
        sink = _sink(_MissingDesktop(), enabled=False)
        assert sink.await_decision("needs.approval", payload={}) is None


class TestToken:
    def test_the_bearer_token_is_sent(self):
        desktop = _FakeDesktop(decision_body={"decision": "deny"}, require_token="tok-abc")
        try:
            sink = _sink(desktop, token="tok-abc")
            out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
            sink.stop()
            assert out == {"decision": "deny"}
            assert desktop.requests[0]["headers"]["authorization"] == "Bearer tok-abc"
        finally:
            desktop.close()

    def test_a_missing_token_is_a_401_and_yields_no_decision(self):
        """A channel any local process could forge must not be trusted."""
        desktop = _FakeDesktop(decision_body={"decision": "allow"}, require_token="tok-abc")
        try:
            sink = _sink(desktop, token="")
            out = sink.await_decision("needs.approval", payload={"approval_id": "c1"})
            sink.stop()
            assert out is None
        finally:
            desktop.close()

    def test_the_token_file_is_read_lazily(self):
        desktop = _FakeDesktop(decision_body={"decision": "allow"}, require_token="from-file")
        token_path = os.path.join(tempfile.mkdtemp(), "notify.token")
        try:
            # File does not exist yet: request should be unauthorized.
            sink = _sink(desktop, token="", token_file=token_path)
            assert sink.await_decision("needs.approval", payload={}) is None
            # The app writes the token afterwards; the next call must pick it up.
            with open(token_path, "w", encoding="utf-8") as fh:
                fh.write("from-file\n")
            out = sink.await_decision("needs.approval", payload={})
            sink.stop()
            assert out == {"decision": "allow"}
        finally:
            desktop.close()


class TestFailuresDoNotBecomeTheAgentsFailures:
    def test_a_broken_transport_factory_does_not_raise(self):
        def boom(_path):
            raise RuntimeError("no transport today")

        cfg = _cfg("/tmp/whatever.sock")
        sink = NotifySink(cfg, transport_factory=boom)
        sink.emit_event("run.started")       # must not raise
        _flush(sink, timeout=1.0)
        assert sink.await_decision("needs.approval", payload={}) is None
        sink.stop()

    def test_a_raising_emit_does_not_escape_and_dispatch_still_works(self):
        """The worker thread must survive a transport that throws on every call."""
        calls = {"n": 0}

        def flaky(path):
            calls["n"] += 1
            raise RuntimeError("flaky")

        cfg = _cfg("/tmp/whatever.sock")
        sink = NotifySink(cfg, transport_factory=flaky)
        for _ in range(3):
            sink.emit_event("run.started")
        _flush(sink, timeout=1.5)
        sink.stop()
        assert calls["n"] >= 1  # it tried, it failed, it did not die

    def test_dispatch_survives_a_record_that_has_no_payload(self):
        class _Bare:
            event_type = "run.started"

        install_sink(_cfg("/tmp/whatever.sock"))
        sink_mod.notify_sink_dispatch(_Bare())  # must not raise
        reset_sink_for_tests()


class TestConfigLoading:
    def test_defaults_are_off(self):
        cfg = load_notify_config({})
        assert cfg.enabled is False
        assert cfg.approve_from_desktop is False
        assert cfg.timeout_seconds == 55
        assert all(cfg.events[e] for e in
                   ("run.started", "run.completed", "run.failed", "run.cancelled"))

    def test_settings_block_is_read(self):
        cfg = load_notify_config({"settings": {"notify": {
            "enabled": True,
            "socket": "/tmp/x.sock",
            "approve_from_desktop": True,
            "timeout_seconds": 12,
            "events": {"run.started": False},
        }}})
        assert cfg.enabled is True
        assert cfg.resolved_socket == "/tmp/x.sock"
        assert cfg.approve_from_desktop is True
        assert cfg.timeout_seconds == 12
        assert cfg.event_enabled("run.started") is False
        assert cfg.event_enabled("run.completed") is True

    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_NOTIFY_ENABLED", "true")
        monkeypatch.setenv("AGENTICA_NOTIFY_SOCKET", "/tmp/env.sock")
        cfg = load_notify_config({"settings": {"notify": {"enabled": False, "socket": "/tmp/cfg.sock"}}})
        assert cfg.enabled is True
        assert cfg.resolved_socket == "/tmp/env.sock"

    def test_a_broken_settings_block_does_not_raise(self):
        cfg = load_notify_config({"settings": {"notify": "nonsense"}})
        assert cfg.enabled is False

    def test_a_non_numeric_timeout_is_ignored(self, monkeypatch):
        monkeypatch.setenv("AGENTICA_NOTIFY_TIMEOUT_SECONDS", "soon")
        cfg = load_notify_config({})
        assert cfg.timeout_seconds == 55


class TestInstall:
    def test_disabled_config_wires_nothing(self):
        assert install_sink(NotifyConfig(enabled=False)) is None
        assert get_sink() is None

    def test_enabled_config_installs_a_usable_sink(self):
        desktop = _FakeDesktop()
        try:
            sink = install_sink(_cfg(desktop.socket_path))
            assert sink is not None and get_sink() is sink
            sink.emit_event("run.started")
            _flush(sink)
            assert len(desktop.requests) == 1
        finally:
            desktop.close()
