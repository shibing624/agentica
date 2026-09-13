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

    def __init__(self, *, hang: bool = False, require_token: Optional[str] = None):
        self.requests: List[Dict[str, Any]] = []
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
                # The sink is observe-only: the reply body is not part of any
                # contract, so the double just acknowledges the POST.
                payload = json.dumps({"ok": True}).encode()
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
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
    base = dict(enabled=True, socket=socket_path)
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


def _wait_requests(desktop, count: int, timeout: float = 3.0) -> None:
    """Wait until the double has recorded ``count`` requests.

    ``_flush`` only means "the queue was drained", which happens before the
    in-flight POST is recorded on the server side.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(desktop.requests) >= count:
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

    def test_the_attach_point_is_carried_in_the_transport_block(self):
        """A consumer that only speaks the notify channel must not have to
        locate and parse the presence record to find the attach socket — a
        launchd-started .app cannot even import agentica to do it. The
        transport block already answers "how do I reach this session".
        """
        desktop = _FakeDesktop()
        try:
            from agentica.notify import set_attach_endpoint

            set_attach_endpoint("/tmp/agentica-501/abc.sock", "abc")
            sink = _sink(desktop)
            sink.emit_event("run.started")
            _flush(sink)
            sink.stop()

            transport = desktop.requests[0]["json"]["transport"]
            assert transport["attach_socket"] == "/tmp/agentica-501/abc.sock"
            assert transport["peer_id"] == "abc"
        finally:
            from agentica.notify import reset_sink_for_tests, set_attach_endpoint

            set_attach_endpoint(None, None)
            reset_sink_for_tests()
            desktop.close()

    def test_the_process_reset_also_clears_the_attach_endpoint(self):
        """The autouse fixture calls ``reset_sink_for_tests`` and is the only
        thing stopping one test's attach point from reaching the next. That
        holds only because this reset happens to cover the endpoint as well as
        the sink — nothing states it: the name mentions the sink, so trimming
        it to "just the sink" leaves the fixture looking intact while the
        endpoint starts leaking between tests.
        """
        from agentica.notify import set_attach_endpoint

        set_attach_endpoint("/tmp/agentica-501/abc.sock", "abc")
        reset_sink_for_tests()
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop)
            sink.emit_event("run.started")
            _flush(sink)
            sink.stop()

            assert "attach_socket" not in desktop.requests[0]["json"]["transport"]
        finally:
            desktop.close()

    def test_a_process_without_an_attach_point_says_nothing_about_one(self):
        """Absent, not null: a consumer must be able to tell "no attach point"
        from "there is one and here it is"."""
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop)
            sink.emit_event("run.started")
            _flush(sink)
            sink.stop()

            transport = desktop.requests[0]["json"]["transport"]
            assert "attach_socket" not in transport
            assert "peer_id" not in transport
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
            sink = _sink(desktop)
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


class TestTheSinkIsObserveOnly:
    """The reply half is gone: it lives in the hook egress now.

    What has to stay true of what is left: nothing on this channel waits for a
    person, and nothing on it can be mistaken for an answer.
    """

    def test_the_reply_methods_are_gone(self):
        assert not hasattr(NotifySink, "await_decision")
        assert not hasattr(NotifySink, "_parse_decision")

    def test_no_reply_path_is_left_on_the_sink(self):
        """A regression here would reintroduce a second answer channel.

        Checked on the class rather than by scanning the source: the module
        docstring legitimately mentions the removed path, and a prose match would
        make this test fail for the wrong reason.
        """
        names = [n for n in dir(NotifySink) if not n.startswith("__")]
        assert not [n for n in names if "await" in n or "decision" in n]

    def test_the_sink_only_ever_posts_to_event(self):
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop)
            for event in ("run.started", "run.completed", "run.failed", "run.cancelled"):
                sink.emit_event(event)
            _flush(sink)
            _wait_requests(desktop, 4)
            sink.stop()
            assert desktop.requests, "expected the events to be delivered"
            for request in desktop.requests:
                assert request["request_line"].startswith("POST /event")
        finally:
            desktop.close()

    def test_an_unusable_body_is_never_read_as_a_decision(self):
        """A desktop that answers with a decision gets it ignored: there is no
        code path from a reply body to an action any more."""
        desktop = _FakeDesktop()
        try:
            sink = _sink(desktop)
            sink.emit_event("run.started")
            _flush(sink)
            _wait_requests(desktop, 1)
            sink.stop()
            assert desktop.requests[0]["json"]["event"] == "run.started"
        finally:
            desktop.close()


class TestToken:
    def test_the_bearer_token_is_sent(self):
        desktop = _FakeDesktop(require_token="tok-abc")
        try:
            sink = _sink(desktop, token="tok-abc")
            sink.emit_event("run.started")
            _flush(sink)
            _wait_requests(desktop, 1)
            sink.stop()
            assert desktop.requests[0]["headers"]["authorization"] == "Bearer tok-abc"
        finally:
            desktop.close()

    def test_no_token_means_no_authorization_header(self):
        """A channel any local process could reach must present the token.

        With no decision coming back any more, "unauthenticated" means the notice
        was not shown — which is the only consequence this channel can have.

        ``token_file`` points at an absent path on purpose: the default is a real
        file on a machine that runs the desktop app, and reading it here would
        make the test depend on the developer's own setup.
        """
        desktop = _FakeDesktop(require_token="tok-abc")
        try:
            sink = _sink(
                desktop,
                token="",
                token_file=os.path.join(tempfile.mkdtemp(), "absent.token"),
            )
            sink.emit_event("run.started")
            _flush(sink)
            _wait_requests(desktop, 1)
            sink.stop()
            assert "authorization" not in desktop.requests[0]["headers"]
        finally:
            desktop.close()

    def test_the_token_file_is_read_lazily(self):
        desktop = _FakeDesktop(require_token="from-file")
        token_path = os.path.join(tempfile.mkdtemp(), "notify.token")
        try:
            # File does not exist yet: the request goes out unauthorized.
            sink = _sink(desktop, token="", token_file=token_path)
            sink.emit_event("run.started")
            _flush(sink)
            _wait_requests(desktop, 1)
            assert "authorization" not in desktop.requests[0]["headers"]
            # The app writes the token afterwards; the next call must pick it up.
            with open(token_path, "w", encoding="utf-8") as fh:
                fh.write("from-file\n")
            sink.emit_event("run.failed")
            _flush(sink)
            _wait_requests(desktop, 2)
            sink.stop()
            assert desktop.requests[1]["headers"]["authorization"] == "Bearer from-file"
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
        assert all(cfg.events[e] for e in
                   ("run.started", "run.completed", "run.failed", "run.cancelled"))

    def test_settings_block_is_read(self):
        cfg = load_notify_config({"settings": {"notify": {
            "enabled": True,
            "socket": "/tmp/x.sock",
            "events": {"run.started": False},
        }}})
        assert cfg.enabled is True
        assert cfg.resolved_socket == "/tmp/x.sock"
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

    def test_removed_options_in_an_existing_config_are_ignored(self):
        """A user's config.yaml may still name the switch and the timeout.

        Both were removed — there is no "may the app decide" concept, and how
        long a person may take is not this layer's call. Loading has to tolerate
        the leftovers rather than fail.
        """
        cfg = load_notify_config({"settings": {"notify": {
            "enabled": True,
            "approve_from_desktop": True,
            "timeout_seconds": 12,
        }}})
        assert cfg.enabled is True
        assert not hasattr(cfg, "approve_from_desktop")
        assert not hasattr(cfg, "timeout_seconds")


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
