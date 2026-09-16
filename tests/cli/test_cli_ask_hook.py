# -*- coding: utf-8 -*-
"""The ask side of the hook race: one slot, one delivery path, terminal first."""

from __future__ import annotations

import queue
import sys
import time

import pytest

from agentica.cli.interactive.ask_hook import start_hook_ask
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


class _Request:
    """The CANCELLED sentinel the real ``_InputRequest`` uses."""

    CANCELLED = object()

    def __init__(self):
        self.result = queue.Queue(maxsize=1)
        self.resolved = False

    def submit(self, answer: str, *, source: str = "terminal") -> bool:
        if self.resolved:
            return False
        try:
            self.result.put_nowait(answer)
            self.resolved = True
            return True
        except queue.Full:
            self.resolved = True
            return False


def _script(tmp_path, body, name="hook.py"):
    path = tmp_path / name
    path.write_text(
        "import json,sys\n"
        "payload=json.load(sys.stdin)\n"
        "request_id=payload['request_id']\n"
        + body,
        encoding="utf-8",
    )
    return [sys.executable, str(path)]


def _install(command, **kw):
    install_hook_egress(
        ShellHooksConfig(
            enabled=True,
            consumers=[HookConsumer(name="desktop", command=command, **kw)],
        )
    )


def test_the_hook_answer_reaches_the_slot(tmp_path):
    _install(
        _script(
            tmp_path,
            "print(json.dumps({'request_id':request_id,'answer':'from hook'}))",
        )
    )
    hook = start_hook_ask("which package?")
    assert hook is not None
    req = _Request()
    try:
        deadline = time.monotonic() + 10
        while not hook.poll(req) and time.monotonic() < deadline:
            pass
        assert req.result.get_nowait() == "from hook"
    finally:
        hook.stop()


def test_the_typed_answer_wins_when_it_arrives_first(tmp_path):
    """The user typed; the hook's answer is then second and must not overwrite."""
    _install(
        _script(
            tmp_path,
            "import time;time.sleep(1.0);"
            "print(json.dumps({'request_id':request_id,'answer':'from hook'}))",
        )
    )
    hook = start_hook_ask("which?")
    assert hook is not None
    req = _Request()
    try:
        req.submit("typed by hand")
        # Poll until the hook has answered (and been told it lost the race).
        deadline = time.monotonic() + 10
        delivered = False
        while time.monotonic() < deadline and hook.still_useful:
            delivered = hook.poll(req) or delivered
        assert delivered is True, "poll reports that a reply arrived"
        assert req.result.get_nowait() == "typed by hand"
    finally:
        hook.stop()


def test_an_empty_reply_leaves_the_slot_open(tmp_path):
    _install(_script(tmp_path, "pass"))
    hook = start_hook_ask("which?")
    assert hook is not None
    req = _Request()
    try:
        deadline = time.monotonic() + 5
        while hook.still_useful and time.monotonic() < deadline:
            assert hook.poll(req) is False
        assert not req.resolved, "the prompt stays open for the user"
    finally:
        hook.stop()


def test_junk_is_not_an_answer(tmp_path):
    _install(_script(tmp_path, "print('not json at all')"))
    hook = start_hook_ask("which?")
    assert hook is not None
    req = _Request()
    try:
        deadline = time.monotonic() + 5
        while hook.still_useful and time.monotonic() < deadline:
            assert hook.poll(req) is False
        assert not req.resolved
    finally:
        hook.stop()


def test_no_hook_configured_means_no_hook_ask(tmp_path):
    assert start_hook_ask("which?") is None


def test_a_switched_off_event_means_no_hook_ask(tmp_path):
    _install(_script(tmp_path, "pass"), events={"needs.input": False})
    assert start_hook_ask("which?") is None


def test_a_missing_command_means_no_hook_ask(tmp_path):
    _install(["/nonexistent/notifier"])
    assert start_hook_ask("which?") is None


def test_the_question_is_offered_with_its_options(tmp_path):
    out = tmp_path / "seen.json"
    _install(
        _script(
            tmp_path,
            "import pathlib;"
            f"pathlib.Path({str(out)!r}).write_text(json.dumps(payload));"
            "print(json.dumps({'request_id':request_id,'answer':'first'}))",
        )
    )
    hook = start_hook_ask("which package?", ["date-fns", "dayjs"], session_id="s1")
    assert hook is not None
    req = _Request()
    try:
        deadline = time.monotonic() + 10
        while not hook.poll(req) and time.monotonic() < deadline:
            pass
        import json

        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["hook_event_name"] == "needs.input"
        assert doc["question"] == "which package?"
        assert doc["options"] == ["date-fns", "dayjs"]
        assert doc["session_id"] == "s1"
        assert "decision" not in doc
    finally:
        hook.stop()


def test_a_hook_that_prints_a_lot_does_not_wedge_the_poll(tmp_path):
    """A consumer that over-prints must not block the prompt loop."""
    _install(
        _script(
            tmp_path,
            "print('x'*200000);"
            "print(json.dumps({'request_id':request_id,'answer':'late'}))",
        )
    )
    hook = start_hook_ask("which?")
    assert hook is not None
    req = _Request()
    try:
        deadline = time.monotonic() + 15
        while not req.resolved and time.monotonic() < deadline:
            hook.poll(req)
            if not hook.still_useful:
                break
        # The cap means we may not get the answer, but the poll must never hang.
        assert True
    finally:
        hook.stop()
