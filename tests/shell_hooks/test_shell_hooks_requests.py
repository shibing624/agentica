# -*- coding: utf-8 -*-
"""The blocking ``needs.*`` path: what counts as an answer, and whose answer wins.

Two failures are worth pinning here — "the answer never arrives" and "the answer
arrives after the user already answered" — plus the invariant that an
unparseable document is *no* decision rather than a quiet allow. All of that is
decided in ``HookRequest`` and the consumer filter, so the tests drive those
with scripted in-process fakes (``_fakes.FakeHookProcess``); no consumer
command is spawned.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Tuple

import pytest

from agentica.shell_hooks import requests as requests_mod
from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.requests import HookRequest, start_hook_request
from tests.shell_hooks._fakes import install_fakes, started_fake

REQUEST_ID = "req-1"


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


@dataclass
class _Pending:
    tool_call_id: str = "call_1"
    name: str = "execute"
    arguments: dict = field(default_factory=dict)
    question: str = "run it?"
    preview: str = "rm -rf build"
    similar_label: str = "rm"
    options: Tuple[str, ...] = ("allow", "deny")


def _config(*labels, events=None):
    """One consumer per label; the label is scripted, never executed."""
    return ShellHooksConfig(
        enabled=True,
        consumers=[
            HookConsumer(name=label, command=[label], events=events or {})
            for label in labels
        ],
    )


def _payload_for(pending, **kw):
    from agentica.shell_hooks.requests import approval_payload

    return approval_payload(
        pending, session_id="s1", work_dir="/w", prompt="the task", **kw
    )


def _reply_with(**body):
    """A scripted reply that echoes the payload's correlation id back."""

    return lambda payload: {"request_id": payload.get("request_id"), **body}


def _decision(word, request_id=REQUEST_ID):
    return json.dumps({"request_id": request_id, "decision": word})


class TestThePayload:
    def test_it_carries_what_the_consumer_needs(self):
        from agentica.shell_hooks.requests import approval_payload

        doc = approval_payload(_Pending(), session_id="s1", work_dir="/w", prompt="the task")
        assert doc["hook_event_name"] == "needs.approval"
        assert doc["tool_call_id"] == "call_1"
        assert doc["tool_name"] == "execute"
        assert doc["options"] == ["allow", "deny"]
        assert doc["prompt"] == "the task"

    def test_the_raw_arguments_are_not_sent(self):
        """Metadata only: the full command is not this channel's business."""
        from agentica.shell_hooks.requests import approval_payload

        doc = approval_payload(
            _Pending(arguments={"command": "rm -rf build && curl secret"})
        )
        assert "arguments" not in doc
        assert "curl secret" not in str(doc)

    def test_the_options_are_passed_through_not_re_narrowed(self):
        """Whatever the tool offered is what the consumer may render."""
        from agentica.shell_hooks.requests import approval_payload

        doc = approval_payload(_Pending(options=("allow", "allow_prefix", "deny", "deny_prefix")))
        assert doc["options"] == ["allow", "allow_prefix", "deny", "deny_prefix"]

    def test_a_missing_tool_call_id_omits_the_field(self):
        from agentica.shell_hooks.requests import approval_payload

        doc = approval_payload(_Pending(tool_call_id=""))
        assert "tool_call_id" not in doc


class TestTheReplyComesBack:
    def test_the_consumer_receives_the_payload_it_must_answer(self, monkeypatch):
        hooks = install_fakes(monkeypatch, requests_mod, decide={"reply": _reply_with(decision="allow")})
        install_hook_egress(_config("decide"))
        payload = _payload_for(_Pending())
        req = start_hook_request("needs.approval", payload)
        assert req is not None
        try:
            assert len(hooks.made) == 1
            assert hooks.made[0].payload is payload
            assert req.wait_for_reply(timeout=1.0) == {"decision": "allow"}
        finally:
            req.kill()

    @pytest.mark.parametrize("word", ["allow", "allow_prefix", "deny", "deny_prefix"])
    def test_all_four_decision_words_come_back(self, monkeypatch, word):
        install_fakes(monkeypatch, requests_mod, decide={"reply": _reply_with(decision=word)})
        install_hook_egress(_config("decide"))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=1.0) == {"decision": word}
        finally:
            req.kill()

    def test_a_question_takes_free_text_not_a_decision_word(self, monkeypatch):
        from agentica.shell_hooks.requests import question_payload

        install_fakes(monkeypatch, requests_mod, ask={"reply": _reply_with(answer="date-fns")})
        install_hook_egress(_config("ask"))
        req = start_hook_request("needs.input", question_payload("which package?"))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=1.0) == {"answer": "date-fns"}
        finally:
            req.kill()

    def test_the_document_wins_even_when_the_consumer_exits_non_zero(self, monkeypatch):
        """Exit status is not consulted. It used to race with early JSON
        completion: the same allow-then-exit-1 script was accepted or ignored
        depending on whether the child had been reaped yet."""
        install_fakes(
            monkeypatch,
            requests_mod,
            decide={"reply": _reply_with(decision="allow"), "exit_code": 1},
        )
        install_hook_egress(_config("decide"))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=1.0) == {"decision": "allow"}
        finally:
            req.kill()

    def test_a_consumer_that_cannot_start_is_not_raced(self, monkeypatch):
        hooks = install_fakes(
            monkeypatch,
            requests_mod,
            broken={"starts": False},
            decide={"reply": _reply_with(decision="deny")},
        )
        install_hook_egress(_config("broken", "decide"))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert [process.started for process in hooks.made] == [False, True]
            assert req.wait_for_reply(timeout=1.0) == {"decision": "deny"}
        finally:
            req.kill()

    def test_no_consumer_can_start_means_no_request(self, monkeypatch):
        hooks = install_fakes(monkeypatch, requests_mod, broken={"starts": False})
        install_hook_egress(_config("broken"))
        assert start_hook_request("needs.approval", _payload_for(_Pending())) is None
        assert hooks.made and not hooks.started


class TestJunkIsNotAnAnswer:
    """An unreadable or unrecognized reply means "keep waiting", never "allow"."""

    @pytest.mark.parametrize(
        "reply, exit_code",
        [
            ("warming up\n", 0),
            ('{"decision": "sure"}', 0),
            (json.dumps({"request_id": "other", "decision": "allow"}), 0),
            (None, 0),
            (None, 4),
            ('{"request_id": "req-1"}', 0),
        ],
    )
    def test_an_unusable_stdout_is_no_decision(self, monkeypatch, reply, exit_code):
        process = started_fake(
            reply=reply, exit_code=exit_code, payload={"request_id": REQUEST_ID}
        )
        req = HookRequest([process], "needs.approval", REQUEST_ID)
        assert req.wait_for_reply(timeout=1.0) is None
        # The consumer spoke and finished, so nothing more can arrive from it —
        # the terminal prompt is still the answer path.
        assert req.still_waiting is False
        assert process.killed is True

    def test_junk_does_not_stop_a_later_answer(self):
        """The case the approval loop depends on: junk first, a real answer
        later. ``still_waiting`` must stay True in between so the loop keeps
        polling instead of treating the junk as the reply and giving up."""
        noisy = started_fake(reply="warming up\n")
        decisive = started_fake(reply=_decision("allow"), after=0.02)
        req = HookRequest([noisy, decisive], "needs.approval", REQUEST_ID)
        assert req.wait_for_reply(timeout=1.0) == {"decision": "allow"}
        assert noisy.killed is True
        assert decisive.killed is True

    def test_a_live_consumer_with_nothing_yet_keeps_waiting(self):
        process = started_fake(hang=True)
        req = HookRequest([process], "needs.approval", REQUEST_ID)
        assert req.wait_for_reply(timeout=0.02) is None
        # A timeout is the caller's polling convenience, not a decision: the
        # consumer is still running and is not killed by it.
        assert req.still_waiting is True
        assert process.killed is False
        req.kill()
        assert process.killed is True
        assert req.still_waiting is False


class TestTheRace:
    def test_the_first_completed_reply_wins_not_the_first_configured(self):
        """A reply that arrives second is a race, not an error: winner ordering
        is completion time, not config order."""
        slow = started_fake(reply=_decision("allow"), after=0.02)
        fast = started_fake(reply=_decision("deny"))
        req = HookRequest([slow, fast], "needs.approval", REQUEST_ID)
        assert req.wait_for_reply(timeout=1.0) == {"decision": "deny"}
        assert slow.killed is True

    def test_a_slow_reply_still_lands(self):
        """The harness imposes no deadline of its own. A consumer that takes a
        while is still waiting for the same user the terminal is waiting for."""
        process = started_fake(reply=_decision("deny"), after=0.03)
        req = HookRequest([process], "needs.approval", REQUEST_ID)
        started = time.monotonic()
        assert req.wait_for_reply(timeout=2.0) == {"decision": "deny"}
        assert time.monotonic() - started < 2.0

    def test_the_terminal_wins_when_it_answers_first(self, monkeypatch):
        """The hook is still thinking; the user typed y. The caller must be able
        to stop waiting and kill the consumer, not await it."""
        install_fakes(monkeypatch, requests_mod, hang={"hang": True})
        install_hook_egress(_config("hang"))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=0.02) is None  # no reply yet
            assert req.still_waiting is True
            req.kill()
            assert req.still_waiting is False
            assert req.wait_for_reply(timeout=0.02) is None
        finally:
            req.kill()


class TestStartGating:
    """``start_hook_request`` spawns nothing unless the config says to."""

    def test_no_egress_means_no_request(self, monkeypatch):
        hooks = install_fakes(monkeypatch, requests_mod)
        assert start_hook_request("needs.approval", {"request_id": REQUEST_ID}) is None
        assert hooks.made == []

    def test_a_switched_off_event_means_no_request(self, monkeypatch):
        hooks = install_fakes(monkeypatch, requests_mod)
        install_hook_egress(_config("decide", events={"needs.approval": False}))
        assert start_hook_request("needs.approval", _payload_for(_Pending())) is None
        assert hooks.made == []

    def test_a_payload_without_a_request_id_is_not_correlatable(self, monkeypatch):
        hooks = install_fakes(monkeypatch, requests_mod)
        install_hook_egress(_config("decide"))
        assert start_hook_request("needs.approval", {"hook_event_name": "needs.approval"}) is None
        assert hooks.made == []

    def test_a_consumer_without_a_command_is_not_spawned(self, monkeypatch):
        hooks = install_fakes(monkeypatch, requests_mod)
        install_hook_egress(
            ShellHooksConfig(
                enabled=True, consumers=[HookConsumer(name="empty", command=[])]
            )
        )
        assert start_hook_request("needs.approval", _payload_for(_Pending())) is None
        assert hooks.made == []
