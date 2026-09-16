# -*- coding: utf-8 -*-
"""The blocking needs.* path. The two failures worth testing are "the answer
never arrives" and "the answer arrives after the user already answered"."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Tuple

import pytest

from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.requests import start_hook_request


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


def _config(command, **kw):
    return ShellHooksConfig(
        enabled=True,
        consumers=[HookConsumer(name="desktop", command=command, **kw)],
    )


def _payload_for(pending, **kw):
    from agentica.shell_hooks.requests import approval_payload

    return approval_payload(
        pending, session_id="s1", work_dir="/w", prompt="the task", **kw
    )


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
    def test_a_decision_is_the_reply(self, tmp_path):
        cmd = _script(
            tmp_path,
            "print(json.dumps({'request_id':request_id,'decision':'allow'}))",
        )
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=10) == {"decision": "allow"}
        finally:
            req.kill()

    def test_all_four_decisions_come_back(self, tmp_path):
        for word in ("allow", "allow_prefix", "deny", "deny_prefix"):
            cmd = _script(
                tmp_path,
                f"print(json.dumps({{'request_id':request_id,'decision':'{word}'}}))",
                name=f"hook_{word}.py",
            )
            install_hook_egress(_config(cmd))
            req = start_hook_request("needs.approval", _payload_for(_Pending()))
            assert req is not None
            try:
                assert req.wait_for_reply(timeout=10) == {"decision": word}
            finally:
                req.kill()

    def test_the_terminal_wins_when_it_answers_first(self, tmp_path):
        """The hook is still thinking; the user typed y. The caller must be able
        to stop waiting and kill the process, not await it."""
        cmd = _script(tmp_path, "import time;time.sleep(30)")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=1.0) is None  # no reply yet
            assert req.still_waiting is True
            req.kill()
            assert req.still_waiting is False
        finally:
            req.kill()

    def test_no_decision_reply_is_not_a_decision(self, tmp_path):
        cmd = _script(
            tmp_path, "print(json.dumps({'request_id':request_id,'nope':1}))"
        )
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=5.0) is None
            # The hook spoke and exited, so nothing more can arrive from it. The
            # point of the case is that an unusable document is not a decision —
            # the terminal prompt is still the answer path.
            assert req.still_waiting is False
        finally:
            req.kill()

    def test_an_empty_stdout_is_not_a_decision(self, tmp_path):
        cmd = _script(tmp_path, "pass")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=5.0) is None
        finally:
            req.kill()

    def test_a_non_zero_exit_is_not_a_decision(self, tmp_path):
        cmd = _script(tmp_path, "import sys;sys.exit(4)")
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=5.0) is None
        finally:
            req.kill()

    def test_a_valid_json_reply_wins_even_when_the_process_exits_non_zero(
        self, tmp_path
    ):
        """The document is the decision. Exit status used to race with early
        JSON completion: the same allow-then-exit-1 script was accepted or
        ignored depending on whether the child had been reaped yet."""
        cmd = _script(
            tmp_path,
            "print(json.dumps({'request_id':request_id,'decision':'allow'}));"
            "sys.stdout.flush();"
            "raise SystemExit(1)",
        )
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=5.0) == {"decision": "allow"}
        finally:
            req.kill()

    def test_an_unknown_word_is_not_a_decision(self, tmp_path):
        cmd = _script(
            tmp_path,
            "print(json.dumps({'request_id':request_id,'decision':'sure'}))",
        )
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=5.0) is None
        finally:
            req.kill()

    def test_a_live_hook_that_printed_junk_keeps_waiting(self, tmp_path):
        """The case the approval loop depends on: junk first, a real answer
        later. ``still_waiting`` must stay True so the loop keeps polling rather
        than treating the junk as the reply and giving up."""
        body = (
            "import time;"
            "print('warming up');sys.stdout.flush();"
            "time.sleep(1.2);"
            "print(json.dumps({'request_id':request_id,'decision':'allow'}));"
            "sys.stdout.flush()"
        )
        install_hook_egress(_config(_script(tmp_path, body)))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=0.4) is None
            assert req.still_waiting is True
            assert req.wait_for_reply(timeout=10) == {"decision": "allow"}
        finally:
            req.kill()

    def test_the_first_valid_consumer_reply_wins(self, tmp_path):
        slow = _script(
            tmp_path,
            "import time;time.sleep(3);"
            "print(json.dumps({'request_id':request_id,'decision':'allow'}))",
            name="slow.py",
        )
        fast = _script(
            tmp_path,
            "print(json.dumps({'request_id':request_id,'decision':'deny'}))",
            name="fast.py",
        )
        install_hook_egress(
            ShellHooksConfig(
                enabled=True,
                consumers=[
                    HookConsumer(name="slow", command=slow),
                    HookConsumer(name="fast", command=fast),
                ],
            )
        )
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            started = time.monotonic()
            assert req.wait_for_reply(timeout=10) == {"decision": "deny"}
            assert time.monotonic() - started < 2
        finally:
            req.kill()

    def test_a_reply_for_another_request_is_ignored(self, tmp_path):
        cmd = _script(
            tmp_path,
            "print(json.dumps({'request_id':'wrong','decision':'allow'}))",
        )
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=5) is None
        finally:
            req.kill()


class TestNoEgress:
    def test_no_egress_means_no_request(self):
        assert start_hook_request("needs.approval", {}) is None

    def test_a_switched_off_event_means_no_request(self, tmp_path):
        cmd = _script(tmp_path, "pass")
        install_hook_egress(_config(cmd, events={"needs.approval": False}))
        assert start_hook_request("needs.approval", {}) is None

    def test_a_missing_command_means_no_request(self):
        install_hook_egress(_config(["/nonexistent/notifier"]))
        assert start_hook_request("needs.approval", {}) is None


class TestNoCapOfOurs:
    def test_a_slow_reply_still_lands(self, tmp_path):
        """The harness imposes no deadline. A consumer that takes 3 seconds is
        still waiting for the same user the terminal is waiting for."""
        cmd = _script(
            tmp_path,
            "import time;time.sleep(3);"
            "print(json.dumps({'request_id':request_id,'decision':'deny'}))",
        )
        install_hook_egress(_config(cmd))
        req = start_hook_request("needs.approval", _payload_for(_Pending()))
        assert req is not None
        try:
            assert req.wait_for_reply(timeout=30) == {"decision": "deny"}
        finally:
            req.kill()
