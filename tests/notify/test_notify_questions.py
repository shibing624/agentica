# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the two-way question path (step 5).

A question is not an approval: the caller reads a string back, and the terminal
implementation is the fallback. The failures worth testing are the ones that
would silently answer for the user or skip the terminal prompt entirely.
"""

from __future__ import annotations

import time
from typing import List, Optional

import pytest

from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.config import NotifyConfig
from agentica.notify.questions import ask_via_desktop, wrap_ask_callback

from tests.notify.test_notify_sink import _FakeDesktop, _MissingDesktop


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


def _install(desktop, **kw):
    install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path,
                              approve_from_desktop=True, **kw))


class TestTheDesktopCanAnswer:
    def test_an_answer_comes_back(self):
        desktop = _FakeDesktop(decision_body={"answer": "the second one"})
        try:
            _install(desktop)
            out = ask_via_desktop("which one?", ["first", "second"])
            assert out == "the second one"
            body = desktop.requests[0]["json"]
            assert body["event"] == "needs.input"
            # The kind must distinguish a question from an approval, or the
            # desktop app would use its urgent "approve me" voice to ask
            # "which do you prefer".
            assert body["payload"]["kind"] == "question"
            assert body["payload"]["question"] == "which one?"
            assert body["payload"]["options"] == ["first", "second"]
        finally:
            desktop.close()

    def test_a_question_without_options_still_works(self):
        desktop = _FakeDesktop(decision_body={"answer": "just do it"})
        try:
            _install(desktop)
            assert ask_via_desktop("what now?") == "just do it"
            assert "options" not in desktop.requests[0]["json"]["payload"]
        finally:
            desktop.close()


class TestFallbacksGoToTheTerminal:
    def test_a_missing_app_falls_back_to_the_terminal(self):
        """No desktop app must mean the terminal prompt, unchanged."""
        _install(_MissingDesktop())
        calls = []

        def tui(prompt, options=None):
            calls.append((prompt, options))
            return "typed in the terminal"

        wrapped = wrap_ask_callback(tui)
        assert wrapped("which?", ["a", "b"]) == "typed in the terminal"
        assert calls == [("which?", ["a", "b"])]

    @pytest.mark.parametrize("body", [
        {"answer": ""},              # empty is not an answer
        {"answer": "   "},           # whitespace is not an answer
        {"decision": "allow"},       # an approval body; wrong event
        {"reject": True},
        {},
        "not json",
    ])
    def test_an_unusable_answer_goes_to_the_terminal(self, body):
        desktop = _FakeDesktop(decision_body=body)
        try:
            _install(desktop)
            calls = []
            wrapped = wrap_ask_callback(
                lambda p, o=None: calls.append(p) or "from terminal"
            )
            assert wrapped("which?") == "from terminal"
            assert calls == ["which?"]
        finally:
            desktop.close()

    def test_a_timeout_falls_back_to_the_terminal(self):
        desktop = _FakeDesktop(hang=True)
        try:
            _install(desktop, timeout_seconds=0.4)
            wrapped = wrap_ask_callback(lambda p, o=None: "from terminal")
            started = time.monotonic()
            assert wrapped("which?") == "from terminal"
            # It waited for the desktop, then handed over control.
            assert time.monotonic() - started >= 0.3
        finally:
            desktop.close()

    def test_with_the_switch_off_the_terminal_answers(self):
        """approve_from_desktop gates answering questions too: deciding is one
        capability, whether it is a y/n or a typed answer."""
        desktop = _FakeDesktop(decision_body={"answer": "from desktop"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path,
                                      approve_from_desktop=False))
            assert ask_via_desktop("which?") is None
            time.sleep(0.2)
            assert desktop.requests == []
        finally:
            desktop.close()

    def test_a_disabled_sink_means_the_terminal(self):
        desktop = _FakeDesktop(decision_body={"answer": "from desktop"})
        try:
            install_sink(NotifyConfig(enabled=False, socket=desktop.socket_path))
            assert ask_via_desktop("which?") is None
            assert desktop.requests == []
        finally:
            desktop.close()


class TestTheWrapperNeverFailsTheCall:
    def test_a_broken_sink_still_lets_the_terminal_answer(self):
        def boom(_path):
            raise RuntimeError("sink is broken")

        from agentica.notify.sink import NotifySink
        import agentica.notify.sink as sink_mod
        sink_mod._sink = NotifySink(NotifyConfig(enabled=True, socket="/tmp/x.sock",
                                                 approve_from_desktop=True),
                                    transport_factory=boom)
        wrapped = wrap_ask_callback(lambda p, o=None: "from terminal")
        assert wrapped("which?") == "from terminal"
        sink_mod._sink.stop()

    def test_the_terminal_callback_receives_the_original_arguments(self):
        _install(_MissingDesktop())
        seen = {}

        def tui(prompt, options=None):
            seen["prompt"] = prompt
            seen["options"] = options
            return "ok"

        wrap_ask_callback(tui)("the question", ["x", "y"])
        assert seen == {"prompt": "the question", "options": ["x", "y"]}

    def test_the_terminal_callback_is_used_when_no_options_are_given(self):
        _install(_MissingDesktop())
        seen = {}
        wrap_ask_callback(lambda p, options=None: seen.setdefault("o", options) or "ok")("q")
        assert seen["o"] is None
