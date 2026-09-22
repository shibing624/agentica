# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The conversation fields on the wire: ``prompt``, ``answer``,
``answered_at``. The desktop app shows "what was asked / what came back / when",
which a bubble can render — so these must be present, short, and honest about
what they contain.
"""
import time

import pytest

from agentica.notify import install_sink, notify_sink_dispatch, reset_sink_for_tests
from agentica.notify.config import NotifyConfig
from agentica.run.events import RunEventRecord, RunEventType

from tests.notify.test_notify_sink import _FakeDesktop


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


def _drain(desktop, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if len(desktop.requests) >= expected:
            return
        time.sleep(0.02)


class _Resp:
    def __init__(self, content):
        self.content = content


class _Agent:
    def __init__(self, answer=None):
        self.run_response = _Resp(answer)
        self.name = "deepagent"


def _started(source_query):
    return RunEventRecord(
        run_id="r1",
        event_type=RunEventType.run_started,
        payload={"agent_name": "deepagent", "source_query": source_query},
    )


def _completed(ts=None):
    kwargs = {"timestamp": ts} if ts is not None else {}
    return RunEventRecord(
        run_id="r1",
        event_type=RunEventType.run_completed,
        payload={"duration_seconds": 1.0, "had_response": True},
        **kwargs,
    )


def _run(desktop, record, agent, want=1):
    install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path))
    notify_sink_dispatch(record, agent=agent)
    _drain(desktop, want)
    return [r["json"] for r in desktop.requests]


def _only(events, name):
    return [e for e in events if e["event"] == name][0]


class TestTheQuestionAndAnswerGoOnTheWire:
    def test_the_prompt_goes_out_on_started(self):
        desktop = _FakeDesktop()
        events = _run(desktop, _started("auth 为什么失败？"), _Agent())
        assert _only(events, "run.started")["payload"]["prompt"] == "auth 为什么失败？"

    def test_the_answer_goes_out_on_completed(self):
        desktop = _FakeDesktop()
        events = _run(desktop, _completed(), _Agent("token 提前 5 秒过期，已修。"))
        assert _only(events, "run.completed")["payload"]["answer"] == "token 提前 5 秒过期，已修。"

    def test_a_long_answer_is_clipped_and_the_cut_is_visible(self):
        """A bubble is not a reader — but a silent cut is a lie about length."""
        desktop = _FakeDesktop()
        events = _run(desktop, _completed(), _Agent("先看 token 刷新。" + "很长" * 800))
        answer = _only(events, "run.completed")["payload"]["answer"]
        assert answer.startswith("先看 token 刷新。")
        assert len(answer) == 501, "500 chars plus the ellipsis marker"
        assert answer.endswith("…")

    def test_a_short_answer_is_not_annotated(self):
        desktop = _FakeDesktop()
        events = _run(desktop, _completed(), _Agent("做完了。"))
        assert _only(events, "run.completed")["payload"]["answer"] == "做完了。"

    def test_no_answer_field_when_there_was_no_reply(self):
        """A cancelled turn must not look like it answered with nothing."""
        desktop = _FakeDesktop()
        events = _run(desktop, _completed(), _Agent(None))
        assert "answer" not in _only(events, "run.completed")["payload"]

    def test_a_prompt_over_the_limit_is_clipped_too(self):
        desktop = _FakeDesktop()
        events = _run(desktop, _started("问" * 900), _Agent())
        prompt = _only(events, "run.started")["payload"]["prompt"]
        assert len(prompt) == 501 and prompt.endswith("…")


class TestAnsweredAtIsTheRunTime:
    """``answered_at`` and the envelope ``ts`` answer different questions.

    The envelope is stamped when the event is *sent*, which is not when the
    answer was produced whenever a completion was held back for a goal. A UI
    showing "when did it answer" needs the run's own timestamp.
    """

    def test_answered_at_is_the_run_time(self):
        desktop = _FakeDesktop()
        events = _run(desktop, _completed(ts=1000.0), _Agent("好了"))
        done = _only(events, "run.completed")
        assert done["payload"]["answered_at"] == 1000.0

    def test_the_envelope_ts_is_a_different_number(self):
        desktop = _FakeDesktop()
        events = _run(desktop, _completed(ts=1000.0), _Agent("好了"))
        done = _only(events, "run.completed")
        assert done["ts"] != 1000.0, "the envelope is stamped at send time"
