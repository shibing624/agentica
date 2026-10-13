# -*- coding: utf-8 -*-
"""``acp_agent``: the ACP client against a scripted agent (tests/tools/fake_acp_agent.py).

Every call runs in its own ``asyncio.run()`` on purpose: the interactive CLI
gives each turn a fresh loop, and a session must survive from one to the next.
"""
import asyncio
import json
import os
import sys
import time
import weakref
from pathlib import Path

import pytest

from agentica.agent.approvals import (
    SessionGrants,
    _approval_options,
    classify,
    describe_approval,
)
from agentica.tools.acp_agent_tool import (
    ACP_AGENT_TOOL,
    ACP_PERMISSION_TOOL,
    DEFAULT_RUNNERS,
    MAX_SESSIONS,
    ACPAgentTool,
    ACPSessionRegistry,
    merge_runners,
)
from agentica.tools.acp_client import ACPClientError, select_permission_outcome
from agentica.tools.base import Function, FunctionCall

FAKE_AGENT = str(Path(__file__).with_name("fake_acp_agent.py"))


class _FakeAgent:
    def __init__(self, decision=None):
        self.decision = decision
        self.calls = []
        if decision is None:
            self.approve = None

    async def approve(self, fc):
        self.calls.append(fc)
        return self.decision


@pytest.fixture
def record(tmp_path):
    return tmp_path / "record.jsonl"


@pytest.fixture
def registry():
    reg = ACPSessionRegistry()
    yield reg
    reg.close_all()


def _tool(registry, record, work_dir, agent=None):
    runners = {"fake": {"command": sys.executable, "args": [FAKE_AGENT],
                        "env": {"FAKE_ACP_RECORD": str(record)}}}
    tool = ACPAgentTool(registry, runners=runners, work_dir=str(work_dir))
    if agent is not None:
        tool.functions[ACP_AGENT_TOOL]._agent_ref = weakref.ref(agent)
    return tool


def _run(coro):
    return asyncio.run(coro)


def _received(record):
    return [json.loads(line) for line in record.read_text().splitlines()]


def _session_id(report):
    return report.split(" session ")[1].split(" ")[0]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pid, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_start_streams_text_tool_calls_and_plan(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    report = _run(tool.acp_agent(action="start", runner="fake", message="echo:hello world"))
    lines = report.splitlines()
    assert lines[0].startswith("[fake session s-") and lines[0].endswith("stopReason: end_turn]")
    assert lines[1] == "hello world"  # both chunks, in order, nothing else
    assert "- [completed] execute: printf hi > a.txt" in lines
    assert "- [completed] write a.txt" in lines
    methods = [r["method"] for r in _received(record) if r["method"]]
    assert methods[:3] == ["initialize", "session/new", "session/prompt"]


def test_follow_up_reuses_the_process_and_session_across_loops(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    first = _run(tool.acp_agent(action="start", runner="fake", message="pid"))
    session_id = _session_id(first)
    pid_1, sid_1 = first.splitlines()[1].split()
    second = _run(tool.acp_agent(action="message", session_id=session_id, message="pid"))
    pid_2, sid_2 = second.splitlines()[1].split()
    assert (pid_2, sid_2) == (pid_1, sid_1) == (pid_1, session_id)
    assert sum(r["method"] == "session/new" for r in _received(record)) == 1


@pytest.mark.parametrize(
    "decision,option",
    [("allow", "once"), ("allow_prefix", "once"), ("deny", "no")],
)
def test_permission_goes_through_the_agents_approver(registry, record, tmp_path, decision, option):
    agent = _FakeAgent(decision)
    tool = _tool(registry, record, tmp_path, agent=agent)
    report = _run(tool.acp_agent(action="start", runner="fake", message="permission"))
    reply = next(r["permission_reply"] for r in _received(record) if "permission_reply" in r)
    assert reply["result"] == {"outcome": {"outcome": "selected", "optionId": option}}
    (fc,) = agent.calls
    assert fc.function.name == ACP_PERMISSION_TOOL
    assert fc.call_id == "acp:p1"
    assert fc.arguments == {"runner": "fake", "title": "rm -rf build", "kind": "delete",
                            "paths": ["/w/build"]}
    allowed = 0 if decision == "deny" else 1
    assert f"Permission requests: {allowed} allowed, {1 - allowed} denied" in report


def test_permission_without_an_approver_is_rejected(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path, agent=_FakeAgent(None))
    _run(tool.acp_agent(action="start", runner="fake", message="permission"))
    reply = next(r["permission_reply"] for r in _received(record) if "permission_reply" in r)
    assert reply["result"]["outcome"] == {"outcome": "selected", "optionId": "no"}


def test_unknown_agent_request_gets_method_not_found(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    report = _run(tool.acp_agent(action="start", runner="fake", message="unknown"))
    reply = next(r["unknown_reply"] for r in _received(record) if "unknown_reply" in r)
    assert reply["error"]["code"] == -32601
    assert report.splitlines()[1] == "still here"


def test_a_line_larger_than_the_asyncio_stream_limit(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    report = _run(tool.acp_agent(action="start", runner="fake", message="big"))
    assert "chars omitted" in report  # arrived whole, then clipped for the model


def test_crash_mid_prompt_fails_fast_and_drops_the_session(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    started = time.monotonic()
    with pytest.raises(ACPClientError, match=r"exited with code 3[\s\S]*fake agent: boom"):
        _run(tool.acp_agent(action="start", runner="fake", message="crash"))
    assert time.monotonic() - started < 10
    assert len(registry) == 0


def test_timeout_cancels_the_turn_and_keeps_the_session(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    report = _run(tool.acp_agent(action="start", runner="fake", message="hang", timeout=1))
    assert "stopReason: cancelled" in report
    assert "Timed out" in report
    assert any(r["method"] == "session/cancel" for r in _received(record))
    session_id = _session_id(report)
    again = _run(tool.acp_agent(action="message", session_id=session_id, message="echo:ok"))
    assert again.splitlines()[1] == "ok"


def test_cancelling_the_call_sends_session_cancel(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    first = _run(tool.acp_agent(action="start", runner="fake", message="pid"))
    session_id = _session_id(first)

    async def cancel_midway():
        task = asyncio.ensure_future(
            tool.acp_agent(action="message", session_id=session_id, message="hang")
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _run(cancel_midway())
    deadline = time.monotonic() + 5
    while not any(r["method"] == "session/cancel" for r in _received(record)):
        assert time.monotonic() < deadline, "session/cancel never reached the agent"
        time.sleep(0.05)


def test_close_stops_the_process(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    report = _run(tool.acp_agent(action="start", runner="fake", message="pid"))
    session_id = _session_id(report)
    pid = int(report.splitlines()[1].split()[0])
    assert _run(tool.acp_agent(action="close", session_id=session_id)) == (
        f"Closed acp_agent session {session_id}."
    )
    assert _wait_dead(pid)
    with pytest.raises(ValueError, match="no open acp_agent session"):
        _run(tool.acp_agent(action="message", session_id=session_id, message="pid"))


def test_close_kills_the_whole_process_group_promptly(registry, record, tmp_path):
    runners = {"fake": {"command": sys.executable, "args": [FAKE_AGENT],
                        "env": {"FAKE_ACP_RECORD": str(record), "FAKE_ACP_GRANDCHILD": "1"}}}
    tool = ACPAgentTool(registry, runners=runners, work_dir=str(tmp_path))
    report = _run(tool.acp_agent(action="start", runner="fake", message="pid"))
    grandchild = next(r["grandchild"] for r in _received(record) if "grandchild" in r)
    assert _alive(grandchild)
    started = time.monotonic()
    _run(tool.acp_agent(action="close", session_id=_session_id(report)))
    assert time.monotonic() - started < 2  # SIGTERM answered; no wait for the SIGKILL fallback
    assert _wait_dead(grandchild)


def test_close_all_stops_every_process(record, tmp_path):
    registry = ACPSessionRegistry()
    tool = _tool(registry, record, tmp_path)
    pids = []
    for _ in range(2):
        report = _run(tool.acp_agent(action="start", runner="fake", message="pid"))
        pids.append(int(report.splitlines()[1].split()[0]))
    registry.close_all()
    assert all(_wait_dead(pid) for pid in pids)
    assert len(registry) == 0


def test_session_cap(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    for _ in range(MAX_SESSIONS):
        _run(tool.acp_agent(action="start", runner="fake", message="pid"))
    with pytest.raises(ValueError, match="already open"):
        _run(tool.acp_agent(action="start", runner="fake", message="pid"))
    listing = _run(tool.acp_agent(action="list"))
    assert f"Open sessions ({MAX_SESSIONS}/{MAX_SESSIONS})" in listing


def test_unknown_runner_and_missing_command(registry, record, tmp_path):
    tool = _tool(registry, record, tmp_path)
    with pytest.raises(ValueError, match="unknown runner"):
        _run(tool.acp_agent(action="start", runner="nope", message="x"))
    missing = ACPAgentTool(registry, runners={"gone": {"command": "no-such-acp-binary", "args": []}},
                           work_dir=str(tmp_path))
    with pytest.raises(ACPClientError, match="not found"):
        _run(missing.acp_agent(action="start", runner="gone", message="x"))
    assert len(registry) == 0


# --- pure pieces ----------------------------------------------------------


def test_outcome_mapping_prefers_once_and_falls_back():
    opts = [{"optionId": "a", "kind": "allow_always"}, {"optionId": "r", "kind": "reject_always"}]
    assert select_permission_outcome(opts, "allow")["outcome"]["optionId"] == "a"
    assert select_permission_outcome(opts, "deny")["outcome"]["optionId"] == "r"
    assert select_permission_outcome([], "allow") == {"outcome": {"outcome": "cancelled"}}


def test_merge_runners_overrides_by_name():
    runners = merge_runners({"codex": {"command": "/opt/codex-acp"},
                             "mine": {"command": "my-agent", "args": ["--acp"], "env": {"K": "v"}}})
    assert runners["codex"] == {"command": "/opt/codex-acp", "args": [], "env": {}}
    assert runners["mine"]["args"] == ["--acp"] and runners["mine"]["env"] == {"K": "v"}
    assert runners["claude_code"] == DEFAULT_RUNNERS["claude_code"]


def _fc(name, arguments):
    fn = Function(name=name)
    fn.entrypoint = lambda **kwargs: None
    return FunctionCall(function=fn, arguments=arguments, call_id="c1")


@pytest.mark.parametrize("mode,route", [("ask", "ask"), ("auto", "ask"), ("allow-all", "allow")])
def test_permission_routing(mode, route):
    fc = _fc(ACP_PERMISSION_TOOL, {"runner": "codex", "title": "t", "kind": "edit", "paths": []})
    grants = SessionGrants()
    grants.add_tool_name(ACP_PERMISSION_TOOL)  # no earlier grant answers for the agent
    assert classify(mode, fc, grants, work_dir=None) == route


@pytest.mark.parametrize("mode,route", [("ask", "ask"), ("auto", "allow"), ("allow-all", "allow")])
def test_acp_agent_itself_is_a_write_in_ask(mode, route):
    fc = _fc(ACP_AGENT_TOOL, {"action": "start", "runner": "codex", "message": "x"})
    assert classify(mode, fc, SessionGrants(), work_dir=None) == route


def test_permission_card():
    fc = _fc(ACP_PERMISSION_TOOL, {"runner": "codex", "title": "rm -rf build", "kind": "delete",
                                   "paths": ["/w/build"]})
    question, preview = describe_approval(fc)
    assert question == "Allow codex to do this?"
    assert preview == "[delete] rm -rf build\n/w/build"
    assert _approval_options(fc) == ("allow", "deny")


# --- wiring ---------------------------------------------------------------


def test_make_approve_parks_the_request_as_a_card_in_auto(registry, record, tmp_path):
    """The real approver: auto mode must still put the agent's request on a card."""
    from agentica.agent.approvals import ApprovalRegistry, make_approve

    cards = ApprovalRegistry()
    published = []

    def publish(pending):
        published.append(pending)
        asyncio.get_running_loop().call_soon(cards.decide, pending.tool_call_id, "allow")

    class _Agent:
        approve = staticmethod(make_approve(
            get_mode=lambda: "auto",
            get_grants=lambda: SessionGrants(),
            get_registry=lambda: cards,
            get_work_dir=lambda: str(tmp_path),
            publish=publish,
            apply_path_grant=lambda path, prefix: None,
        ))

    agent = _Agent()
    tool = _tool(registry, record, tmp_path, agent=agent)
    _run(tool.acp_agent(action="start", runner="fake", message="permission"))
    (pending,) = published
    assert pending.name == ACP_PERMISSION_TOOL
    assert pending.question == "Allow fake to do this?"
    assert pending.options == ("allow", "deny")
    reply = next(r["permission_reply"] for r in _received(record) if "permission_reply" in r)
    assert reply["result"]["outcome"]["optionId"] == "once"


def test_cli_registers_the_tool_only_with_a_session_registry():
    from unittest.mock import MagicMock, patch

    from agentica.cli.runtime import create_agent

    def build(**extra):
        captured = {}

        class FakeDeepAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.tools = []

        with (
            patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
            patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
        ):
            create_agent(
                {"model_provider": "openai", "model_name": "m", "debug": False, "work_dir": None},
                extra_tools=[], workspace=None, skills_registry=None,
                include_ask_user_question=True, **extra,
            )
        return [t for t in captured["tools"] if isinstance(t, ACPAgentTool)]

    reg = ACPSessionRegistry()
    (tool,) = build(acp_sessions=reg)
    assert tool._registry is reg
    assert set(DEFAULT_RUNNERS) <= set(tool._runners)
    assert build() == []
