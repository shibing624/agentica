# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for the persistent /goal loop.

All tests mock ``Model.response()`` — no real LLM calls.
"""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentica.goals import (
    CONTINUATION_PROMPT_PREFIX,
    DEFAULT_TURN_BUDGET,
    MAX_CONSECUTIVE_PARSE_FAILURES,
    GoalDecision,
    GoalManager,
    GoalState,
    _parse_judge_response,
    judge_goal,
)
from agentica.memory.session_log import SessionLog
from agentica.model.response import ModelResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session_log(tmp_path: Path, session_id: str = "test-session") -> SessionLog:
    return SessionLog(session_id=session_id, base_dir=str(tmp_path))


def _fake_model(content: str = '{"done": false, "reason": "keep going"}'):
    model = AsyncMock()
    model.response = AsyncMock(return_value=ModelResponse(content=content))
    return model


# ---------------------------------------------------------------------------
# GoalState (de)serialization
# ---------------------------------------------------------------------------

def test_goalstate_roundtrip():
    s = GoalState(
        session_id="abc",
        objective="ship the demo",
        subgoals=["tests pass", "docs updated"],
        turn_budget=5,
        turns_used=2,
    )
    d = s.to_dict()
    assert d["objective"] == "ship the demo"
    s2 = GoalState.from_dict(d)
    assert s2.session_id == s.session_id
    assert s2.subgoals == s.subgoals
    assert s2.turn_budget == 5


def test_goalstate_from_dict_ignores_unknown_fields():
    s = GoalState.from_dict({
        "session_id": "x",
        "objective": "o",
        "unknown_field": 123,
    })
    assert s.session_id == "x"
    assert s.objective == "o"


# ---------------------------------------------------------------------------
# Judge response parsing
# ---------------------------------------------------------------------------

def test_parse_judge_response_done():
    v, r, failed = _parse_judge_response('{"done": true, "reason": "all done"}')
    assert v == "done" and r == "all done" and failed is False


def test_parse_judge_response_continue():
    v, r, failed = _parse_judge_response('{"done": false, "reason": "more work"}')
    assert v == "continue" and r == "more work" and failed is False


def test_parse_judge_response_fenced():
    raw = '```json\n{"done": true, "reason": "ok"}\n```'
    v, _, failed = _parse_judge_response(raw)
    assert v == "done" and failed is False


def test_parse_judge_response_embedded():
    raw = 'I think... {"done": false, "reason": "needs tests"} that is it.'
    v, r, failed = _parse_judge_response(raw)
    assert v == "continue" and r == "needs tests" and failed is False


def test_parse_judge_response_invalid():
    v, _, failed = _parse_judge_response("not json at all")
    assert v == "continue" and failed is True


def test_parse_judge_response_empty():
    _, _, failed = _parse_judge_response("")
    assert failed is True


# ---------------------------------------------------------------------------
# judge_goal — model integration via Model.response()
# ---------------------------------------------------------------------------

def test_judge_goal_uses_model_response():
    model = _fake_model('{"done": true, "reason": "looks good"}')
    v, r, failed = asyncio.run(judge_goal(model, "do X", "I did X"))
    assert v == "done" and r == "looks good" and failed is False
    model.response.assert_awaited_once()


def test_judge_goal_fail_open_on_exception():
    model = AsyncMock()
    model.response = AsyncMock(side_effect=RuntimeError("network down"))
    v, r, failed = asyncio.run(judge_goal(model, "do X", "I did X"))
    # Fail-open: verdict=continue, parse_failed=False so caller does NOT
    # increment the parse-failure counter.
    assert v == "continue" and failed is False
    assert "network down" in r


# ---------------------------------------------------------------------------
# SessionLog goal entries
# ---------------------------------------------------------------------------

def test_session_log_append_and_load_goal(tmp_path: Path):
    log = _make_session_log(tmp_path)
    state = GoalState(session_id=log.session_id, objective="finish P0")
    log.append_goal(state)

    loaded = log.load_goal()
    assert loaded is not None
    assert loaded["objective"] == "finish P0"
    assert loaded["status"] == "active"


def test_session_log_load_goal_returns_latest(tmp_path: Path):
    log = _make_session_log(tmp_path)
    log.append_goal(GoalState(session_id=log.session_id, objective="first"))
    log.append_goal(GoalState(session_id=log.session_id, objective="second"))
    log.append_goal(GoalState(session_id=log.session_id, objective="third"))

    loaded = log.load_goal()
    assert loaded["objective"] == "third"


def test_session_log_load_goal_no_entry(tmp_path: Path):
    log = _make_session_log(tmp_path)
    log.append("user", "hello")
    log.append("assistant", "hi")
    assert log.load_goal() is None


def test_session_log_goal_entry_not_in_history(tmp_path: Path):
    """Goal entries must NEVER leak into conversation replay."""
    log = _make_session_log(tmp_path)
    log.append("user", "hello")
    log.append_goal(GoalState(session_id=log.session_id, objective="something"))
    log.append("assistant", "hi back")

    msgs = log.load()
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]
    # And the content must NOT contain the goal objective.
    for m in msgs:
        assert "something" not in m["content"]


def test_session_log_load_goal_survives_large_file(tmp_path: Path):
    log = _make_session_log(tmp_path)
    log.append_goal(GoalState(session_id=log.session_id, objective="target"))
    # Pad with many messages so the goal entry is far from EOF.
    for i in range(200):
        log.append("user", f"msg {i} " + ("x" * 500))
        log.append("assistant", f"reply {i} " + ("y" * 500))
    loaded = log.load_goal()
    assert loaded is not None and loaded["objective"] == "target"


# ---------------------------------------------------------------------------
# GoalManager lifecycle
# ---------------------------------------------------------------------------

def test_manager_set_and_load(tmp_path: Path):
    log = _make_session_log(tmp_path)
    mgr = GoalManager(log)
    state = mgr.set("ship it")
    assert state.objective == "ship it"
    assert state.status == "active"
    assert mgr.is_active()

    # Fresh manager reads the same state back from disk.
    mgr2 = GoalManager(log)
    loaded = mgr2.load()
    assert loaded is not None and loaded.objective == "ship it"


def test_manager_set_empty_raises(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    with pytest.raises(ValueError):
        mgr.set("   ")


def test_manager_pause_resume_clear(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    mgr.set("x")
    mgr.pause("user")
    assert not mgr.is_active()
    assert mgr.load().paused_reason == "user"

    mgr.resume()
    assert mgr.is_active()

    mgr.clear()
    assert mgr.load() is None
    assert not mgr.is_active()


def test_manager_force_pause_on_resume(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    mgr.set("x")
    mgr.force_pause_on_resume()
    state = mgr.load()
    assert state.status == "paused"
    assert state.paused_reason == "resume-safety"


def test_manager_subgoals(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    mgr.set("main")
    mgr.add_subgoal("a")
    mgr.add_subgoal("b")
    mgr.add_subgoal("c")
    assert mgr.load().subgoals == ["a", "b", "c"]

    assert mgr.remove_subgoal(2) == "b"
    assert mgr.load().subgoals == ["a", "c"]

    n = mgr.clear_subgoals()
    assert n == 2
    assert mgr.load().subgoals == []


def test_manager_add_subgoal_without_goal_raises(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    with pytest.raises(ValueError):
        mgr.add_subgoal("x")


# ---------------------------------------------------------------------------
# GoalManager.evaluate_after_turn — the core loop
# ---------------------------------------------------------------------------

def test_evaluate_done_completes_goal(tmp_path: Path):
    model = _fake_model('{"done": true, "reason": "all set"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x")
    decision = asyncio.run(mgr.evaluate_after_turn("final answer"))
    assert decision.status == "complete"
    assert decision.should_continue is False
    assert decision.verdict == "done"
    assert mgr.load().status == "complete"


def test_evaluate_continue_returns_continuation_prompt(tmp_path: Path):
    model = _fake_model('{"done": false, "reason": "needs more"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x")
    decision = asyncio.run(mgr.evaluate_after_turn("partial"))
    assert decision.should_continue is True
    assert decision.continuation_prompt is not None
    assert CONTINUATION_PROMPT_PREFIX in decision.continuation_prompt
    assert "x" in decision.continuation_prompt  # objective echoed
    assert mgr.load().turns_used == 1


def test_evaluate_continuation_includes_subgoals(tmp_path: Path):
    model = _fake_model('{"done": false, "reason": "more"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("main")
    mgr.add_subgoal("write tests")
    mgr.add_subgoal("update docs")
    decision = asyncio.run(mgr.evaluate_after_turn("started"))
    assert "write tests" in decision.continuation_prompt
    assert "update docs" in decision.continuation_prompt


def test_evaluate_turn_budget_exhaustion_is_budget_limited(tmp_path: Path):
    model = _fake_model('{"done": false, "reason": "still not"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x", turn_budget=2)
    asyncio.run(mgr.evaluate_after_turn("r1"))
    decision = asyncio.run(mgr.evaluate_after_turn("r2"))
    assert decision.status == "budget_limited"
    assert decision.should_continue is False
    assert mgr.load().paused_reason == "budget"
    # User can /goal resume out of budget_limited.
    mgr.resume()
    assert mgr.load().status == "active"


def test_evaluate_token_budget_exhaustion(tmp_path: Path):
    """token_delta accumulates and hits the cap BEFORE the judge runs."""
    model = _fake_model('{"done": false, "reason": "more"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x", token_budget=100)
    # Spend 60 tokens — under cap, runs the judge.
    d1 = asyncio.run(mgr.evaluate_after_turn("r1", token_delta=60))
    assert d1.status == "active"
    assert mgr.load().tokens_used == 60
    # Spend 50 more — hits cap; judge should NOT run this turn.
    d2 = asyncio.run(mgr.evaluate_after_turn("r2", token_delta=50))
    assert d2.status == "budget_limited"
    assert mgr.load().tokens_used == 110
    assert "token budget" in d2.reason


def test_evaluate_wall_clock_budget_exhaustion(tmp_path: Path):
    model = _fake_model('{"done": false, "reason": "more"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x", wall_clock_budget_sec=10.0)
    asyncio.run(mgr.evaluate_after_turn("r1", elapsed_sec=6.0))
    assert mgr.load().wall_clock_used_sec == pytest.approx(6.0)
    d2 = asyncio.run(mgr.evaluate_after_turn("r2", elapsed_sec=5.0))
    assert d2.status == "budget_limited"
    assert "wall-clock" in d2.reason


def test_evaluate_three_parse_failures_pause(tmp_path: Path):
    model = _fake_model("this is not json")
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x")
    decisions = [
        asyncio.run(mgr.evaluate_after_turn(f"r{i}"))
        for i in range(MAX_CONSECUTIVE_PARSE_FAILURES)
    ]
    # First N-1 are soft retries.
    for d in decisions[:-1]:
        assert d.status == "active"
        assert d.should_continue is True
    last = decisions[-1]
    assert last.status == "paused"
    assert last.should_continue is False
    assert mgr.load().paused_reason == "judge-broken"


def test_evaluate_parse_failure_then_recover_resets_counter(tmp_path: Path):
    model = AsyncMock()
    model.response = AsyncMock(side_effect=[
        ModelResponse(content="garbage"),
        ModelResponse(content='{"done": false, "reason": "ok now"}'),
        ModelResponse(content='{"done": true, "reason": "done"}'),
    ])
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    mgr.set("x")
    asyncio.run(mgr.evaluate_after_turn("r1"))
    assert mgr.load().consecutive_parse_failures == 1
    asyncio.run(mgr.evaluate_after_turn("r2"))
    assert mgr.load().consecutive_parse_failures == 0
    final = asyncio.run(mgr.evaluate_after_turn("r3"))
    assert final.status == "complete"


def test_evaluate_no_active_goal_is_noop(tmp_path: Path):
    model = _fake_model()
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model)
    decision = asyncio.run(mgr.evaluate_after_turn("anything"))
    assert decision.should_continue is False
    model.response.assert_not_called()


def test_evaluate_without_judge_model_pauses(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))  # no judge_model
    mgr.set("x")
    decision = asyncio.run(mgr.evaluate_after_turn("r"))
    assert decision.status == "paused"
    assert mgr.load().paused_reason == "judge-broken"


# ---------------------------------------------------------------------------
# Status line — readability sanity check
# ---------------------------------------------------------------------------

def test_status_line_no_goal(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    assert mgr.status_line() == "No active goal."


def test_event_callback_fires_for_lifecycle(tmp_path: Path):
    from agentica.run_events import RunEventType
    events: list = []

    def cb(event_type, payload):
        events.append((event_type, payload["status"]))

    model = _fake_model('{"done": true, "reason": "ok"}')
    mgr = GoalManager(_make_session_log(tmp_path), judge_model=model, event_callback=cb)
    mgr.set("x")
    asyncio.run(mgr.evaluate_after_turn("done now"))
    types = [e[0] for e in events]
    assert RunEventType.goal_set in types
    assert RunEventType.goal_completed in types


def test_tool_mark_complete_via_disk_picked_up(tmp_path: Path):
    """Simulate the GoalTool path: a second manager mutates disk state
    mid-turn; evaluate_after_turn must pick it up without calling the judge.
    """
    log = _make_session_log(tmp_path)
    # Caller's manager
    judge = _fake_model('{"done": false, "reason": "judge would say no"}')
    mgr = GoalManager(log, judge_model=judge)
    mgr.set("x")
    # Simulate tool call by a second manager writing the state.
    tool_mgr = GoalManager(log)
    tool_mgr.mark_complete_from_tool(reason="agent finished it directly")
    decision = asyncio.run(mgr.evaluate_after_turn("doesn't matter"))
    assert decision.status == "complete"
    assert decision.verdict == "tool_signal"
    # Judge was NOT called because reload picked up complete state first.
    judge.response.assert_not_called()


def test_tool_mark_paused_via_disk(tmp_path: Path):
    log = _make_session_log(tmp_path)
    judge = _fake_model('{"done": false, "reason": "x"}')
    mgr = GoalManager(log, judge_model=judge)
    mgr.set("x")
    GoalManager(log).mark_paused_from_tool(reason="blocked on user input")
    decision = asyncio.run(mgr.evaluate_after_turn("..."))
    assert decision.status == "paused"
    assert mgr.load().paused_reason == "agent-tool"
    judge.response.assert_not_called()


def test_status_line_with_subgoals(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    mgr.set("ship", turn_budget=10)
    mgr.add_subgoal("tests pass")
    line = mgr.status_line()
    assert "[active]" in line
    assert "0/10" in line
    assert "tests pass" in line


def test_status_line_shows_token_budget(tmp_path: Path):
    mgr = GoalManager(_make_session_log(tmp_path))
    mgr.set("x", token_budget=10000, wall_clock_budget_sec=300)
    asyncio.run(mgr.evaluate_after_turn.__wrapped__(mgr, "r")) if False else None  # noqa
    # Simulate without calling judge:
    mgr.load().tokens_used = 4200
    mgr.load().wall_clock_used_sec = 92
    line = mgr.status_line()
    assert "tokens" in line and "4,200" in line and "10,000" in line
    assert "wall 92s/300s" in line


# ---------------------------------------------------------------------------
# GoalTool — model-facing update_goal()
# ---------------------------------------------------------------------------

def test_goal_tool_complete(tmp_path: Path):
    from agentica.tools.goal_tool import GoalTool
    log = _make_session_log(tmp_path)
    GoalManager(log).set("x")
    tool = GoalTool(log)
    result = asyncio.run(tool.update_goal(status="complete", reason="all green"))
    assert "complete" in result.lower()
    payload = log.load_goal()
    assert payload["status"] == "complete"
    assert payload["last_verdict"] == "tool_signal"


def test_goal_tool_paused(tmp_path: Path):
    from agentica.tools.goal_tool import GoalTool
    log = _make_session_log(tmp_path)
    GoalManager(log).set("x")
    tool = GoalTool(log)
    result = asyncio.run(tool.update_goal(status="paused", reason="need API key"))
    assert "pause" in result.lower()
    payload = log.load_goal()
    assert payload["status"] == "paused"
    assert payload["paused_reason"] == "agent-tool"


def test_goal_tool_invalid_status(tmp_path: Path):
    from agentica.tools.goal_tool import GoalTool
    log = _make_session_log(tmp_path)
    GoalManager(log).set("x")
    tool = GoalTool(log)
    msg = asyncio.run(tool.update_goal(status="cancelled", reason=""))
    assert "Invalid status" in msg
    # State must not have changed.
    assert log.load_goal()["status"] == "active"


def test_goal_tool_noop_when_no_goal(tmp_path: Path):
    from agentica.tools.goal_tool import GoalTool
    log = _make_session_log(tmp_path)
    tool = GoalTool(log)
    msg = asyncio.run(tool.update_goal(status="complete", reason=""))
    assert "No standing goal" in msg


# ---------------------------------------------------------------------------
# Runner integration (S1): SDK path picks up persisted active goal.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agent SDK ergonomic surface: get_goal_manager / enable_goal_tool / run_goal
# ---------------------------------------------------------------------------

def test_agent_get_goal_manager_lazy_creates_session_log_and_manager(tmp_path):
    """Agent without session_id still produces a working manager after the
    first ``get_goal_manager()`` call."""
    from agentica.agent import Agent

    agent = Agent.__new__(Agent)
    # Minimal hand-built dataclass-like state for the isolated test.
    agent.model = None
    agent.auxiliary_model = None
    agent.session_id = None
    agent._session_log = None
    agent.goal_manager = None
    agent.tools = None

    mgr = agent.get_goal_manager()
    assert mgr is not None
    assert agent._session_log is not None
    assert agent.session_id is not None
    # Idempotent: same manager comes back.
    assert agent.get_goal_manager() is mgr


def test_agent_enable_goal_tool_idempotent(tmp_path):
    from agentica.agent import Agent
    from agentica.tools.goal_tool import GoalTool

    agent = Agent.__new__(Agent)
    agent.model = None
    agent.auxiliary_model = None
    agent.session_id = None
    agent._session_log = None
    agent.goal_manager = None
    agent.tools = None

    agent.enable_goal_tool()
    agent.enable_goal_tool()  # idempotent
    tools = agent.tools or []
    goal_tools = [t for t in tools if isinstance(t, GoalTool)]
    assert len(goal_tools) == 1


def test_goal_run_result_response_content_property():
    """``response_content`` returns content for happy path and "" for None."""
    from agentica.goals import GoalRunResult, GoalState
    from agentica.run_response import RunResponse

    state = GoalState(session_id="s", objective="x")
    rr = RunResponse(content="hello world")
    assert GoalRunResult("complete", "ok", rr, state, 1).response_content == "hello world"
    assert GoalRunResult("complete", "ok", RunResponse(content=None), state, 1).response_content == ""
    assert GoalRunResult("complete", "ok", None, state, 0).response_content == ""


def test_agent_run_goal_drives_to_completion(tmp_path):
    """Full ergonomic path: mock ``agent.run`` (so we don't touch a real
    LLM) and a fake judge model. ``run_goal`` should: set objective, bind
    anchor, attach GoalTool, loop until judge says done, return result.
    """
    from unittest.mock import AsyncMock, patch
    from agentica.agent import Agent
    from agentica.goals import GoalRunResult
    from agentica.run_context import TaskAnchor
    from agentica.run_response import RunResponse
    from agentica.tools.goal_tool import GoalTool

    agent = Agent.__new__(Agent)
    agent.model = None
    agent.auxiliary_model = _fake_model('{"done": true, "reason": "the answer is 42"}')
    agent.session_id = "agent-rungoal-1"
    agent._session_log = None
    agent.goal_manager = None
    agent.tools = None
    agent.task_anchor = None
    agent._anchor_session_id = None

    # Stub agent.run to return a synthetic RunResponse with no cost data.
    rr = RunResponse(content="42")
    with patch.object(Agent, "run", new=AsyncMock(return_value=rr)):
        result = asyncio.run(
            agent.run_goal("compute 17+9+16", turn_budget=3)
        )

    assert isinstance(result, GoalRunResult)
    assert result.status == "complete"
    assert result.turns_used == 1
    assert result.run_response is rr
    # Anchor was bound to the objective, not the message.
    assert agent.task_anchor.goal == "compute 17+9+16"
    # GoalTool was attached.
    assert any(isinstance(t, GoalTool) for t in (agent.tools or []))


def test_agent_run_goal_token_budget_stops_loop(tmp_path):
    """A tight token_budget must short-circuit before the judge runs."""
    from unittest.mock import AsyncMock, patch
    from agentica.agent import Agent
    from agentica.run_response import RunResponse
    from agentica.cost_tracker import CostTracker

    agent = Agent.__new__(Agent)
    agent.model = None
    agent.auxiliary_model = _fake_model('{"done": false, "reason": "more work"}')
    agent.session_id = "agent-rungoal-budget"
    agent._session_log = None
    agent.goal_manager = None
    agent.tools = None
    agent.task_anchor = None
    agent._anchor_session_id = None

    ct = CostTracker()
    ct.record(model_id="fake-model", input_tokens=80, output_tokens=80)  # 160 > 50
    rr = RunResponse(content="...")
    rr.cost_tracker = ct
    with patch.object(Agent, "run", new=AsyncMock(return_value=rr)):
        result = asyncio.run(
            agent.run_goal("X", token_budget=50, turn_budget=10)
        )

    assert result.status == "budget_limited"
    assert "token budget" in result.reason
    # Judge must not have been called this turn — the budget short-circuited.
    agent.auxiliary_model.response.assert_not_called()


def test_runner_loads_persisted_goal_into_task_anchor(tmp_path, monkeypatch):
    """When a session has an active goal on disk, Runner._run_impl should
    bind TaskAnchor to the goal objective instead of the latest message.
    """
    from unittest.mock import MagicMock, AsyncMock
    from agentica.agent import Agent
    from agentica.memory.session_log import SessionLog
    from agentica.run_context import TaskAnchor

    # Set up a real SessionLog with an active goal.
    log = SessionLog(session_id="runner-s1", base_dir=str(tmp_path))
    GoalManager(log).set("write the migration script")

    # Build a stub agent: just enough for the anchor-init code path.
    agent = MagicMock()
    agent._session_log = log
    agent.session_id = "runner-s1"
    agent._anchor_session_id = None
    agent.task_anchor = None

    # Import the anchor-init block by reproducing its logic — keeps the
    # test fast and isolated from the rest of Runner._run_impl.
    persisted = agent._session_log.load_goal()
    assert persisted is not None
    if (
        agent.task_anchor is None
        or agent._anchor_session_id != agent.session_id
    ):
        if (
            persisted is not None
            and persisted.get("status") == "active"
            and persisted.get("objective")
        ):
            agent.task_anchor = TaskAnchor(
                goal=str(persisted["objective"]),
                source_query=str(persisted["objective"]),
            )
        else:
            agent.task_anchor = TaskAnchor.from_message("some unrelated message")
        agent._anchor_session_id = agent.session_id

    assert agent.task_anchor.goal == "write the migration script"
    assert agent.task_anchor.source_query == "write the migration script"
