# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for GoalTool.verify_completion — the agent-driven,
evidence-backed completion check. No real LLM calls (criteria mode mocks the
judge model); test mode runs real short shell commands.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agentica.goals import GoalState
from agentica.memory.session_log import SessionLog
from agentica.model.response import ModelResponse
from agentica.tools.goal_tool import GoalTool


def _make_session_log(tmp_path: Path) -> SessionLog:
    return SessionLog(session_id="verify-test", base_dir=str(tmp_path))


def _set_active_goal(log: SessionLog, objective: str = "make it green") -> None:
    state = GoalState(session_id="verify-test", objective=objective)
    state.status = "active"
    log.append_goal(state)


def _judge_model(content: str):
    m = AsyncMock()
    m.response = AsyncMock(return_value=ModelResponse(content=content))
    return m


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------- mode=test
def test_verify_test_green_marks_complete(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    out = _run(tool.verify_completion(mode="test", verify_command="exit 0", summary="done"))
    assert "passed" in out.lower()
    assert log.load_goal()["status"] == "complete"


def test_verify_test_red_stays_active(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    out = _run(
        tool.verify_completion(
            mode="test", verify_command="echo boom >&2; exit 1", summary="try"
        )
    )
    assert "failed" in out.lower()
    assert "boom" in out  # stderr fed back
    assert log.load_goal()["status"] == "active"


def test_verify_test_requires_command(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    out = _run(tool.verify_completion(mode="test", verify_command=""))
    assert "requires verify_command" in out
    assert log.load_goal()["status"] == "active"


def test_verify_test_captures_final_answer(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    _run(
        tool.verify_completion(
            mode="test", verify_command="true", final_answer="THE RESULT"
        )
    )
    assert log.load_goal()["final_answer"] == "THE RESULT"


# ----------------------------------------------------------- mode=criteria
def test_verify_criteria_done_marks_complete(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log, judge_model=_judge_model('{"done": true, "reason": "all met"}'))
    out = _run(
        tool.verify_completion(
            mode="criteria",
            acceptance_criteria="- report written\n- sources cited",
            summary="report at /tmp/r.md with 5 sources",
        )
    )
    assert "passed" in out.lower()
    assert log.load_goal()["status"] == "complete"


def test_verify_criteria_not_done_stays_active(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log, judge_model=_judge_model('{"done": false, "reason": "no sources"}'))
    out = _run(
        tool.verify_completion(
            mode="criteria", acceptance_criteria="- sources cited", summary="draft only"
        )
    )
    assert "failed" in out.lower()
    assert "no sources" in out
    assert log.load_goal()["status"] == "active"


def test_verify_criteria_without_judge_model(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log, judge_model=None)
    out = _run(
        tool.verify_completion(mode="criteria", acceptance_criteria="- x")
    )
    assert "no judge model" in out.lower()
    assert log.load_goal()["status"] == "active"


def test_verify_criteria_requires_criteria(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log, judge_model=_judge_model('{"done": true, "reason": "x"}'))
    out = _run(tool.verify_completion(mode="criteria", acceptance_criteria=""))
    assert "requires acceptance_criteria" in out


# ---------------------------------------------------------------- guards
def test_verify_invalid_mode(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    out = _run(tool.verify_completion(mode="bogus"))
    assert "invalid mode" in out.lower()


def test_verify_no_active_goal(tmp_path):
    log = _make_session_log(tmp_path)
    tool = GoalTool(log)
    out = _run(tool.verify_completion(mode="test", verify_command="true"))
    assert "no standing goal" in out.lower()


def test_update_goal_paused(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    out = _run(tool.update_goal(status="paused", reason="need creds"))
    assert "paused" in out.lower()
    assert log.load_goal()["status"] == "paused"


def test_update_goal_complete_escape_hatch(tmp_path):
    log = _make_session_log(tmp_path)
    _set_active_goal(log)
    tool = GoalTool(log)
    out = _run(tool.update_goal(status="complete", final_answer="ANS"))
    assert "complete" in out.lower()
    g = log.load_goal()
    assert g["status"] == "complete"
    assert g["final_answer"] == "ANS"