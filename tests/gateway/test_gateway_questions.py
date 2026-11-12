# -*- coding: utf-8 -*-
"""ask_user_question on the gateway: one question, answered from Web, IM card, or nobody."""
import asyncio
import json
import os
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from fastapi import HTTPException

from agentica.tools.ask_user_question_tool import AskUserQuestionTool
from agentica_gateway.routes.chat import QuestionAnswerRequest, answer_session_question
from agentica_gateway.services import agent_service as svc_mod
from agentica_gateway.services import live_turn
from agentica_gateway.services.agent_service import AgentService


@pytest.fixture(autouse=True)
def _reset_live_turn():
    live_turn.reset()
    yield
    live_turn.reset()


def _request(owner: str):
    return SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(user_id=owner)))


def _tool(tmp_path, session_id="s1"):
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    svc._loop = asyncio.get_running_loop()
    return AskUserQuestionTool(input_callback=svc._make_ask_callback(session_id, "default"))


async def _next_event(q: asyncio.Queue, name: str) -> dict:
    while True:
        ev = await asyncio.wait_for(q.get(), timeout=5)
        if ev["event"] == name:
            return ev


def test_web_answer_is_what_the_model_gets(tmp_path):
    async def run():
        turn = live_turn.start("s1", owner="default")
        sub = turn.subscribe()
        call = asyncio.create_task(_tool(tmp_path).ask_user_question("部署到哪？", ["staging（推荐）", "prod"]))

        asked = await _next_event(sub, "question_request")
        assert asked["data"]["prompt"] == "部署到哪？"
        assert asked["data"]["options"] == ["staging（推荐）", "prod"]
        qid = asked["data"]["question_id"]

        # Another account cannot answer this session's question.
        with pytest.raises(HTTPException):
            await answer_session_question("s1", qid, QuestionAnswerRequest(answer="prod"), _request("mallory"))

        await answer_session_question("s1", qid, QuestionAnswerRequest(answer="prod"), _request("default"))
        result = json.loads(await asyncio.wait_for(call, timeout=5))
        assert result["response"] == "prod"
        assert "timed_out" not in result

        resolved = await _next_event(sub, "question_resolved")
        assert resolved["data"] == {"question_id": qid, "answer": "prod", "by": "Web"}
        assert turn.open_questions == []

        # Second answer to the same question (e.g. the IM card after the Web) is refused.
        with pytest.raises(HTTPException):
            await answer_session_question("s1", qid, QuestionAnswerRequest(answer="staging"), _request("default"))

    asyncio.run(run())


def test_notifier_answer_wins_and_reconnect_replays_open_question(tmp_path):
    async def run():
        turn = live_turn.start("s1", owner="default")
        seen = []

        async def notifier(question):
            seen.append(question)

        turn.question_notifier = notifier
        call = asyncio.create_task(_tool(tmp_path).ask_user_question("继续吗？", ["是", "否"]))
        while not turn.open_questions:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert [q.prompt for q in seen] == ["继续吗？"]

        # A page that reconnects mid-question gets the card again.
        late = turn.subscribe()
        turn.republish_pending_approvals()
        again = await _next_event(late, "question_request")
        assert again["data"]["question_id"] == seen[0].question_id

        assert turn.answer_question(seen[0].question_id, "否", by="IM")
        assert json.loads(await asyncio.wait_for(call, timeout=5))["response"] == "否"

    asyncio.run(run())


def test_nobody_answers_is_timed_out_not_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(svc_mod, "QUESTION_WAIT_S", 0.2)

    async def run():
        turn = live_turn.start("s1", owner="default")
        sub = turn.subscribe()
        result = json.loads(await asyncio.wait_for(_tool(tmp_path).ask_user_question("选哪个？", ["A", "B"]), timeout=5))
        assert result["timed_out"] is True
        assert result["response"] == ""
        # The card is taken down everywhere.
        assert turn.open_questions == []
        resolved = await _next_event(sub, "question_resolved")
        assert resolved["data"]["answer"] == "" and resolved["data"]["by"] == ""

    asyncio.run(run())


def test_turn_ending_releases_the_question(tmp_path):
    async def run():
        turn = live_turn.start("s1", owner="default")
        sub = turn.subscribe()
        closed = []
        turn.question_closed = lambda qid, by: closed.append((qid, by))
        call = asyncio.create_task(_tool(tmp_path).ask_user_question("还在吗？"))
        while not turn.open_questions:
            await asyncio.sleep(0.01)
        qid = turn.open_questions[0].question_id
        turn.finish("cancelled")
        result = json.loads(await asyncio.wait_for(call, timeout=5))
        assert result["timed_out"] is True
        resolved = await _next_event(sub, "question_resolved")
        assert resolved["data"] == {"question_id": qid, "answer": "", "by": ""}
        assert closed == [(qid, "")]
        # Stop releases first; the route's later finish must not close the card twice.
        turn.release_questions()
        assert closed == [(qid, "")]

    asyncio.run(run())


def test_notifier_failure_still_accepts_an_answer(tmp_path):
    async def run():
        turn = live_turn.start("s1", owner="default")

        async def notifier(_question):
            raise RuntimeError("im down")

        turn.question_notifier = notifier
        call = asyncio.create_task(_tool(tmp_path).ask_user_question("继续？", ["是"]))
        while not turn.open_questions:
            await asyncio.sleep(0.01)
        qid = turn.open_questions[0].question_id
        await answer_session_question("s1", qid, QuestionAnswerRequest(answer="是"), _request("default"))
        assert json.loads(await asyncio.wait_for(call, timeout=5))["response"] == "是"

    asyncio.run(run())


def test_no_live_turn_does_not_block(tmp_path):
    async def run():
        result = json.loads(await asyncio.wait_for(_tool(tmp_path, "nobody").ask_user_question("在吗？"), timeout=5))
        assert result["timed_out"] is True

    asyncio.run(run())


def test_gateway_agent_always_has_the_tool(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("AGENTICA_HOME", str(home))
    monkeypatch.setenv("AGENTICA_PROJECTS_DIR", str(tmp_path / "projects"))
    svc = AgentService(workspace_path=str(tmp_path / "ws"))
    work = tmp_path / "proj"
    work.mkdir()
    svc.set_session_work_dir("web-1", str(work))
    agent = svc._build_agent("web-1")
    asks = [t for t in agent.tools or [] if isinstance(t, AskUserQuestionTool)]
    assert len(asks) == 1
    assert asks[0].input_callback is not None
