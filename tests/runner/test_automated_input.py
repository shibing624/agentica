# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: A turn no person wrote (another agent's message, a job report)
stays in the model's history but out of transcripts shown to people.
"""
import json
import tempfile

from agentica.agent import Agent
from agentica.agent.base import SteerItem
from agentica.cli.commands.session import _qa_blocks_for_display, hydrate_resumed_session
from agentica.memory.session_log import SessionLog
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run.config import RunConfig
from agentica.runner import Runner

PEER_TURN = "[Message from another agent session 'beta'] I renamed the loader"


def _agent(tmp, requests):
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")

    async def fake_response(messages=None, **_kw):
        requests.append([m.to_model_dict() for m in messages])
        messages.append(Message(role="assistant", content=f"answer {len(requests)}"))
        return ModelResponse(content=f"answer {len(requests)}")

    model.response = fake_response
    return Agent(name="t", model=model, session_id="s-auto", session_base_dir=tmp)


def _user_entries(agent):
    text = agent._session_log.path.read_text(encoding="utf-8")
    return [e for e in map(json.loads, text.splitlines()) if e["type"] == "user"]


def test_automated_turn_is_recorded_hidden_from_people_and_kept_for_the_model():
    requests = []
    with tempfile.TemporaryDirectory() as tmp:
        agent = _agent(tmp, requests)
        agent.run_sync("rename the loader")
        agent.run_sync(PEER_TURN, config=RunConfig(automated_input=True))

        entries = _user_entries(agent)
        assert [e["content"] for e in entries] == ["rename the loader", PEER_TURN]
        assert "automated" not in entries[0]
        assert entries[1]["automated"] is True
        # Provenance never reaches the provider.
        assert all("automated" not in m for request in requests for m in request)

        preview = SessionLog.session_preview(str(agent._session_log.path))
        assert preview["user_count"] == 1
        assert preview["recent_users"] == ["rename the loader"]
        # Not a reading for people: fork points are positions in the
        # transcript, and search_session is the model's recall of what it saw.
        assert [m["content"] for m in agent._session_log.list_user_messages()] == [
            PEER_TURN, "rename the loader",
        ]
        assert [q["snippet"] for q in agent._session_log.list_user_questions()] == [
            PEER_TURN, "rename the loader",
        ]

        resumed = _agent(tmp, requests)
        _, runs_built = hydrate_resumed_session(resumed)
        assert runs_built == 2
        messages = [m for run in resumed.working_memory.runs for m in run.response.messages]
        # The model's history still carries the peer turn ...
        assert any(m.role == "user" and m.content == PEER_TURN and m.automated for m in messages)
        # ... the transcript a person reads does not, but keeps both answers.
        assert _qa_blocks_for_display(messages) == [
            ("user", "rename the loader"),
            ("assistant", "answer 1\n\nanswer 2"),
        ]


def test_a_human_turn_after_resume_is_not_marked():
    requests = []
    with tempfile.TemporaryDirectory() as tmp:
        agent = _agent(tmp, requests)
        agent.run_sync(PEER_TURN, config=RunConfig(automated_input=True))
        agent.run_sync("and now a person speaks")

        assert [e.get("automated", False) for e in _user_entries(agent)] == [True, False]


def test_automated_steer_is_not_presented_as_user_guidance():
    agent = Agent()
    agent._begin_steer_window()
    agent.steer("[Background terminal #1 (term_1) finished: exit 0]", automated=True)
    messages = [Message(role="user", content="do the task")]
    Runner._inject_steering(messages, agent)

    injected = messages[-1]
    assert injected.content == "[Background terminal #1 (term_1) finished: exit 0]"
    assert injected.automated


def test_automated_steer_with_an_image_keeps_its_provenance():
    agent = Agent()
    agent._begin_steer_window()
    agent.steer("screenshot from the job", automated=True, images=["/tmp/shot.png"])
    messages = [Message(role="user", content="do the task")]
    Runner._inject_steering(messages, agent)

    injected = messages[-1]
    assert injected.images == ["/tmp/shot.png"]
    assert injected.automated


def test_steer_typed_by_a_person_is_not_automated():
    agent = Agent()
    agent._begin_steer_window()
    agent.steer("look at this", images=["/tmp/shot.png"])
    agent.steer("job report", automated=True)
    messages = [Message(role="user", content="do the task")]
    Runner._inject_steering(messages, agent)

    assert not messages[-1].automated


def test_parked_automated_steer_keeps_its_provenance():
    agent = Agent()
    agent._begin_steer_window()
    agent.steer("typed")
    agent.steer("job report", automated=True)
    agent._end_steer_window()

    assert agent.pop_undelivered_steer() == [
        SteerItem("typed"),
        SteerItem("job report", True, (), True),
    ]
