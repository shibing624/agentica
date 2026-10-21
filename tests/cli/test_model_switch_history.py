# -*- coding: utf-8 -*-
"""A model switch keeps tool rounds; a rebuild of the same model touches nothing."""
import os

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake_anthropic_key")

import pytest
from types import SimpleNamespace
from unittest.mock import patch

from agentica.agent import Agent
from agentica.agent.history_filter import ensure_history_portable, model_identity
from agentica.guardrails import GuardrailOutput, InputGuardrailTripwireTriggered, input_guardrail
from agentica.memory.models import AgentRun
from agentica.memory.working import WorkingMemory
from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message
from agentica.model.openai.chat import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run.response import RunResponse


def _claude_round_memory() -> WorkingMemory:
    history = [
        Message(role="user", content="read a.py"),
        Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "plan", "signature": "sig-from-opus"},
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
            ],
            tool_calls=[
                {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ],
        ),
        Message(
            role="user",
            content=[{"type": "tool_result", "tool_use_id": "t1", "content": "PORT = 9091"}],
        ),
        Message(role="assistant", content="9091"),
    ]
    memory = WorkingMemory()
    memory.add_messages(history)
    memory.add_run(AgentRun(response=RunResponse(messages=list(history))))
    return memory


def _switch(previous, new):
    agent = SimpleNamespace(
        model=new,
        working_memory=_claude_round_memory(),
        _history_model_identity=model_identity(previous),
    )
    before = list(agent.working_memory.runs[0].response.messages)
    ensure_history_portable(agent)
    return before, agent.working_memory


def test_rebuilding_the_same_model_leaves_history_untouched():
    """A profile switch that lands on the same model keeps the signed thinking
    and the exact prefix the provider has cached."""
    before, memory = _switch(
        Claude(id="claude-opus-5", api_key="k", base_url="https://gw/a"),
        Claude(id="claude-opus-5", api_key="k", base_url="https://gw/a"),
    )
    after = memory.runs[0].response.messages
    assert all(a is b for a, b in zip(after, before)) and len(after) == len(before)


def test_another_model_gets_the_tool_round_without_thinking():
    _, memory = _switch(
        Claude(id="claude-opus-5", api_key="k"),
        OpenAIChat(id="gpt-5", api_key="k"),
    )
    messages = memory.runs[0].response.messages
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[1].tool_calls[0]["id"] == "t1"
    assert messages[2].content == "PORT = 9091"
    assert "sig-from-opus" not in repr([m.content for m in messages])


def test_same_id_on_another_endpoint_counts_as_a_switch():
    assert model_identity(Claude(id="claude-opus-5", api_key="k", base_url="https://gw/a")) != model_identity(
        Claude(id="claude-opus-5", api_key="k", base_url="https://gw/b")
    )
    _, memory = _switch(
        Claude(id="claude-opus-5", api_key="k", base_url="https://gw/a"),
        Claude(id="claude-opus-5", api_key="k", base_url="https://gw/b"),
    )
    assert not any(isinstance(m.content, list) for m in memory.runs[0].response.messages)


def test_same_id_on_another_wire_counts_as_a_switch():
    assert model_identity(Claude(id="claude-opus-5", api_key="k")) != model_identity(
        OpenAIChat(id="claude-opus-5", api_key="k")
    )


def test_a_fresh_history_is_left_alone_and_claimed_by_the_model():
    """No request yet (or history replayed from the log, already portable):
    there is nothing model-bound to strip."""
    agent = SimpleNamespace(
        model=OpenAIChat(id="gpt-5", api_key="k"),
        working_memory=_claude_round_memory(),
        _history_model_identity=None,
    )
    before = list(agent.working_memory.runs[0].response.messages)
    assert ensure_history_portable(agent) is False
    assert all(a is b for a, b in zip(agent.working_memory.runs[0].response.messages, before))
    assert agent._history_model_identity == model_identity(agent.model)


# --- Through the runner: the rewrite happens on the next request, not on /model ---


def _scripted(model, seen):
    async def response(messages):
        seen.append(list(messages))
        reply = Message(
            role="assistant",
            content=[
                {"type": "thinking", "thinking": "plan", "signature": f"sig-{model.id}"},
                {"type": "text", "text": "done"},
            ],
        )
        messages.append(reply)
        return ModelResponse(content="done")

    return response


def _run(agent, model, prompt):
    seen = []
    agent.model = model
    with patch.object(model, "response", new=_scripted(model, seen)):
        agent.run_sync(prompt)
    return seen[-1]


def _history_signatures(agent):
    return [
        b.get("signature")
        for run in agent.working_memory.runs
        for m in run.response.messages
        if isinstance(m.content, list)
        for b in m.content
        if isinstance(b, dict) and b.get("type") == "thinking"
    ]


def _opus():
    return Claude(id="claude-opus-5", api_key="k", base_url="https://gw/a")


def test_switching_away_and_back_before_any_request_keeps_signed_thinking():
    """A→B→A with no request in between: the history was never sent to B, so
    A gets its own signed thinking back and the cached prefix stays intact."""
    agent = Agent(model=_opus(), add_history_to_context=True)
    _run(agent, agent.model, "first")
    assert _history_signatures(agent) == ["sig-claude-opus-5"]

    agent.model = OpenAIChat(id="gpt-5", api_key="k")  # /model B, no request
    request = _run(agent, _opus(), "second")  # /model A, then a request

    assert _history_signatures(agent)[0] == "sig-claude-opus-5"
    assert "sig-claude-opus-5" in repr([m.content for m in request])


def test_the_next_request_on_another_model_carries_the_history_over():
    agent = Agent(model=_opus(), add_history_to_context=True)
    _run(agent, agent.model, "first")

    request = _run(agent, OpenAIChat(id="gpt-5", api_key="k"), "second")

    assert "sig-claude-opus-5" not in repr([m.content for m in request])
    assert "done" in repr([m.content for m in request if m.role == "assistant"])
    assert "sig-claude-opus-5" not in _history_signatures(agent)


def test_guardrail_block_on_another_model_does_not_claim_or_rewrite_history():
    @input_guardrail
    def block(ctx, agent, input_data):
        return GuardrailOutput(output_info="blocked", tripwire_triggered=True)

    opus = _opus()
    agent = Agent(model=opus, add_history_to_context=True)
    _run(agent, opus, "first")
    original_identity = agent._history_model_identity

    agent.model = OpenAIChat(id="gpt-5", api_key="k")
    agent.input_guardrails = [block]
    with pytest.raises(InputGuardrailTripwireTriggered):
        agent.run_sync("blocked")

    assert agent._history_model_identity == original_identity
    assert _history_signatures(agent) == ["sig-claude-opus-5"]

    agent.input_guardrails = []
    request = _run(agent, _opus(), "second")
    assert "sig-claude-opus-5" in repr([m.content for m in request])


def test_explicit_messages_request_does_not_rewrite_unused_working_history():
    opus = _opus()
    agent = Agent(model=opus, add_history_to_context=True)
    _run(agent, opus, "first")
    original_identity = agent._history_model_identity

    other = OpenAIChat(id="gpt-5", api_key="k")
    agent.model = other
    with patch.object(other, "response", new=_scripted(other, [])):
        agent.run_sync(messages=[Message(role="user", content="standalone")])

    assert agent._history_model_identity == original_identity
    assert _history_signatures(agent) == ["sig-claude-opus-5"]
