# -*- coding: utf-8 -*-
"""A model switch keeps tool rounds; a rebuild of the same model touches nothing."""
import os

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")
os.environ.setdefault("ANTHROPIC_API_KEY", "fake_anthropic_key")

from types import SimpleNamespace

from agentica.cli.commands.helpers import _sanitize_history_for_model_switch, model_identity
from agentica.memory.models import AgentRun
from agentica.memory.working import WorkingMemory
from agentica.model.anthropic.claude import Claude
from agentica.model.message import Message
from agentica.model.openai.chat import OpenAIChat
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
    agent = SimpleNamespace(model=new, working_memory=_claude_round_memory())
    before = list(agent.working_memory.runs[0].response.messages)
    _sanitize_history_for_model_switch(agent, previous)
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
