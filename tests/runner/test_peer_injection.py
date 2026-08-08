# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Runner-side delivery of cross-session peer messages.
"""
from unittest.mock import MagicMock

from agentica.model.message import Message
from agentica.peers import PeerMessage
from agentica.runner import Runner


def _agent(messages):
    """An agent whose peer channel returns ``messages`` once, then nothing."""
    agent = MagicMock()
    agent.peer_session.drain.side_effect = [messages, []]
    return agent


def _message(text="schema changed", name="alpha", peer_id="abcd1234"):
    return PeerMessage(text=text, from_name=name, from_peer_id=peer_id, to_peer_id="beef")


def test_no_peer_channel_is_a_noop():
    agent = MagicMock()
    agent.peer_session = None
    messages = [Message(role="user", content="hi")]

    Runner._inject_peer_messages(messages, agent)

    assert [m.content for m in messages] == ["hi"]


def test_an_empty_mailbox_leaves_the_request_untouched():
    agent = _agent([])
    messages = [Message(role="tool", content="tool output", tool_call_id="t1")]

    Runner._inject_peer_messages(messages, agent)

    assert len(messages) == 1
    assert messages[0].content == "tool output"


def test_a_message_folds_into_the_trailing_tool_result():
    agent = _agent([_message()])
    messages = [
        Message(role="assistant", content="", tool_calls=[{"id": "t1"}]),
        Message(role="tool", content="tool output", tool_call_id="t1"),
    ]

    Runner._inject_peer_messages(messages, agent)

    # Folding (rather than appending a user turn) is what keeps the request from
    # ending on two consecutive user-role turns, which strict providers reject.
    assert len(messages) == 2
    assert messages[-1].role == "tool"
    assert "tool output" in messages[-1].content
    assert "schema changed" in messages[-1].content
    assert "alpha" in messages[-1].content


def test_a_message_becomes_a_user_turn_when_there_is_no_tool_result():
    agent = _agent([_message()])
    messages = [Message(role="user", content="do the thing")]

    Runner._inject_peer_messages(messages, agent)

    assert len(messages) == 2
    assert messages[-1].role == "user"
    assert "schema changed" in messages[-1].content


def test_several_messages_are_delivered_together():
    agent = _agent([_message(text="first", name="alpha"), _message(text="second", name="beta")])
    messages = [Message(role="user", content="go")]

    Runner._inject_peer_messages(messages, agent)

    injected = messages[-1].content
    assert "first" in injected and "second" in injected
    assert "alpha" in injected and "beta" in injected


def test_delivery_happens_once_per_inference():
    agent = _agent([_message()])
    messages = [Message(role="user", content="go")]

    Runner._inject_peer_messages(messages, agent)
    Runner._inject_peer_messages(messages, agent)

    assert sum(1 for m in messages if "schema changed" in (m.content or "")) == 1


def test_a_broken_mailbox_does_not_kill_the_turn():
    agent = MagicMock()
    agent.peer_session.drain.side_effect = OSError("mailbox unreadable")
    messages = [Message(role="user", content="go")]

    Runner._inject_peer_messages(messages, agent)

    assert [m.content for m in messages] == ["go"]
