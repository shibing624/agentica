# -*- coding: utf-8 -*-
"""One-shot repair for Claude system placement errors on OpenAI-compatible gateways."""

import asyncio
from unittest import mock

import pytest

from agentica.model.message import Message
from agentica.model.openai import OpenAIChat


ERROR = "messages.3: role 'system' must precede an 'assistant' message or end the array"


def _messages_with_invalid_mid_system():
    return [
        Message(role="system", content="main instructions"),
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
        Message(role="system", content="directive for the next reply"),
        Message(role="user", content="second question"),
        Message(role="assistant", content="second answer"),
    ]


def test_non_stream_repairs_client_mid_system_and_retries_once():
    model = OpenAIChat(id="claude-via-gateway", api_key="fake_openai_key")
    payloads = []

    async def create(**kwargs):
        payloads.append(kwargs["messages"])
        if len(payloads) == 1:
            raise RuntimeError(ERROR)
        return "ok"

    client = mock.Mock()
    client.chat.completions.create = create
    with mock.patch.object(model, "get_client", return_value=client):
        result = asyncio.run(model.invoke(_messages_with_invalid_mid_system()))
        second_result = asyncio.run(model.invoke(_messages_with_invalid_mid_system()))

    assert result == "ok"
    assert second_result == "ok"
    assert [message["role"] for message in payloads[0]] == [
        "system", "user", "assistant", "system", "user", "assistant",
    ]
    assert [message["role"] for message in payloads[1]] == [
        "system", "user", "assistant", "user", "system", "assistant",
    ]
    assert [message["role"] for message in payloads[2]] == [
        "system", "user", "assistant", "user", "system", "assistant",
    ]


def test_stream_repairs_before_opening_the_second_stream():
    model = OpenAIChat(id="claude-via-gateway", api_key="fake_openai_key", max_retries=0)
    payloads = []

    async def empty_stream():
        if False:
            yield None

    async def create(**kwargs):
        payloads.append(kwargs["messages"])
        if len(payloads) == 1:
            raise RuntimeError(ERROR)
        return empty_stream()

    client = mock.Mock()
    client.chat.completions.create = create

    async def run():
        with mock.patch.object(model, "get_client", return_value=client):
            return [chunk async for chunk in model.invoke_stream(_messages_with_invalid_mid_system())]

    assert asyncio.run(run()) == []
    assert len(payloads) == 2
    assert [message["role"] for message in payloads[1]] == [
        "system", "user", "assistant", "user", "system", "assistant",
    ]


def test_gateway_only_system_error_is_not_blindly_retried():
    model = OpenAIChat(id="claude-via-gateway", api_key="fake_openai_key")
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(ERROR)

    client = mock.Mock()
    client.chat.completions.create = create
    messages = [
        Message(role="system", content="main instructions"),
        Message(role="user", content="question"),
    ]

    with mock.patch.object(model, "get_client", return_value=client), pytest.raises(RuntimeError, match="must precede"):
        asyncio.run(model.invoke(messages))

    assert calls == 1


def test_failed_structural_retry_is_not_sent_a_third_time():
    model = OpenAIChat(id="claude-via-gateway", api_key="fake_openai_key")
    calls = 0

    async def create(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError(ERROR)

    client = mock.Mock()
    client.chat.completions.create = create
    with mock.patch.object(model, "get_client", return_value=client), pytest.raises(RuntimeError, match="must precede"):
        asyncio.run(model.invoke(_messages_with_invalid_mid_system()))

    assert calls == 2
