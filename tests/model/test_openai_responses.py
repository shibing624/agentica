# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the OpenAI Responses API adapter and config routing.
"""

import asyncio
from types import SimpleNamespace

import pytest
from openai.types.responses.response import Response
from openai.types.responses.compacted_response import CompactedResponse

from agentica.cli.runtime import get_model
from agentica.cli.setup import _validate_profile
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat, OpenAIResponses


def _response(output, *, status="completed", usage=True, incomplete_reason=None, tools=None):
    return Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 1,
            "error": None,
            "incomplete_details": {"reason": incomplete_reason} if incomplete_reason else None,
            "instructions": None,
            "metadata": None,
            "model": "gpt-5.6-sol",
            "object": "response",
            "output": output,
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": tools if tools is not None else [],
            "top_p": None,
            "background": False,
            "completed_at": 2,
            "conversation": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "prompt_cache_key": None,
            "prompt_cache_retention": None,
            "reasoning": {"effort": "high"},
            "safety_identifier": None,
            "service_tier": "default",
            "status": status,
            "text": {"format": {"type": "text"}},
            "top_logprobs": 0,
            "truncation": "disabled",
            "usage": (
                {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 0},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 15,
                }
                if usage
                else None
            ),
            "user": None,
        }
    )


class _AsyncEvents:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        self._iterator = iter(self._events)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class _FakeResponses:
    def __init__(self, response, stream_events=None, compacted_response=None):
        self.response = response
        self.stream_events = stream_events
        self.compacted_response = compacted_response
        self.calls = []
        self.compact_calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _AsyncEvents(self.stream_events or [])
        return self.response

    async def compact(self, **kwargs):
        self.compact_calls.append(kwargs)
        return self.compacted_response


class _FakeClient:
    def __init__(self, response, stream_events=None, compacted_response=None):
        self.responses = _FakeResponses(response, stream_events, compacted_response)


def _tool_response():
    return _response(
        [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "encrypted-reasoning-state",
                "summary": [{"type": "summary_text", "text": "Need the probe tool."}],
                "status": "completed",
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "venus_probe",
                "arguments": "{\"value\":\"ping\"}",
                "status": "completed",
            },
        ]
    )


def _compacted_response():
    return CompactedResponse.model_validate(
        {
            "id": "resp_compact_test",
            "created_at": 3,
            "object": "response.compaction",
            "output": [
                {
                    "id": "cmp_1",
                    "type": "compaction",
                    "encrypted_content": "opaque-compacted-state",
                }
            ],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 7},
                "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 5},
                "total_tokens": 120,
            },
        }
    )


def test_request_maps_reasoning_tools_and_replays_response_items():
    fake_client = _FakeClient(_tool_response())
    model = OpenAIResponses(
        id="gpt-5.6-sol",
        api_key="test",
        reasoning="high",
        max_tokens=4096,
        client=fake_client,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "venus_probe",
                    "description": "Return a probe value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        run_tools=False,
    )
    messages = [Message(role="user", content="Call the probe tool.")]

    result = asyncio.run(model.response(messages))

    request = fake_client.responses.calls[0]
    assert request["reasoning"] == {"effort": "high"}
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["max_output_tokens"] == 4096
    assert request["tools"] == [
        {
            "type": "function",
            "name": "venus_probe",
            "description": "Return a probe value.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        }
    ]
    assert "messages" not in request
    assert request["input"] == [{"role": "user", "content": "Call the probe tool."}]

    assistant = messages[-1]
    assert result.finish_reason == "tool_calls"
    assert result.reasoning_content == "Need the probe tool."
    assert assistant.tool_calls[0]["id"] == "call_1"
    assert assistant.provider_data["object"] == "response"
    assert assistant.metrics["input_tokens"] == 10
    assert assistant.metrics["completion_tokens_details"]["reasoning_tokens"] == 3

    messages.append(Message(role="tool", tool_call_id="call_1", content="probe-ok"))
    replay = model.format_messages(messages)
    assert [item["type"] for item in replay[1:]] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert replay[1]["encrypted_content"] == "encrypted-reasoning-state"
    assert replay[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "probe-ok",
    }


def test_request_preserves_custom_include_with_encrypted_reasoning():
    model = OpenAIResponses(
        id="gpt-5.6-sol",
        api_key="test",
        reasoning="high",
        request_params={"include": ["web_search_call.action.sources"]},
    )

    assert model.request_kwargs["include"] == [
        "web_search_call.action.sources",
        "reasoning.encrypted_content",
    ]


def test_provider_data_keeps_only_replayable_parts_of_the_response():
    """The Response echoes the whole request; only ``object``/``output`` is read.

    Persisting that echo put a full tool schema on every assistant entry — 89%
    of the provider_data bytes in a real transcript corpus — for something
    nothing reads: replay uses ``object``/``output``, the wire allowlist never
    sends provider_data at all, and Responses stateful chaining rides on
    ``provider_checkpoint``, a separate field.
    """
    fat_schema = [
        {
            "type": "function",
            "name": f"tool_{i}",
            "description": "x" * 500,
            "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
        }
        for i in range(8)
    ]
    response = _response(
        [
            {
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "encrypted-reasoning-state",
                "summary": [{"type": "summary_text", "text": "Need the probe tool."}],
                "status": "completed",
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "venus_probe",
                "arguments": "{\"value\":\"ping\"}",
                "status": "completed",
            },
        ],
        tools=fat_schema,
    )
    assert response.tools, "fixture must carry a request echo for this to mean anything"

    model = OpenAIResponses(id="gpt-5.6-sol", api_key="test",
                            client=_FakeClient(response), run_tools=False)
    messages = [Message(role="user", content="Call the probe tool.")]

    asyncio.run(model.response(messages))
    assistant = messages[-1]

    # Allowlist, not a `tools` blacklist: a future fat echo key cannot creep in.
    assert set(assistant.provider_data) == {"object", "output"}
    assert assistant.provider_data["object"] == "response"

    # ... and what survives is still enough to rebuild the turn for the provider.
    messages.append(Message(role="tool", tool_call_id="call_1", content="probe-ok"))
    replay = model.format_messages(messages)
    assert [item["type"] for item in replay[1:]] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert replay[1]["encrypted_content"] == "encrypted-reasoning-state"


def test_stream_maps_text_reasoning_and_usage():
    response = _response(
        [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Short plan"}],
                "status": "completed",
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hello", "annotations": []}],
            },
        ]
    )
    events = [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="Short plan"),
        SimpleNamespace(type="response.output_text.delta", delta="hello"),
        SimpleNamespace(type="response.completed", response=response),
    ]
    fake_client = _FakeClient(response, events)
    model = OpenAIResponses(
        id="gpt-5.6-sol",
        api_key="test",
        reasoning="high",
        client=fake_client,
    )
    messages = [Message(role="user", content="Say hello.")]

    async def collect():
        return [chunk async for chunk in model.response_stream(messages)]

    chunks = asyncio.run(collect())

    assert [chunk.reasoning_content for chunk in chunks if chunk.reasoning_content] == ["Short plan"]
    assert [chunk.content for chunk in chunks if chunk.content] == ["hello"]
    assert messages[-1].content == "hello"
    assert messages[-1].reasoning_content == "Short plan"
    assert model.last_finish_reason == "stop"
    assert model.usage.requests == 1


def test_native_compact_request_and_canonical_checkpoint_replay():
    fake_client = _FakeClient(_response([]), compacted_response=_compacted_response())
    model = OpenAIResponses(
        id="gpt-5.6-sol",
        api_key="test",
        base_url="https://v2.open.venus.woa.com/llmproxy/v1",
        client=fake_client,
    )
    messages = [
        Message(role="system", content="You are precise."),
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
    ]

    result = asyncio.run(model.compact_context(messages, instructions="Keep file paths."))

    request = fake_client.responses.compact_calls[0]
    assert request == {
        "model": "gpt-5.6-sol",
        "input": [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ],
        "instructions": "Keep file paths.",
    }
    assert result.checkpoint["type"] == "openai_responses_compaction"
    assert result.checkpoint["base_url"] == "https://v2.open.venus.woa.com/llmproxy/v1"
    assert result.checkpoint["output"][0]["encrypted_content"] == "opaque-compacted-state"
    assert result.usage["total_tokens"] == 120
    assert model.usage.requests == 1
    assert model.usage.total_tokens == 120
    assert model.usage.input_tokens_details.cache_creation_tokens == 7

    messages[-1].provider_checkpoint = result.checkpoint
    messages.append(Message(role="user", content="new question"))
    replay = model.format_messages(messages)
    assert replay == [
        {"role": "system", "content": "You are precise."},
        {
            "id": "cmp_1",
            "encrypted_content": "opaque-compacted-state",
            "type": "compaction",
        },
        {"role": "user", "content": "new question"},
    ]


def test_native_checkpoint_is_ignored_by_other_endpoint_identity():
    checkpoint = {
        "type": "openai_responses_compaction",
        "provider": "OpenAI",
        "model": "gpt-5.6-sol",
        "base_url": "https://api.openai.com/v1",
        "output": [{"id": "cmp_1", "type": "compaction", "encrypted_content": "opaque"}],
    }
    model = OpenAIResponses(
        id="gpt-5.6-sol",
        api_key="test",
        base_url="https://v2.open.venus.woa.com/llmproxy/v1",
    )
    messages = [
        Message(role="user", content="portable question"),
        Message(role="assistant", content="portable answer", provider_checkpoint=checkpoint),
        Message(role="user", content="next"),
    ]

    assert model.format_messages(messages) == [
        {"role": "user", "content": "portable question"},
        {"role": "assistant", "content": "portable answer"},
        {"role": "user", "content": "next"},
    ]


def test_runtime_selects_responses_only_for_wire_api():
    responses_model = get_model(
        model_provider="openai",
        model_name="gpt-5.6-sol",
        api_key="test",
        wire_api="responses",
        reasoning="high",
    )
    responses_without_reasoning = get_model(
        model_provider="openai",
        model_name="gpt-5.6-sol",
        api_key="test",
        wire_api="responses",
    )
    chat_model = get_model(
        model_provider="openai",
        model_name="gpt-5.6-sol",
        api_key="test",
        reasoning_effort="high",
    )

    assert isinstance(responses_model, OpenAIResponses)
    assert isinstance(responses_without_reasoning, OpenAIResponses)
    assert isinstance(chat_model, OpenAIChat)
    assert not isinstance(chat_model, OpenAIResponses)

    with pytest.raises(ValueError, match="requires wire_api: responses"):
        get_model(
            model_provider="openai",
            model_name="gpt-5.6-sol",
            api_key="test",
            reasoning="high",
        )

    with pytest.raises(ValueError, match="uses 'reasoning'"):
        get_model(
            model_provider="openai",
            model_name="gpt-5.6-sol",
            api_key="test",
            wire_api="responses",
            reasoning="high",
            reasoning_effort="high",
        )


def test_profile_validation_accepts_responses_reasoning():
    profile = {
        "model_provider": "openai",
        "model_name": "gpt-5.6-sol",
        "base_url": "https://v2.open.venus.woa.com/llmproxy/v1",
        "wire_api": "responses",
        "reasoning": "high",
    }
    assert _validate_profile(profile) == []

    profile["reasoning_effort"] = "high"
    assert "wire_api: responses uses reasoning, not reasoning_effort." in _validate_profile(profile)

    profile.pop("reasoning_effort")
    profile.pop("wire_api")
    assert "reasoning requires wire_api: responses." in _validate_profile(profile)
