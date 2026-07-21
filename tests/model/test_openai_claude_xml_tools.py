"""Regression tests for Claude tool calls leaked through OpenAI-compatible APIs."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.tools.base import Function


def _claude_proxy_model() -> OpenAIChat:
    return OpenAIChat(id="claude-opus-4-8", api_key="fake_openai_key")


def test_claude_model_uses_text_tool_call_compatibility():
    """Claude model IDs on OpenAIChat must enable the XML fallback."""
    assert _claude_proxy_model()._uses_claude_text_tool_call_compatibility()
    assert not OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")._uses_claude_text_tool_call_compatibility()


def test_claude_invoke_text_becomes_standard_function_call():
    """A proxy-leaked ``<invoke>`` block becomes an executable OpenAI call."""
    calls = _claude_proxy_model()._parse_claude_text_tool_calls(
        "course\n"
        '<invoke name="read_file">\n'
        '<parameter name="file_path">/tmp/example.py</parameter>\n'
        "</invoke>"
    )

    assert calls is not None
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "file_path": "/tmp/example.py"
    }


def test_malformed_claude_invoke_text_raises_protocol_error():
    """Incomplete invoke markup must fail rather than polluting history."""
    with pytest.raises(ValueError, match="malformed Claude XML tool call"):
        _claude_proxy_model()._parse_claude_text_tool_calls(
            '<invoke name="read_file"><parameter name="file_path">/tmp/example.py'
        )


def test_streaming_claude_xml_call_is_not_yielded_as_assistant_text():
    """A streaming XML tool call must execute as a tool, never as CLI text."""
    model = _claude_proxy_model()
    model.run_tools = True
    model.functions = {
        "read_file": Function(
            name="read_file",
            entrypoint=lambda file_path: f"read {file_path}",
        )
    }
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(
                    content=(
                        "course\n"
                        '<invoke name="read_file">'
                        '<parameter name="file_path">/tmp/example.py</parameter>'
                        "</invoke>"
                    ),
                    reasoning_content=None,
                    audio=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )

    async def fake_invoke_stream(messages):
        yield chunk

    model.invoke_stream = fake_invoke_stream
    messages = [Message(role="user", content="Read the file")]

    async def run():
        return [response async for response in model.response_stream(messages)]

    responses = asyncio.run(run())
    assistant = next(message for message in messages if message.role == "assistant")
    assert assistant.content is None
    assert assistant.tool_calls[0]["function"]["name"] == "read_file"
    assert "<invoke" not in "".join(str(response.content or "") for response in responses)


def test_non_streaming_claude_xml_call_is_not_saved_as_assistant_text():
    """The non-streaming path must normalize XML calls identically."""
    model = _claude_proxy_model()
    model.run_tools = True
    model.functions = {
        "read_file": Function(
            name="read_file",
            entrypoint=lambda file_path: f"read {file_path}",
        )
    }
    response_message = SimpleNamespace(
        role="assistant",
        content=(
            '<invoke name="read_file">'
            '<parameter name="file_path">/tmp/example.py</parameter>'
            "</invoke>"
        ),
        tool_calls=None,
        audio=None,
        reasoning_content=None,
        reasoning=None,
        model_dump=lambda **_kwargs: {},
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=response_message, finish_reason="tool_calls")],
        usage=None,
    )

    async def fake_invoke(messages):
        return response

    model.invoke = fake_invoke
    messages = [Message(role="user", content="Read the file")]
    asyncio.run(model.response(messages))

    assistant = next(message for message in messages if message.role == "assistant")
    assert assistant.content is None
    assert assistant.tool_calls[0]["function"]["name"] == "read_file"


def test_non_claude_models_keep_incremental_text_streaming():
    """The Claude compatibility buffer must not delay standard OpenAI output."""
    model = OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key")
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content="Hello",
                        reasoning_content=None,
                        audio=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        content=" world",
                        reasoning_content=None,
                        audio=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        ),
    ]

    async def fake_invoke_stream(messages):
        for chunk in chunks:
            yield chunk

    model.invoke_stream = fake_invoke_stream

    async def run():
        return [
            response.content
            async for response in model.response_stream([Message(role="user", content="Hi")])
            if response.content
        ]

    assert asyncio.run(run()) == ["Hello", " world"]
