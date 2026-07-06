import asyncio
from types import SimpleNamespace

from agentica.model.message import Message
from agentica.model.metrics import Metrics
from agentica.model.openai import OpenAIChat
from agentica import DeepSeekChat, NvidiaChat


class FakeDeepSeekMessage:
    role = "assistant"
    content = "final answer"
    tool_calls = None
    audio = None
    reasoning_content = "thinking trace"

    def model_dump(self, exclude_none=True):
        return {
            "role": self.role,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "provider_extra_field": "future-compatible value",
        }


def test_assistant_message_keeps_reasoning_content_for_next_request():
    message = Message(
        role="assistant",
        content="final answer",
        reasoning_content="thinking trace",
    )

    assert message.to_model_dict() == {
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "thinking trace",
    }


def test_openai_compatible_message_preserves_provider_payload():
    model = OpenAIChat(id="deepseek-v4-flash", api_key="fake_openai_key")

    message = model.create_assistant_message(
        response_message=FakeDeepSeekMessage(),
        metrics=Metrics(),
        response_usage=None,
    )

    assert message.reasoning_content == "thinking trace"
    assert message.provider_data == {
        "role": "assistant",
        "content": "final answer",
        "reasoning_content": "thinking trace",
        "provider_extra_field": "future-compatible value",
    }


def test_deepseek_thinking_omits_unsupported_sampling_params():
    # Thinking is opt-in: user must pass reasoning_effort + extra_body.thinking explicitly.
    model = DeepSeekChat(
        temperature=0.7,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    request_kwargs = model.request_kwargs

    assert request_kwargs["reasoning_effort"] == "high"
    assert request_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "temperature" not in request_kwargs
    assert "top_p" not in request_kwargs
    assert "presence_penalty" not in request_kwargs
    assert "frequency_penalty" not in request_kwargs


def test_deepseek_no_default_thinking_params():
    # Plain DeepSeekChat() must NOT inject any thinking defaults, so user
    # `extra_body={"thinking": {"type": "disabled"}}` won't collide with a baked-in
    # `reasoning_effort`.
    model = DeepSeekChat()
    request_kwargs = model.request_kwargs

    assert "reasoning_effort" not in request_kwargs
    assert "extra_body" not in request_kwargs


def test_deepseek_non_thinking_keeps_sampling_params():
    model = DeepSeekChat(
        temperature=0.7,
        top_p=0.9,
        extra_body={"thinking": {"type": "disabled"}},
    )

    request_kwargs = model.request_kwargs

    assert request_kwargs["temperature"] == 0.7
    assert request_kwargs["top_p"] == 0.9


async def _collect_stream_reasoning(model):
    chunks = []
    async for chunk in model.response_stream([Message(role="user", content="hello")]):
        if chunk.reasoning_content:
            chunks.append(chunk.reasoning_content)
    return chunks


def test_nvidia_streaming_reasoning_field_maps_to_reasoning_content():
    model = NvidiaChat(api_key="fake_key")

    async def fake_invoke_stream(messages):
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        reasoning="nvidia reasoning",
                        content=None,
                        audio=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        reasoning=None,
                        content="final",
                        audio=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        )

    model.invoke_stream = fake_invoke_stream

    assert asyncio.run(_collect_stream_reasoning(model)) == ["nvidia reasoning"]
