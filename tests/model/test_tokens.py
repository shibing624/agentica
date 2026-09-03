import base64
import struct
from types import SimpleNamespace

from agentica.media import Image
from agentica.model.message import Message
from agentica.tools.base import Function
from agentica.utils.tokens import (
    count_image_tokens,
    count_message_tokens,
    count_tool_tokens,
    count_video_tokens,
)


def _png_header(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">LL", width, height)
    )


def test_count_image_tokens_remote_url_does_not_fetch_network(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(content=_png_header(64, 64))

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    tokens = count_image_tokens(Image(url="https://example.com/image.png"))

    assert calls == []
    assert tokens > 0


def test_count_image_tokens_accepts_url_dict():
    """Gateway attaches {"url": data_url} dicts; OpenAI process_image already accepts them."""
    tokens = count_image_tokens({"url": "https://example.com/image.png"})
    assert tokens > 0


def test_count_image_tokens_accepts_data_url_dict():
    url = "data:image/png;base64," + base64.b64encode(_png_header(64, 64)).decode()
    tokens = count_image_tokens({"url": url})
    assert tokens > 0


def test_count_message_tokens_accepts_gateway_image_dicts():
    url = "data:image/png;base64," + base64.b64encode(_png_header(64, 64)).decode()
    message = Message(
        role="user",
        content="请看这条图片。",
        images=[{"url": url}],
    )
    assert count_message_tokens(message) > 0


def test_count_message_tokens_remote_image_url_content_does_not_fetch_network(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(content=_png_header(64, 64))

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    message = Message(
        role="user",
        content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image.png",
                    "detail": "auto",
                },
            }
        ],
    )

    tokens = count_message_tokens(message)

    assert calls == []
    assert tokens > 0


def test_count_message_tokens_ignores_reasoning_content():
    """reasoning_content is response-only: providers ignore it on input.

    Taiji/hy4 (and DeepSeek-style APIs generally) accept the field on input
    messages but do not count it (verified against the live endpoint: a
    215KB reasoning_content changed prompt_tokens by 0). Estimating the
    request with it included inflates context occupancy and fires Layer-1
    eviction early — observed as a status bar reading 144% while the
    provider saw ~80%.
    """
    reasoning = "let me think about this step by step. " * 2000
    with_reasoning = Message(role="assistant", content="answer", reasoning_content=reasoning)
    without = Message(role="assistant", content="answer")

    assert count_message_tokens(with_reasoning, "hy4-preview") == count_message_tokens(
        without, "hy4-preview"
    )


def test_count_tool_tokens_accepts_function_objects_and_dicts():
    function = Function(
        name="read_file",
        description="Read a file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
    )

    tokens = count_tool_tokens(
        [
            function,
            {
                "type": "function",
                "function": {
                    "name": "glob",
                    "description": "Search files",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
    )

    assert tokens > 0


def test_count_video_tokens_uses_same_visual_formula_as_images():
    tokens = count_video_tokens(
        SimpleNamespace(width=1024, height=1024, duration=1, fps=1)
    )

    assert tokens == 765
