import asyncio
import base64

import pytest

from agentica.model.anthropic import Claude
from agentica.model.litellm import LiteLLMChat
from agentica.model.message import Message
from agentica.model.ollama import Ollama
from agentica.model.openai import OpenAIChat


def test_model_infers_image_support_from_catalog():
    vision_model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
    text_model = OpenAIChat(id="gpt-4", api_key="fake_openai_key")

    assert vision_model.supports_images is True
    assert text_model.supports_images is False


def test_explicit_image_support_overrides_catalog():
    disabled = OpenAIChat(
        id="gpt-4o",
        api_key="fake_openai_key",
        supports_images=False,
    )
    custom = OpenAIChat(
        id="my-company-vision-model",
        api_key="fake_openai_key",
        supports_images=True,
    )

    assert disabled.supports_images is False
    assert custom.supports_images is True


def test_text_model_rejects_image_input_before_provider_call():
    model = OpenAIChat(id="gpt-4", api_key="fake_openai_key")
    message = Message(role="user", content="describe this image")

    with pytest.raises(ValueError, match="does not support image input"):
        model.add_images_to_message(message, [b"image-bytes"])


def test_vision_model_converts_image_bytes_to_multimodal_content():
    model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")
    message = Message(role="user", content="describe this image")

    formatted = model.add_images_to_message(message, [b"image-bytes"])

    assert formatted.content[0] == {"type": "text", "text": "describe this image"}
    assert formatted.content[1]["type"] == "image_url"
    assert formatted.content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_litellm_forwards_allowed_images_as_multimodal_content():
    model = object.__new__(LiteLLMChat)
    model.id = "private-vision-model"
    model.supports_images = True
    message = Message(
        role="user",
        content="describe this image",
        images=[b"image-bytes"],
    )

    formatted = model.format_message(message)

    assert formatted["content"][0] == {"type": "text", "text": "describe this image"}
    assert formatted["content"][1]["type"] == "image_url"


def test_anthropic_rejects_images_for_text_only_model():
    model = object.__new__(Claude)
    model.id = "private-text-model"
    model.supports_images = False
    model.enable_cache_control = False
    message = Message(
        role="user",
        content="describe this image",
        images=[b"image-bytes"],
    )

    with pytest.raises(ValueError, match="does not support image input"):
        asyncio.run(model.format_messages([message]))


def test_anthropic_converts_image_bytes_to_native_block():
    model = object.__new__(Claude)
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )

    image_block = asyncio.run(model.add_image(png_bytes))

    assert image_block["type"] == "image"
    assert image_block["source"]["media_type"] == "image/png"
    assert image_block["source"]["data"] == base64.b64encode(png_bytes).decode("utf-8")


def test_ollama_rejects_images_for_text_only_model():
    model = object.__new__(Ollama)
    model.id = "private-text-model"
    model.supports_images = False
    message = Message(
        role="user",
        content="describe this image",
        images=[b"image-bytes"],
    )

    with pytest.raises(ValueError, match="does not support image input"):
        model.format_message(message)
