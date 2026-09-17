# -*- coding: utf-8 -*-
"""Built-in ``analyze_image``: capability routing and how pixels reach the model.

The routing decisions are what these tests pin down, because each wrong branch
fails in its own expensive way: handing an image to a text-only model is a hard
API error, while asking a second model to describe a picture the agent could
have read itself silently loses detail and costs an extra call.
"""
import asyncio
import base64
import struct
from types import SimpleNamespace

import pytest

from agentica.media import (
    is_multimodal_tool_result,
    multimodal_text_summary,
    multimodal_tool_result,
)
from agentica.tools.builtin.vision_tool import BuiltinVisionTool, _data_url


def _png(width: int = 8, height: int = 8) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">LL", width, height)


def _jpeg() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _model(*, supports_images: bool):
    return SimpleNamespace(supports_images=supports_images, id="test-model", provider="openai")


class TestRouting:
    def test_vision_capable_agent_gets_the_pixels_itself(self, tmp_path):
        """No second model, no OCR: the envelope carries the image to the agent."""
        png = tmp_path / "shot.png"
        png.write_bytes(_png())
        tool = BuiltinVisionTool(vision_model=_unusable_vision_model())
        tool.set_agent_model(_model(supports_images=True))

        result = asyncio.run(tool.analyze_image(str(png), "什么字符串？"))

        assert is_multimodal_tool_result(result)
        assert result["images"][0]["url"].startswith("data:image/png;base64,")
        # The question travels with the image so the model knows what to answer.
        assert "什么字符串？" in result["text"]

    def test_text_only_agent_falls_back_to_the_vision_model(self, tmp_path):
        png = tmp_path / "shot.png"
        png.write_bytes(_png())
        asked = {}

        class _Vision:
            async def response(self, messages):
                asked["content"] = messages[0].content
                asked["images"] = list(messages[0].images)
                return SimpleNamespace(content="一张写着 ZX-41827 的图")

        tool = BuiltinVisionTool(vision_model=_Vision())
        tool.set_agent_model(_model(supports_images=False))

        result = asyncio.run(tool.analyze_image(str(png), "念出来"))

        assert result == "一张写着 ZX-41827 的图"
        assert asked["content"] == "念出来"
        assert asked["images"][0]["url"].startswith("data:image/png;base64,")

    def test_no_vision_anywhere_falls_back_to_ocr(self, tmp_path, monkeypatch):
        png = tmp_path / "shot.png"
        png.write_bytes(_png())
        monkeypatch.setattr(
            "agentica.tools.builtin.vision_tool.ocr_image_text",
            lambda path, max_chars=None: "ZX-41827",
        )
        tool = BuiltinVisionTool()
        tool.set_agent_model(_model(supports_images=False))

        result = asyncio.run(tool.analyze_image(str(png)))

        assert "ZX-41827" in result
        # The model must not mistake extracted text for a description.
        assert "OCR" in result

    def test_all_three_unavailable_raises_with_a_way_forward(self, tmp_path, monkeypatch):
        png = tmp_path / "shot.png"
        png.write_bytes(_png())
        monkeypatch.setattr(
            "agentica.tools.builtin.vision_tool.ocr_image_text",
            lambda path, max_chars=None: "",
        )
        tool = BuiltinVisionTool()
        tool.set_agent_model(_model(supports_images=False))

        with pytest.raises(RuntimeError) as exc:
            asyncio.run(tool.analyze_image(str(png)))

        assert "imgocr" in str(exc.value)

    def test_vision_model_failure_degrades_to_ocr(self, tmp_path, monkeypatch):
        """A broken vision endpoint must not lose the OCR tier behind it."""
        png = tmp_path / "shot.png"
        png.write_bytes(_png())

        class _Broken:
            async def response(self, messages):
                raise RuntimeError("502 bad gateway")

        monkeypatch.setattr(
            "agentica.tools.builtin.vision_tool.ocr_image_text",
            lambda path, max_chars=None: "fallback text",
        )
        tool = BuiltinVisionTool(vision_model=_Broken())
        tool.set_agent_model(_model(supports_images=False))

        assert "fallback text" in asyncio.run(tool.analyze_image(str(png)))


class TestInputHandling:
    def test_http_url_is_passed_through_without_reading_disk(self):
        tool = BuiltinVisionTool()
        tool.set_agent_model(_model(supports_images=True))

        result = asyncio.run(tool.analyze_image("https://example.com/a.png"))

        assert result["images"][0]["url"] == "https://example.com/a.png"

    def test_missing_file_is_reported_as_missing(self, tmp_path):
        tool = BuiltinVisionTool()
        tool.set_agent_model(_model(supports_images=True))

        with pytest.raises(FileNotFoundError):
            asyncio.run(tool.analyze_image(str(tmp_path / "nope.png")))

    def test_non_image_file_is_rejected_before_any_model_call(self, tmp_path):
        """A text file named .png must not be base64'd into a vision request."""
        fake = tmp_path / "notreally.png"
        fake.write_text("just text\n")
        tool = BuiltinVisionTool()
        tool.set_agent_model(_model(supports_images=True))

        with pytest.raises(ValueError):
            asyncio.run(tool.analyze_image(str(fake)))

    def test_mime_comes_from_magic_bytes_not_the_filename(self, tmp_path):
        """A JPEG named .png must be declared image/jpeg, or providers reject it."""
        mislabelled = tmp_path / "actually_jpeg.png"
        mislabelled.write_bytes(_jpeg())
        tool = BuiltinVisionTool()
        tool.set_agent_model(_model(supports_images=True))

        result = asyncio.run(tool.analyze_image(str(mislabelled)))

        assert result["images"][0]["url"].startswith("data:image/jpeg;base64,")

    def test_relative_path_resolves_against_work_dir(self, tmp_path):
        png = tmp_path / "shot.png"
        png.write_bytes(_png())
        tool = BuiltinVisionTool(work_dir=str(tmp_path))
        tool.set_agent_model(_model(supports_images=True))

        result = asyncio.run(tool.analyze_image("shot.png"))

        assert is_multimodal_tool_result(result)

    def test_data_url_types_each_format(self):
        assert _data_url(_png()).startswith("data:image/png;base64,")
        assert _data_url(_jpeg()).startswith("data:image/jpeg;base64,")


class TestEnvelope:
    def test_text_summary_is_what_a_blind_model_sees(self):
        envelope = multimodal_tool_result(
            text="look at this", images=[{"url": "x"}], text_summary="[image attached]"
        )

        assert multimodal_text_summary(envelope) == "[image attached]"

    def test_plain_strings_pass_through_unchanged(self):
        assert multimodal_text_summary("plain result") == "plain result"
        assert not is_multimodal_tool_result("plain result")

    def test_a_dict_without_the_marker_is_not_an_envelope(self):
        """Ordinary dict tool results must keep being stringified as before."""
        assert not is_multimodal_tool_result({"images": [], "text": "x"})


def _unusable_vision_model():
    """A vision model that fails the test if the native path is skipped."""

    class _Boom:
        async def response(self, messages):
            raise AssertionError("vision model called although the agent can see")

    return _Boom()
