"""Tests for gateway media understanding (image/voice/video routing).

Covers:
- base Gemini / catalog vision vs text-only
- settings.media_model resolve (no profile scan)
- prepare(): base-model attach vs media-model describe vs unconfigured note
- voice silk decoding plumbing
"""
import base64
import sys
from types import SimpleNamespace

import pytest

# Collection must not import agentica.gateway until extras are present —
# gateway/__init__.py raises ImportError without fastapi, which aborts CI.
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")

from agentica.gateway.channels.base import InboundMedia
from agentica.gateway.services import media_understanding as mu
from agentica.model.message import Message
from agentica.utils.tokens import count_message_tokens


# ---------------------------------------------------------------- helpers
class _FakeModel:
    """Stands in for a Model: captures messages and returns canned content."""

    def __init__(self, content: str):
        self._content = content
        self.calls: list = []

    async def response(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self._content)


def _svc(create_model_fn=None) -> mu.MediaUnderstandingService:
    return mu.MediaUnderstandingService(create_model_fn=create_model_fn or (lambda *a, **kw: _FakeModel("ok")))


def _jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 32


def _wav_bytes() -> bytes:
    return b"RIFF" + b"\x00" * 40 + b"WAVE"


def _media_model(**overrides):
    spec = {
        "model_provider": "openai",
        "model_name": "gemini-3.6-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key": "k",
    }
    spec.update(overrides)
    return spec


def _patch_media_model(monkeypatch, spec):
    monkeypatch.setattr(mu, "get_setting", lambda key, default=None: spec if key == "media_model" else default)


# ------------------------------------------------------------ detection
class TestBaseSupportsModality:
    def test_catalog_gemini_all_modalities(self):
        assert mu.base_supports_modality("gemini-2.5-flash", "image")
        assert mu.base_supports_modality("gemini-2.5-flash", "audio")
        assert mu.base_supports_modality("gemini-2.5-flash", "video")

    def test_catalog_text_only_model(self):
        assert not mu.base_supports_modality("deepseek-v4-flash", "image")
        assert not mu.base_supports_modality("deepseek-v4-flash", "audio")
        assert not mu.base_supports_modality("deepseek-v4-flash", "video")

    def test_catalog_vision_model(self):
        assert mu.base_supports_modality("gpt-4o", "image")
        assert not mu.base_supports_modality("gpt-4o", "audio")

    def test_uncatalogued_gemini_by_name(self):
        assert mu.is_gemini("gemini-3.6-flash")
        assert mu.base_supports_modality("gemini-3.6-flash", "video")
        assert mu.base_supports_modality("gemini-3.6-flash", "audio")

    def test_no_vl_or_4o_name_heuristics(self):
        assert not mu.base_supports_modality("acme-vision-x1", "image")
        assert not mu.base_supports_modality("acme-qwen2-vl", "image")
        assert not mu.base_supports_modality("acme-plain-1", "image")


# ------------------------------------------------------------ media_model resolve
class TestResolveMediaModel:
    def test_full_block(self, monkeypatch):
        _patch_media_model(monkeypatch, _media_model())
        spec = mu.resolve_media_model()
        assert spec["model_name"] == "gemini-3.6-flash"
        assert spec["api_key"] == "k"
        assert spec["model_provider"] == "openai"

    def test_model_name_defaults_when_omitted(self, monkeypatch):
        _patch_media_model(monkeypatch, {
            "model_provider": "openai",
            "base_url": "https://example.com/v1",
            "api_key": "k",
        })
        spec = mu.resolve_media_model()
        assert spec["model_name"] == mu.DEFAULT_MEDIA_MODEL_NAME

    def test_missing_setting_returns_none(self, monkeypatch):
        _patch_media_model(monkeypatch, None)
        assert mu.resolve_media_model() is None

    def test_name_alone_is_not_an_endpoint(self, monkeypatch):
        _patch_media_model(monkeypatch, {"model_name": "gemini-3.6-flash"})
        assert mu.resolve_media_model() is None


# ------------------------------------------------------------ prepare: image
class TestPrepareImage:
    @pytest.mark.asyncio
    async def test_base_model_capable_attaches_directly(self):
        svc = _svc()
        plan = await svc.prepare(
            [InboundMedia(kind="image", data=_jpeg_bytes())],
            base_model_id="gpt-4o",
            base_supports_images=True,
        )
        assert len(plan.images) == 1
        assert plan.images[0]["url"].startswith("data:image/jpeg;base64,")
        assert plan.text_parts == []
        assert plan.notes == []
        assert count_message_tokens(
            Message(role="user", content="请看这条图片。", images=plan.images)
        ) > 0

    @pytest.mark.asyncio
    async def test_uncatalogued_gemini_base_attaches_image(self):
        svc = _svc()
        plan = await svc.prepare(
            [InboundMedia(kind="image", data=_jpeg_bytes())],
            base_model_id="gemini-3.6-flash",
            base_supports_images=False,
        )
        assert len(plan.images) == 1
        assert plan.text_parts == []

    @pytest.mark.asyncio
    async def test_media_model_describes_and_notifies(self, monkeypatch):
        fake = _FakeModel("一只猫")
        _patch_media_model(monkeypatch, _media_model())
        svc = _svc(create_model_fn=lambda *a, **kw: fake)
        plan = await svc.prepare(
            [InboundMedia(kind="image", data=_jpeg_bytes())],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.images == []
        assert plan.text_parts == ["[图片内容]\n一只猫"]
        assert len(plan.notes) == 1
        assert "gemini-3.6-flash" in plan.notes[0]
        sent = fake.calls[0][0]
        blocks = sent.content
        assert any(b.get("type") == "image_url" for b in blocks)

    @pytest.mark.asyncio
    async def test_unconfigured_warns_and_notifies(self, monkeypatch):
        import logging

        from agentica.utils.log import logger

        _patch_media_model(monkeypatch, None)
        warnings: list = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                warnings.append(record.getMessage())

        handler = _ListHandler(level=logging.WARNING)
        logger.addHandler(handler)
        try:
            svc = _svc()
            plan = await svc.prepare(
                [InboundMedia(kind="image", data=_jpeg_bytes())],
                base_model_id="deepseek-v4-flash",
                base_supports_images=False,
            )
        finally:
            logger.removeHandler(handler)
        assert plan.images == []
        assert plan.text_parts == []
        assert any("暂不支持" in n and "图片" in n and "media_model" in n for n in plan.notes)
        assert any("media_model" in w for w in warnings)


# ------------------------------------------------------------ prepare: voice
class TestPrepareVoice:
    @pytest.mark.asyncio
    async def test_base_audio_capable_attaches_wav(self):
        svc = _svc()
        plan = await svc.prepare(
            [InboundMedia(kind="voice", data=_wav_bytes())],
            base_model_id="gemini-2.5-flash",
            base_supports_images=True,
        )
        assert plan.audio is not None
        assert plan.audio["format"] == "wav"
        assert base64.b64decode(plan.audio["data"]) == _wav_bytes()
        assert plan.notes == []

    @pytest.mark.asyncio
    async def test_media_model_transcribes_and_notifies(self, monkeypatch):
        fake = _FakeModel("你好，世界")
        _patch_media_model(monkeypatch, _media_model())
        svc = _svc(create_model_fn=lambda *a, **kw: fake)
        plan = await svc.prepare(
            [InboundMedia(kind="voice", data=_wav_bytes())],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.audio is None
        assert plan.text_parts == ["[语音转写]\n你好，世界"]
        assert any("转写" in n and "gemini-3.6-flash" in n for n in plan.notes)
        blocks = fake.calls[0][0].content
        assert any(b.get("type") == "audio_url" for b in blocks)

    @pytest.mark.asyncio
    async def test_silk_without_decoder_notifies(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pilk", None)
        silk = b"#!SILK_V3" + b"\x01" * 20
        svc = _svc()
        plan = await svc.prepare(
            [InboundMedia(kind="voice", data=silk)],
            base_model_id="gemini-2.5-flash",
            base_supports_images=True,
        )
        assert plan.audio is None
        assert any("解码" in n for n in plan.notes)

    def test_silk_decoded_to_wav_with_pilk(self, monkeypatch):
        pcm = b"\x00\x01" * 100

        def fake_decode(silk_path, pcm_path, pcm_rate=24000):
            with open(pcm_path, "wb") as f:
                f.write(pcm)

        monkeypatch.setitem(sys.modules, "pilk", SimpleNamespace(decode=fake_decode))
        out, fmt = mu.decode_voice(b"\x02#!SILK_V3" + b"\x01" * 20)
        assert fmt == "wav"
        assert out.startswith(b"RIFF")
        assert pcm in out

    def test_wav_passthrough(self):
        out, fmt = mu.decode_voice(_wav_bytes())
        assert fmt == "wav"
        assert out == _wav_bytes()


# ------------------------------------------------------------ prepare: video
class TestPrepareVideo:
    @pytest.mark.asyncio
    async def test_base_gemini_gets_inline_mp4(self):
        svc = _svc()
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        plan = await svc.prepare(
            [InboundMedia(kind="video", data=mp4)],
            base_model_id="gemini-2.5-flash",
            base_supports_images=True,
        )
        assert len(plan.images) == 1
        assert plan.images[0]["url"] == "data:video/mp4;base64," + base64.b64encode(mp4).decode()
        assert plan.notes == []
        assert count_message_tokens(
            Message(role="user", content="请看这条视频。", images=plan.images)
        ) > 0

    @pytest.mark.asyncio
    async def test_media_model_describes_video(self, monkeypatch):
        fake = _FakeModel("一段日落视频")
        _patch_media_model(monkeypatch, _media_model())
        svc = _svc(create_model_fn=lambda *a, **kw: fake)
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        plan = await svc.prepare(
            [InboundMedia(kind="video", data=mp4)],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.text_parts == ["[视频内容]\n一段日落视频"]
        assert any("视频" in n and "gemini-3.6-flash" in n for n in plan.notes)

    @pytest.mark.asyncio
    async def test_unconfigured_video_passes_with_note(self, monkeypatch):
        _patch_media_model(monkeypatch, None)
        svc = _svc()
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        plan = await svc.prepare(
            [InboundMedia(kind="video", data=mp4)],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.images == []
        assert plan.text_parts == []
        assert any("暂不支持" in n and "视频" in n and "media_model" in n for n in plan.notes)

    @pytest.mark.asyncio
    async def test_oversize_video_skipped(self):
        svc = _svc()
        big = b"\x00" * (mu.MAX_INLINE_VIDEO_BYTES + 1)
        plan = await svc.prepare(
            [InboundMedia(kind="video", data=big)],
            base_model_id="gemini-2.5-flash",
            base_supports_images=True,
        )
        assert plan.images == []
        assert any("过大" in n for n in plan.notes)
