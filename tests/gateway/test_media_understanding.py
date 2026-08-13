"""Tests for gateway media understanding (image/voice/video routing).

Covers:
- capability detection (catalog -> name hints -> explicit declaration)
- fallback model discovery across config.yaml profiles
- prepare(): base-model attach vs fallback description vs pass-through
- voice silk decoding plumbing
"""
import base64
import sys
from types import SimpleNamespace

import pytest

from agentica.gateway.channels.base import InboundMedia
from agentica.gateway.services import media_understanding as mu


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


# ------------------------------------------------------------ detection
class TestSupportsModality:
    def test_catalog_gemini_all_modalities(self):
        assert mu.supports_modality("gemini-2.5-flash", "image")
        assert mu.supports_modality("gemini-2.5-flash", "audio")
        assert mu.supports_modality("gemini-2.5-flash", "video")

    def test_catalog_text_only_model(self):
        assert not mu.supports_modality("deepseek-v4-flash", "image")
        assert not mu.supports_modality("deepseek-v4-flash", "audio")
        assert not mu.supports_modality("deepseek-v4-flash", "video")

    def test_catalog_vision_model(self):
        assert mu.supports_modality("gpt-4o", "image")
        assert not mu.supports_modality("gpt-4o", "audio")

    def test_declared_overrides_catalog(self):
        assert mu.supports_modality("deepseek-v4-flash", "image", declared=["image"])
        assert not mu.supports_modality("deepseek-v4-flash", "audio", declared=["image"])

    def test_name_hints_when_catalog_misses(self):
        assert mu.supports_modality("acme-gemini-pro", "video")
        assert mu.supports_modality("acme-gemini-pro", "audio")
        assert mu.supports_modality("acme-vision-x1", "image")
        assert mu.supports_modality("acme-qwen2-vl", "image")
        assert not mu.supports_modality("acme-plain-1", "image")

    def test_gemini_hint_only_as_fallback(self):
        # a catalog-known text model whose name happens to contain no hint stays incapable
        assert not mu.supports_modality("glm-4-flash", "video")


# ------------------------------------------------------------ profile scan
class TestFindMediaModel:
    def test_finds_first_supporting_profile(self, monkeypatch):
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "a": {"model_provider": "deepseek", "model_name": "deepseek-v4-flash"},
            "b": {"model_provider": "openai", "model_name": "gpt-4o", "api_key": "k"},
        })
        svc = _svc()
        spec = svc.find_model_for("image")
        assert spec["profile"] == "b"
        assert spec["model_name"] == "gpt-4o"
        assert spec["api_key"] == "k"

    def test_returns_none_when_nobody_supports(self, monkeypatch):
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "a": {"model_provider": "deepseek", "model_name": "deepseek-v4-flash"},
        })
        assert _svc().find_model_for("video") is None

    def test_scans_auxiliary_model_block(self, monkeypatch):
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "a": {
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "auxiliary_model": {
                    "model_provider": "openai",
                    "model_name": "gemini-2.5-flash",
                    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                    "api_key": "gk",
                },
            },
        })
        spec = _svc().find_model_for("audio")
        assert spec is not None
        assert spec["model_name"] == "gemini-2.5-flash"
        assert spec["role"] == "auxiliary"

    def test_scans_fallback_models_with_declared_modalities(self, monkeypatch):
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "a": {
                "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash",
                "fallback_models": [{
                    "model_provider": "openai",
                    "model_name": "my-private-vl",
                    "base_url": "http://internal/v1",
                    "api_key": "x",
                    "modalities": ["image"],
                }],
            },
        })
        spec = _svc().find_model_for("image")
        assert spec is not None
        assert spec["model_name"] == "my-private-vl"
        assert spec["role"] == "fallback"


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

    @pytest.mark.asyncio
    async def test_fallback_model_describes_and_notifies(self, monkeypatch):
        fake = _FakeModel("一只猫")
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "v": {"model_provider": "openai", "model_name": "gpt-4o", "api_key": "k"},
        })
        svc = _svc(create_model_fn=lambda *a, **kw: fake)
        plan = await svc.prepare(
            [InboundMedia(kind="image", data=_jpeg_bytes())],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.images == []
        assert plan.text_parts == ["[图片内容]\n一只猫"]
        assert len(plan.notes) == 1
        assert "gpt-4o" in plan.notes[0]
        # the fallback model really received an image block
        sent = fake.calls[0][0]
        blocks = sent.content
        assert any(b.get("type") == "image_url" for b in blocks)

    @pytest.mark.asyncio
    async def test_no_capable_model_warns_and_notifies(self, monkeypatch):
        import logging

        from agentica.utils.log import logger

        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "a": {"model_provider": "deepseek", "model_name": "deepseek-v4-flash"},
        })
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
        assert any("暂不支持" in n and "图片" in n for n in plan.notes)
        assert any("image" in w for w in warnings)


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
    async def test_fallback_transcribes_and_notifies(self, monkeypatch):
        fake = _FakeModel("你好，世界")
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "g": {"model_provider": "openai", "model_name": "gemini-2.5-flash", "api_key": "k"},
        })
        svc = _svc(create_model_fn=lambda *a, **kw: fake)
        plan = await svc.prepare(
            [InboundMedia(kind="voice", data=_wav_bytes())],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.audio is None
        assert plan.text_parts == ["[语音转写]\n你好，世界"]
        assert any("转写" in n and "gemini-2.5-flash" in n for n in plan.notes)

    @pytest.mark.asyncio
    async def test_silk_without_decoder_notifies(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pilk", None)
        monkeypatch.setitem(sys.modules, "graiax", None)
        monkeypatch.setitem(sys.modules, "graiax.silkcoder", None)
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
        fake_pilk = SimpleNamespace(decode=lambda data: pcm)
        monkeypatch.setitem(sys.modules, "pilk", fake_pilk)
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

    @pytest.mark.asyncio
    async def test_fallback_video_model_describes(self, monkeypatch):
        fake = _FakeModel("一段日落视频")
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "g": {"model_provider": "openai", "model_name": "gemini-2.5-flash", "api_key": "k"},
        })
        svc = _svc(create_model_fn=lambda *a, **kw: fake)
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        plan = await svc.prepare(
            [InboundMedia(kind="video", data=mp4)],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.text_parts == ["[视频内容]\n一段日落视频"]
        assert any("视频" in n and "gemini-2.5-flash" in n for n in plan.notes)

    @pytest.mark.asyncio
    async def test_no_video_model_passes_with_note(self, monkeypatch):
        monkeypatch.setattr(mu, "get_profiles", lambda: {
            "a": {"model_provider": "deepseek", "model_name": "deepseek-v4-flash"},
        })
        svc = _svc()
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64
        plan = await svc.prepare(
            [InboundMedia(kind="video", data=mp4)],
            base_model_id="deepseek-v4-flash",
            base_supports_images=False,
        )
        assert plan.images == []
        assert plan.text_parts == []
        assert any("暂不支持" in n and "视频" in n for n in plan.notes)

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
