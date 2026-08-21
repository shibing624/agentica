# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Inbound media understanding for gateway channels.

Two rules, no profile scan:

* **Images** — if the base model can see (``supports_images``, or the base
  id is Gemini), attach the payload to ``agent.run(images=)``. Otherwise a
  one-shot describe via ``settings.media_model``.
* **Audio / video** — if the base id is Gemini (natively multimodal), attach
  (video rides ``images`` as a ``data:video/mp4;base64`` URL dict, the
  convention Gemini's OpenAI-compatible endpoint understands). Otherwise the
  same ``settings.media_model`` transcribes / describes into the user text.

``settings.media_model`` is a model block (``model_provider`` / ``model_name``
/ ``base_url`` / ``api_key``). ``model_name`` defaults to
``gemini`` when omitted; provider or ``base_url`` must be set so
this does not guess an endpoint. Missing config yields a user-facing note,
not a hunt through every profile.

Voice payloads from WeChat arrive silk-encoded; they are decoded to wav
first via ``pilk`` (installed with the ``wechat`` extra).
Videos larger than ``MAX_INLINE_VIDEO_BYTES`` are skipped (Gemini inline
payloads cap at ~20MB).
"""
import base64
import io
import os
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentica.cost_tracker import get_model_supports_modality
from agentica.global_config import get_setting
from agentica.model.message import Message
from agentica.utils.log import logger

from ..channels.base import InboundMedia

# Gemini inline payloads cap at ~20MB; stay well under it.
MAX_INLINE_VIDEO_BYTES = 15 * 1024 * 1024

# Default name inside settings.media_model when the block omits model_name.
DEFAULT_MEDIA_MODEL_NAME = "gemini-3.6-flash"

_MODALITY_LABELS = {"image": "图片", "audio": "语音", "video": "视频"}
_KIND_TO_MODALITY = {"image": "image", "voice": "audio", "video": "video"}

# Silk (WeChat voice) decodes to 24kHz mono s16 PCM.
_SILK_PCM_RATE = 24000

_IMAGE_PROMPT = "请详细描述这张图片的内容，包括画面中的文字、物体和场景，用中文简洁回答。"
_VOICE_PROMPT = "请将这段语音完整转写为文字。只输出转写内容，不要解释、不要翻译。"
_VIDEO_PROMPT = "请描述这段视频的内容，包括画面、人物、动作和语音要点，用中文简洁回答。"


@dataclass
class MediaPlan:
    """Routing decision for one inbound message's media payloads.

    ``images`` holds everything that goes to the base model via
    ``agent.run(images=...)``: real images *and* videos (as
    ``data:video/mp4;base64`` URL dicts — see module docstring).
    ``text_parts`` are media-model descriptions/transcriptions to append
    to the user text; ``notes`` are user-facing one-liners about non-base
    models used or media that had to be skipped.
    """
    images: List[Any] = field(default_factory=list)
    audio: Optional[dict] = None
    text_parts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def is_gemini(model_id: str) -> bool:
    """Uncatalogued Gemini deploys (e.g. gemini-3.6-flash) are natively multimodal."""
    return "gemini" in (model_id or "").lower()


def base_supports_modality(model_id: str, modality: str) -> bool:
    """Whether the base model can take ``modality`` as native input.

    Catalog first; Gemini by name is the only extra (no vl/4o/seed heuristics).
    """
    if get_model_supports_modality(model_id, modality):
        return True
    return is_gemini(model_id)


def resolve_media_model() -> Optional[Dict[str, Any]]:
    """The model used for describe/transcribe, or None if unconfigured.

    Reads ``settings.media_model``. ``model_name`` defaults to
    ``DEFAULT_MEDIA_MODEL_NAME``; ``model_provider`` or ``base_url`` must be
    present so we do not invent an endpoint.
    """
    raw = get_setting("media_model")
    if not isinstance(raw, dict):
        return None
    name = (raw.get("model_name") or "").strip() or DEFAULT_MEDIA_MODEL_NAME
    provider = (raw.get("model_provider") or "").strip()
    base_url = (raw.get("base_url") or "").strip()
    api_key = (raw.get("api_key") or "").strip()
    wire_api = (raw.get("wire_api") or "").strip()
    if not provider and not base_url:
        return None
    return {
        "model_provider": provider or "openai",
        "model_name": name,
        "base_url": base_url,
        "api_key": api_key,
        "wire_api": wire_api,
    }


def media_model_label() -> Optional[str]:
    """Configured ``settings.media_model`` name, or None if the block is missing."""
    spec = resolve_media_model()
    if spec is None:
        return None
    return spec["model_name"]


def sniff_image_mime(data: bytes) -> str:
    """Best-effort image MIME sniffing from magic bytes (defaults to jpeg)."""
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _looks_like_silk(data: bytes) -> bool:
    return data.startswith(b"#!SILK") or data.startswith(b"\x02#!SILK")


def _looks_like_mp3(data: bytes) -> bool:
    return data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)


def _pcm_to_wav(pcm: bytes, rate: int = _SILK_PCM_RATE) -> bytes:
    """Wrap mono s16 PCM bytes in a wav container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _silk_to_pcm(data: bytes) -> Optional[bytes]:
    """Decode silk bytes to raw PCM using ``pilk``.

    ``pilk`` (0.2.x) exposes a file-path based API (``decode(silk_path,
    pcm_path, pcm_rate=...)``), not a bytes-in/bytes-out one, so the payload is
    staged through temp files. Returns None when ``pilk`` is not installed or
    decoding fails.
    """
    try:
        import pilk
    except ImportError:
        return None

    silk_path = tempfile.mktemp(suffix=".silk")
    pcm_path = tempfile.mktemp(suffix=".pcm")
    try:
        with open(silk_path, "wb") as f:
            f.write(data)
        pilk.decode(silk_path, pcm_path, pcm_rate=_SILK_PCM_RATE)
        with open(pcm_path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"pilk silk decode failed: {e}")
        return None
    finally:
        for p in (silk_path, pcm_path):
            try:
                os.remove(p)
            except OSError:
                pass


def decode_voice(data: bytes) -> Optional[Tuple[bytes, str]]:
    """Normalise a voice payload to (bytes, format) for audio input.

    wav/mp3 pass through untouched; silk is decoded and wrapped as wav.
    Returns None when the payload needs decoding but no decoder is installed.
    """
    if data.startswith(b"RIFF"):
        return data, "wav"
    if _looks_like_mp3(data):
        return data, "mp3"
    if not _looks_like_silk(data):
        return None
    pcm = _silk_to_pcm(data)
    if pcm is None:
        return None
    return _pcm_to_wav(pcm), "wav"


class MediaUnderstandingService:
    """Routes inbound media to the base model or ``settings.media_model``."""

    def __init__(self, create_model_fn: Optional[Callable] = None):
        if create_model_fn is None:
            from .model_factory import create_model
            create_model_fn = create_model
        self._create_model = create_model_fn
        self._model_cache: Dict[tuple, Any] = {}

    def _get_model(self, spec: Dict[str, Any]):
        key = (spec["model_provider"], spec["model_name"], spec["base_url"])
        model = self._model_cache.get(key)
        if model is None:
            model = self._create_model(
                spec["model_provider"],
                spec["model_name"],
                base_url=spec["base_url"] or None,
                api_key=spec["api_key"] or None,
                wire_api=spec["wire_api"],
            )
            self._model_cache[key] = model
        return model

    async def prepare(
        self,
        media: List[InboundMedia],
        *,
        base_model_id: str,
        base_supports_images: bool,
    ) -> MediaPlan:
        """Decide how each media item reaches the agent (see module docstring)."""
        plan = MediaPlan()
        for item in media:
            try:
                await self._route(item, plan, base_model_id, base_supports_images)
            except Exception as e:  # external I/O boundary: CDN/LLM calls
                label = _MODALITY_LABELS.get(_KIND_TO_MODALITY.get(item.kind, ""), item.kind)
                logger.warning(f"Media understanding failed for {item.kind}: {e}")
                plan.notes.append(f"⚠️ {label}处理失败：{e}")
        return plan

    async def _route(
        self,
        item: InboundMedia,
        plan: MediaPlan,
        base_model_id: str,
        base_supports_images: bool,
    ) -> None:
        if item.kind == "image":
            await self._route_image(item, plan, base_model_id, base_supports_images)
        elif item.kind == "voice":
            await self._route_voice(item, plan, base_model_id)
        elif item.kind == "video":
            await self._route_video(item, plan, base_model_id)
        else:
            logger.debug(f"Media understanding: skipping unsupported kind {item.kind!r}")

    def _note_unconfigured(self, plan: MediaPlan, modality: str) -> None:
        label = _MODALITY_LABELS[modality]
        logger.warning(
            f"Media understanding: no settings.media_model for {modality} "
            f"(base model cannot take it natively)"
        )
        plan.notes.append(
            f"⚠️ 暂不支持{label}理解：底模不能处理，且未配置 settings.media_model"
            f"（在 config.yaml 的 settings 中增加 media_model："
            f"需要 model_name，以及 model_provider 或 base_url）"
        )

    async def _describe(self, spec: Dict[str, Any], blocks: List[dict]) -> str:
        model = self._get_model(spec)
        resp = await model.response([Message(role="user", content=blocks)])
        return (resp.content or "").strip()

    async def _route_image(
        self,
        item: InboundMedia,
        plan: MediaPlan,
        base_model_id: str,
        base_supports_images: bool,
    ) -> None:
        mime = item.mime or sniff_image_mime(item.data)
        url = f"data:{mime};base64,{base64.b64encode(item.data).decode()}"
        if base_supports_images or is_gemini(base_model_id):
            plan.images.append({"url": url})
            return
        spec = resolve_media_model()
        if spec is None:
            self._note_unconfigured(plan, "image")
            return
        text = await self._describe(spec, [
            {"type": "text", "text": _IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ])
        plan.text_parts.append(f"[图片内容]\n{text}")
        plan.notes.append(f"🖼 图片由 {spec['model_name']} 识别（底模不支持读图）")

    async def _route_voice(self, item: InboundMedia, plan: MediaPlan, base_model_id: str) -> None:
        decoded = decode_voice(item.data)
        if decoded is None:
            logger.warning("Media understanding: voice decode failed (pilk installed?)")
            plan.notes.append(
                "⚠️ 语音解码失败：请安装 silk 解码库 pilk"
                "（pip install pilk 或 pip install 'agentica[wechat]'）"
            )
            return
        payload, fmt = decoded
        if plan.audio is None and base_supports_modality(base_model_id, "audio"):
            plan.audio = {"data": base64.b64encode(payload).decode(), "format": fmt}
            return
        spec = resolve_media_model()
        if spec is None:
            self._note_unconfigured(plan, "audio")
            return
        audio_b64 = base64.b64encode(payload).decode()
        # Gemini's OpenAI-compatible endpoint takes audio_url data URLs and
        # rejects OpenAI's native input_audio block.
        text = await self._describe(spec, [
            {"type": "text", "text": _VOICE_PROMPT},
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/{fmt};base64,{audio_b64}"},
            },
        ])
        plan.text_parts.append(f"[语音转写]\n{text}")
        plan.notes.append(f"🎤 语音由 {spec['model_name']} 转写（底模不支持语音）")

    async def _route_video(self, item: InboundMedia, plan: MediaPlan, base_model_id: str) -> None:
        if len(item.data) > MAX_INLINE_VIDEO_BYTES:
            logger.warning(f"Media understanding: video too large ({len(item.data)} bytes), skipped")
            plan.notes.append(f"⚠️ 视频过大（>{MAX_INLINE_VIDEO_BYTES // 1024 // 1024}MB），已跳过")
            return
        url = f"data:video/mp4;base64,{base64.b64encode(item.data).decode()}"
        if base_supports_modality(base_model_id, "video"):
            plan.images.append({"url": url})
            return
        spec = resolve_media_model()
        if spec is None:
            self._note_unconfigured(plan, "video")
            return
        text = await self._describe(spec, [
            {"type": "text", "text": _VIDEO_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ])
        plan.text_parts.append(f"[视频内容]\n{text}")
        plan.notes.append(f"🎬 视频由 {spec['model_name']} 解读（底模不支持视频）")


# Process-wide singleton — the media-model cache lives here.
media_understanding = MediaUnderstandingService()

__all__ = [
    "DEFAULT_MEDIA_MODEL_NAME",
    "MAX_INLINE_VIDEO_BYTES",
    "MediaPlan",
    "MediaUnderstandingService",
    "base_supports_modality",
    "decode_voice",
    "is_gemini",
    "media_model_label",
    "media_understanding",
    "resolve_media_model",
    "sniff_image_mime",
]
