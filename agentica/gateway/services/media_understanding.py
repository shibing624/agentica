# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Inbound media understanding for gateway channels.

Decides, per media item (image / voice / video), how it reaches the agent:

* **Base model handles the modality** → attach the payload to the agent run
  (``images=`` / ``audio=``). Videos ride the ``images`` channel as a
  ``data:video/mp4;base64,<...>`` URL dict — the convention Gemini's
  OpenAI-compatible endpoint understands for inline video. (No Agentica model
  class consumes the native ``videos=`` run kwarg today.) Gemini models are
  natively multi-modal, so a Gemini base model always takes this path.
* **Base model lacks the modality** → scan config.yaml profiles (main fields,
  ``auxiliary_model`` and ``fallback_models`` blocks) for a model that
  supports it, make a one-shot describe/transcribe call, and inject the
  result into the user message as a text part. A note is added to the reply
  so the user knows a non-base model was used.
* **Nobody supports it** → log a warning and add a user-facing note telling
  the user how to configure a capable model (e.g. a ``modalities: [image]``
  declaration on a profile with a private/unknown model name).

Voice payloads from WeChat arrive silk-encoded; they are decoded to wav first
(``pilk`` or ``graiax-silkcoder`` — pure-Python, in the ``wechat`` extra).
Videos larger than ``MAX_INLINE_VIDEO_BYTES`` are skipped (Gemini inline
payloads cap at ~20MB).
"""
import asyncio
import base64
import inspect
import io
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from agentica.cost_tracker import get_model_supports_modality
from agentica.global_config import get_profiles
from agentica.model.message import Message
from agentica.utils.log import logger

from ..channels.base import InboundMedia

# Gemini inline payloads cap at ~20MB; stay well under it.
MAX_INLINE_VIDEO_BYTES = 15 * 1024 * 1024

_MODALITY_LABELS = {"image": "图片", "audio": "语音", "video": "视频"}
_KIND_TO_MODALITY = {"image": "image", "voice": "audio", "video": "video"}

# Name hints, consulted only when the model catalog has no entry for the id
# (private deployments, brand-new releases). Catalog data wins when present.
_NAME_HINTS = {
    "image": ("gemini", "vl", "vision", "-4v", "4o", "omni", "seed", "qvq"),
    "audio": ("gemini", "omni", "audio"),
    "video": ("gemini", "omni", "seed"),
}

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
    ``text_parts`` are fallback-model descriptions/transcriptions to append
    to the user text; ``notes`` are user-facing one-liners about non-base
    models used or media that had to be skipped.
    """
    images: List[Any] = field(default_factory=list)
    audio: Optional[dict] = None
    text_parts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def supports_modality(
    model_id: str,
    modality: str,
    declared: Optional[Iterable[str]] = None,
) -> bool:
    """Whether ``model_id`` can take ``modality`` (image/audio/video) input.

    Resolution order: an explicit ``declared`` list (per-profile
    ``modalities`` in config.yaml) > the model catalog > conservative name
    hints (only reachable for models the catalog doesn't know).
    """
    declared_set = {str(m).strip().lower() for m in (declared or ())}
    if modality in declared_set:
        return True
    if get_model_supports_modality(model_id, modality):
        return True
    name = (model_id or "").lower()
    return any(hint in name for hint in _NAME_HINTS.get(modality, ()))


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
    """Decode silk to PCM with whichever optional decoder is installed."""
    try:
        import pilk
        try:
            return pilk.decode(data)
        except Exception as e:
            logger.warning(f"pilk silk decode failed: {e}")
    except ImportError:
        pass
    try:
        from graiax import silkcoder
    except ImportError:
        return None
    try:
        result = silkcoder.decode(data)
        if inspect.isawaitable(result):
            # graiax-silkcoder's API is a coroutine; decode_voice is sync and
            # is typically called on a running event loop, so run it on a
            # private loop in a helper thread.
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, result).result()
        return result
    except Exception as e:
        logger.warning(f"graiax-silkcoder silk decode failed: {e}")
        return None


def decode_voice(data: bytes) -> Optional[Tuple[bytes, str]]:
    """Normalise a voice payload to (bytes, format) for ``input_audio``.

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
    """Routes inbound media to the base model or a config.yaml fallback model."""

    def __init__(self, create_model_fn: Optional[Callable] = None):
        if create_model_fn is None:
            from .model_factory import create_model
            create_model_fn = create_model
        self._create_model = create_model_fn
        self._model_cache: Dict[tuple, Any] = {}

    # ------------------------------------------------------ model discovery
    @staticmethod
    def _iter_model_blocks(profiles: Dict[str, Any]):
        """Yield (profile, role, block) for every model defined in config.yaml.

        ``auxiliary_model`` / ``fallback_models`` entries inherit
        provider/base_url/api_key from their profile's main block when
        omitted (same-inheritance rule as global_config).
        """
        for profile, prof in profiles.items():
            if not isinstance(prof, dict):
                continue
            yield profile, "main", prof
            for role, blocks in (
                ("auxiliary", [prof.get("auxiliary_model")]),
                ("fallback", prof.get("fallback_models") or []),
            ):
                for block in blocks:
                    if not isinstance(block, dict) or not block.get("model_name"):
                        continue
                    merged = dict(block)
                    for key in ("model_provider", "base_url", "api_key"):
                        if not merged.get(key) and prof.get(key):
                            merged[key] = prof[key]
                    yield profile, role, merged

    def find_model_for(self, modality: str) -> Optional[Dict[str, Any]]:
        """First config.yaml model that supports ``modality``, or None."""
        for profile, role, block in self._iter_model_blocks(get_profiles()):
            model_name = block.get("model_name") or ""
            if supports_modality(model_name, modality, declared=block.get("modalities")):
                return {
                    "profile": profile,
                    "role": role,
                    "model_provider": block.get("model_provider") or "",
                    "model_name": model_name,
                    "base_url": block.get("base_url") or "",
                    "api_key": block.get("api_key") or "",
                    "wire_api": block.get("wire_api") or "",
                }
        return None

    # ------------------------------------------------------ one-shot calls
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

    # ------------------------------------------------------ routing
    async def prepare(
        self,
        media: List[InboundMedia],
        *,
        base_model_id: str,
        base_supports_images: bool,
        base_declared: Optional[Iterable[str]] = None,
    ) -> MediaPlan:
        """Decide how each media item reaches the agent (see module docstring)."""
        plan = MediaPlan()
        for item in media:
            try:
                await self._route(item, plan, base_model_id, base_supports_images, base_declared)
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
        base_declared: Optional[Iterable[str]],
    ) -> None:
        if item.kind == "image":
            await self._route_image(item, plan, base_supports_images)
        elif item.kind == "voice":
            await self._route_voice(item, plan, base_model_id, base_declared)
        elif item.kind == "video":
            await self._route_video(item, plan, base_model_id, base_declared)
        else:
            logger.debug(f"Media understanding: skipping unsupported kind {item.kind!r}")

    def _base_supports(self, base_model_id: str, modality: str, base_declared) -> bool:
        return supports_modality(base_model_id, modality, declared=base_declared)

    def _note_unsupported(self, plan: MediaPlan, modality: str) -> None:
        label = _MODALITY_LABELS[modality]
        logger.warning(
            f"Media understanding: no model for {modality} — base model lacks it and "
            "no config.yaml profile supports it. Declare one with `modalities:`."
        )
        plan.notes.append(
            f"⚠️ 暂不支持{label}理解：底模不支持，且 config.yaml 中没有可用的{label}模型"
            f"（可给某个 profile 加 modalities: [{modality}] 声明后重试）"
        )

    async def _describe(self, spec: Dict[str, Any], blocks: List[dict]) -> str:
        model = self._get_model(spec)
        resp = await model.response([Message(role="user", content=blocks)])
        return (getattr(resp, "content", "") or "").strip()

    async def _route_image(self, item: InboundMedia, plan: MediaPlan, base_supports_images: bool) -> None:
        mime = item.mime or sniff_image_mime(item.data)
        url = f"data:{mime};base64,{base64.b64encode(item.data).decode()}"
        if base_supports_images:
            plan.images.append({"url": url})
            return
        spec = self.find_model_for("image")
        if spec is None:
            self._note_unsupported(plan, "image")
            return
        text = await self._describe(spec, [
            {"type": "text", "text": _IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ])
        plan.text_parts.append(f"[图片内容]\n{text}")
        plan.notes.append(f"🖼 图片由 {spec['model_name']} 识别（底模不支持读图）")

    async def _route_voice(self, item: InboundMedia, plan: MediaPlan, base_model_id: str, base_declared) -> None:
        decoded = decode_voice(item.data)
        if decoded is None:
            logger.warning("Media understanding: voice decode failed (no silk decoder installed?)")
            plan.notes.append(
                "⚠️ 语音解码失败：缺少 silk 解码库，请安装 graiax-silkcoder 或 pilk"
                "（pip install 'agentica[wechat]'）"
            )
            return
        payload, fmt = decoded
        if plan.audio is None and self._base_supports(base_model_id, "audio", base_declared):
            plan.audio = {"data": base64.b64encode(payload).decode(), "format": fmt}
            return
        spec = self.find_model_for("audio")
        if spec is None:
            self._note_unsupported(plan, "audio")
            return
        text = await self._describe(spec, [
            {"type": "text", "text": _VOICE_PROMPT},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(payload).decode(), "format": fmt}},
        ])
        plan.text_parts.append(f"[语音转写]\n{text}")
        plan.notes.append(f"🎤 语音由 {spec['model_name']} 转写（底模不支持语音）")

    async def _route_video(self, item: InboundMedia, plan: MediaPlan, base_model_id: str, base_declared) -> None:
        if len(item.data) > MAX_INLINE_VIDEO_BYTES:
            logger.warning(f"Media understanding: video too large ({len(item.data)} bytes), skipped")
            plan.notes.append(f"⚠️ 视频过大（>{MAX_INLINE_VIDEO_BYTES // 1024 // 1024}MB），已跳过")
            return
        url = f"data:video/mp4;base64,{base64.b64encode(item.data).decode()}"
        if self._base_supports(base_model_id, "video", base_declared):
            plan.images.append({"url": url})
            return
        spec = self.find_model_for("video")
        if spec is None:
            self._note_unsupported(plan, "video")
            return
        text = await self._describe(spec, [
            {"type": "text", "text": _VIDEO_PROMPT},
            {"type": "image_url", "image_url": {"url": url}},
        ])
        plan.text_parts.append(f"[视频内容]\n{text}")
        plan.notes.append(f"🎬 视频由 {spec['model_name']} 解读（底模不支持视频）")


# Process-wide singleton — the fallback model cache lives here.
media_understanding = MediaUnderstandingService()

__all__ = [
    "MAX_INLINE_VIDEO_BYTES",
    "MediaPlan",
    "MediaUnderstandingService",
    "decode_voice",
    "media_understanding",
    "sniff_image_mime",
    "supports_modality",
]
