# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Built-in image understanding tool (analyze_image)

One tool, three ways to answer, chosen by what this installation can actually
do — never by guessing from a filename or a model id:

1. **the agent's own eyes** — when the agent's model reads images, the pixels
   are handed straight to it (``multimodal_tool_result``). No extra LLM call,
   no transcription loss.
2. **a vision model** — a separately configured vision model describes the
   image and the agent reads that description.
3. **OCR** — ``imgocr`` extracts text locally, no network needed.

OCR is deliberately *not* a second tool. Which tier runs is decided by
capability, which is not something the model should have to reason about: it
asks one question about one image and gets the best answer available here.
"""

import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, Optional

from agentica.model.media import get_image_type, multimodal_tool_result
from agentica.tools.base import Tool
from agentica.utils.log import logger
from agentica.utils.ocr import ocr_image_text

_MIME_BY_TYPE = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "heic": "image/heic",
}

_DEFAULT_QUESTION = "详细描述图片内容"
_OCR_MAX_CHARS = 4000


def _data_url(raw: bytes) -> str:
    """Encode bytes as a data URL typed by magic bytes, not by filename.

    A PNG saved as ``.jpg`` must not be announced as JPEG: providers decode by
    the declared type and reject the mismatch.
    """
    mime = _MIME_BY_TYPE.get(get_image_type(raw) or "", "image/png")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"


class BuiltinVisionTool(Tool):
    """Answer a question about an image, however this installation is able to."""

    def __init__(
        self,
        vision_model: Optional[Any] = None,
        work_dir: Optional[str] = None,
    ):
        super().__init__(name="builtin_vision_tool")
        self.vision_model = vision_model
        self.work_dir = work_dir
        self._agent_model: Optional[Any] = None
        self.register(self.analyze_image)

    def set_agent_model(self, model: Optional[Any]) -> None:
        """Receive the agent's own model (called by Agent at build time)."""
        self._agent_model = model

    async def analyze_image(
        self,
        image_path_or_url: str,
        question: str = "",
    ) -> Any:
        """Look at an image and answer a question about it.

        Use this for any image — screenshot, diagram, photo, scan — whether the
        user pasted a path, a tool produced one, or you found one on disk. Pass
        the path exactly as you received it. For several images, call this once
        per image.

        Args:
            image_path_or_url: Local file path, or an http(s):// image URL.
            question: What you want to know about the image. Defaults to a full
                description.

        Returns:
            The answer, or the image itself when you can see it directly.
        """
        source = str(image_path_or_url or "").strip()
        if not source:
            raise ValueError("image_path_or_url is required")

        question = (question or "").strip() or _DEFAULT_QUESTION
        is_url = source.startswith(("http://", "https://"))

        url = source
        local_path: Optional[Path] = None
        if not is_url:
            path = Path(source).expanduser()
            if not path.is_absolute() and self.work_dir:
                path = Path(self.work_dir) / path
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path_or_url}")
            raw = await asyncio.get_running_loop().run_in_executor(None, path.read_bytes)
            if get_image_type(raw) is None:
                raise ValueError(
                    f"Not a recognizable image: {image_path_or_url}. "
                    "Supported: png, jpeg, gif, webp, heic."
                )
            url = _data_url(raw)
            local_path = path

        # 1. The agent's own eyes. The model layer attaches the image to a
        #    follow-up user message, so this works on every wire that supports
        #    images at all — no per-provider special casing.
        if getattr(self._agent_model, "supports_images", False):
            return multimodal_tool_result(
                text=(
                    "The image is attached below — look at it and answer using "
                    f"your own vision.\n\nQuestion: {question}"
                ),
                images=[{"url": url}],
                text_summary=f"[image attached for direct viewing: {source}]",
            )

        # 2. A separately configured vision model describes it.
        described = await self._describe_with_vision_model(url, question, source)
        if described:
            return described

        # 3. OCR recovers the text, which for a screenshot is often the point.
        if local_path is not None:
            text = await self._ocr(str(local_path))
            if text:
                return (
                    f"[OCR text from {local_path.name} — no vision model available, so "
                    f"this is extracted text only, not a description of the image]\n{text}"
                )

        raise RuntimeError(
            f"Cannot analyze {image_path_or_url}: this agent's model does not "
            "accept images, no vision model is configured, and OCR produced "
            "nothing. Use a vision-capable model, or `pip install imgocr`."
        )

    async def _describe_with_vision_model(
        self, url: str, question: str, source: str
    ) -> Optional[str]:
        """Ask a configured vision model to describe the image.

        Goes through ``Model.response`` rather than a raw ``chat.completions``
        call, so the provider, wire protocol, base_url and auth in effect are
        the configured ones. A direct client call would hardcode one wire format
        and send this model's id to whatever endpoint happened to be set — which
        is how the old image tool ended up asking a private gateway for
        ``gpt-4o``.
        """
        model = self.vision_model
        if model is None:
            return None

        from agentica.model.message import Message

        try:
            response = await model.response(
                [Message(role="user", content=question, images=[{"url": url}])]
            )
        except Exception as error:
            logger.warning(f"Vision model failed on {source}: {error}")
            return None

        content = getattr(response, "content", None)
        return content.strip() if isinstance(content, str) and content.strip() else None

    @staticmethod
    async def _ocr(image_path: str) -> str:
        """Local text extraction with imgocr, off the event loop."""
        return await asyncio.get_running_loop().run_in_executor(
            None, ocr_image_text, image_path, _OCR_MAX_CHARS
        )
