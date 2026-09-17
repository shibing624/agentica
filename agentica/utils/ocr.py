# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Local OCR text extraction (imgocr), shared by every caller.

Two very different callers need the same extraction and disagree only on how
much text they can afford: the CLI injects OCR text into a prompt when the
model cannot see an attached image, while ``analyze_image`` returns it as a
tool result. The budget is therefore the caller's argument, not a constant
baked in here — that difference is the only reason these ever diverged.
"""

from typing import Optional

from agentica.utils.log import logger


def ocr_image_text(image_path: str, max_chars: Optional[int] = None) -> str:
    """Extract text from an image with imgocr; "" when unavailable or empty.

    A missing ``imgocr`` is not an error: OCR is a fallback for installations
    whose model cannot read images, so callers treat "" as "nothing to add" and
    move on. Blocking — run it in an executor from async code.
    """
    try:
        from imgocr import ImgOcr
    except ImportError:
        logger.debug("imgocr not installed; OCR unavailable")
        return ""

    try:
        result = ImgOcr().ocr(image_path)
    except Exception as error:
        logger.warning(f"OCR failed for {image_path}: {error}")
        return ""

    text = " ".join(item["text"] for item in result if "text" in item)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"
    return text
