# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Helpers for normalizing Agent.run input.
"""

from dataclasses import replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentica.model.message import Message
from agentica.run_config import RunConfig


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_KNOWN_RUN_KWARGS = {
    "message",
    "messages",
    "audio",
    "images",
    "videos",
    "timeout",
    "hooks",
    "config",
}
_REMOVED_RUN_KWARGS = {"add_messages"}


def _is_content_block_sequence(items: Sequence[Any]) -> bool:
    return bool(items) and all(isinstance(item, dict) and "type" in item for item in items)


def _is_image_path_or_url(value: str) -> bool:
    if value.startswith("data:image/"):
        return True

    path_part = value.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(path_part).suffix.lower()
    if suffix not in _IMAGE_SUFFIXES:
        return False

    if value.startswith(("http://", "https://")):
        return True

    path = Path(value).expanduser()
    if not path.exists():
        return False
    return True


def _split_sequence_content(items: Sequence[Any]) -> Tuple[str, List[Any]]:
    text_parts: List[str] = []
    parsed_images: List[Any] = []

    for item in items:
        if isinstance(item, str):
            if _is_image_path_or_url(item):
                parsed_images.append(item)
            else:
                text_parts.append(item)
        elif isinstance(item, bytes):
            parsed_images.append(item)
        else:
            parsed_images.append(item)

    return "\n".join(text_parts), parsed_images


def build_user_message_from_sequence(
    sequence: Sequence[Any],
    *,
    role: str = "user",
    audio: Optional[Any] = None,
    images: Optional[Sequence[Any]] = None,
    videos: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> Message:
    """Build a user Message from a list-style run input."""
    if _is_content_block_sequence(sequence):
        return Message(
            role=role,
            content=list(sequence),
            audio=audio,
            images=images,
            videos=videos,
            **kwargs,
        )

    text_content, parsed_images = _split_sequence_content(sequence)
    merged_images = parsed_images + list(images or [])
    return Message(
        role=role,
        content=text_content,
        audio=audio,
        images=merged_images or None,
        videos=videos,
        **kwargs,
    )


def merge_run_config(
    config: Optional[RunConfig] = None,
    *,
    timeout: Optional[float] = None,
    hooks: Optional[Any] = None,
) -> RunConfig:
    """Merge inline run options over RunConfig without mutating the caller's object."""
    merged = config or RunConfig()
    if timeout is not None:
        merged = replace(merged, run_timeout=timeout)
    if hooks is not None:
        merged = replace(merged, hooks=hooks)
    return merged


def reject_unknown_run_kwargs(kwargs: Dict[str, Any]) -> None:
    """Reject unknown run() kwargs loudly. A silently-dropped argument is the
    worst kind of bug — the caller thinks a setting took effect when it didn't.
    Misspellings get a 'did you mean' hint."""
    removed = sorted(set(kwargs) & _REMOVED_RUN_KWARGS)
    if removed:
        raise TypeError(
            "add_messages was removed. Pass a complete transcript via messages=[...] instead."
        )

    known = sorted(_KNOWN_RUN_KWARGS | _REMOVED_RUN_KWARGS)
    for key in kwargs:
        if key in _KNOWN_RUN_KWARGS:
            continue
        matches = get_close_matches(key, known, n=1, cutoff=0.82)
        hint = f" Did you mean '{matches[0]}'?" if matches else ""
        raise TypeError(f"Unknown run() keyword argument '{key}'.{hint}")
