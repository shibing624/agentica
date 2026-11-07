# -*- coding: utf-8 -*-
"""Images attached in the web chat, kept so a reload shows what was sent.

The session log stores the user turn as text. When the base model cannot see
images, the media model's description is appended to that text, and the picture
itself used to be gone after a refresh: the bubble turned into a wall of
description. Each image is therefore written once under
``$AGENTICA_HOME/gateway/chat-images/<sha256>.<ext>`` and the message carries a
``[图片: <name>]`` marker the web UI turns back into a thumbnail (and strips,
together with whatever follows, from the visible text).

Markers sit after the user's own text and before any media-model description,
so everything from the first marker on is machine-added.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

from agentica.config import AGENTICA_HOME

MARK_PREFIX = "[图片: "

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}
MEDIA_TYPE_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}

_NAME_RE = re.compile(r"^[0-9a-f]{64}\.(png|jpg|gif|webp)$")
# Where machine-added text begins: a marker at the start of the message (no
# text typed) or after a blank line.
_CUT_RE = re.compile(r"(?:^|\n\n)\[图片: [0-9a-f]")


def images_dir() -> Path:
    return Path(AGENTICA_HOME).expanduser() / "gateway" / "chat-images"


def save(data: bytes, mime: str = "") -> str:
    """Store ``data`` content-addressed; returns its file name."""
    ext = _EXT_BY_MIME.get((mime or "").lower(), "png")
    name = f"{hashlib.sha256(data).hexdigest()}.{ext}"
    path = images_dir() / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    return name


def marker(names: Iterable[str]) -> str:
    return "".join(f"\n\n{MARK_PREFIX}{n}]" for n in names)


def resolve(name: str) -> Optional[Path]:
    """The stored file for ``name``, or None. Only well-formed names are looked up."""
    if not _NAME_RE.match(name or ""):
        return None
    path = images_dir() / name
    return path if path.is_file() else None


def strip_markers(text: str) -> str:
    """``text`` without the markers and the media description behind them."""
    m = _CUT_RE.search(text or "")
    return text[: m.start()] if m else text
