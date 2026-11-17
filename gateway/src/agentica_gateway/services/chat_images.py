# -*- coding: utf-8 -*-
"""Images attached in the web chat, kept so a reload shows what was sent.

The session log stores the user turn as text. When the base model cannot see
images, the media model's description is appended to that text, and the picture
itself used to be gone after a refresh: the bubble turned into a wall of
description. Each image is therefore written once under
``$AGENTICA_HOME/gateway/chat-images/<sha256>.<ext>`` and the message carries a
marker (``agentica.session.rows.image_marker``); the session API serves the
names back as ``row.images`` and the Web turns them into thumbnails.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from agentica.config import AGENTICA_HOME

_EXT_BY_MIME = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}
MEDIA_TYPE_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}

_NAME_RE = re.compile(r"^[0-9a-f]{64}\.(png|jpg|gif|webp)$")


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


def resolve(name: str) -> Optional[Path]:
    """The stored file for ``name``, or None. Only well-formed names are looked up."""
    if not _NAME_RE.match(name or ""):
        return None
    path = images_dir() / name
    return path if path.is_file() else None
