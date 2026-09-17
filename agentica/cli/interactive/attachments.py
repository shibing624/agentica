# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image/path attachment helpers for the interactive prompt
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List, NamedTuple, Optional

from agentica.cli.commands.context import IMAGE_EXTENSIONS
from agentica.utils.log import logger
from agentica.utils.ocr import ocr_image_text

# ==================== Pasted-block Handles ====================

#: Shape of the placeholder the bracketed-paste handler leaves in the buffer.
#: Owned here because two input paths consume it and they must not drift: the
#: idle turn inlines the file (process loop in ``app.py``), steering keeps the
#: handle and annotates it.
PASTE_REF_RE = re.compile(r"\[Pasted text #\d+: \d+ lines -> (.+?)\]")

_PASTE_STEER_NOTE = (
    "(The pasted block above is not inlined here — its full text is in the "
    "file the placeholder points at. Use read_file on that path when you "
    "need the content.)"
)


def annotate_paste_refs(text: str) -> str:
    """Say what a steered paste placeholder is, without inlining the file.

    An idle turn can afford to inline: it happens once, before the request is
    built. Steering is injected into a *running* run and then replayed in
    every later request of that run, so a few hundred pasted lines would be
    paid for on every lap — which is why the placeholder exists at all. The
    handle carries an absolute path, so the model can read it on demand; what
    was missing is any statement that it can.
    """
    if not PASTE_REF_RE.search(text):
        return text
    return f"{text}\n\n{_PASTE_STEER_NOTE}"

# ==================== Image Attachment Helpers ====================


def _split_path_input(raw: str) -> tuple:
    """Split a leading file path token from trailing free-form text."""
    raw = str(raw or "").strip()
    if not raw:
        return "", ""

    if raw[0] in {'"', "'"}:
        quote = raw[0]
        pos = 1
        while pos < len(raw):
            ch = raw[pos]
            if ch == "\\" and pos + 1 < len(raw):
                pos += 2
                continue
            if ch == quote:
                token = raw[1:pos]
                remainder = raw[pos + 1 :].strip()
                return token, remainder
            pos += 1
        return raw[1:], ""

    pos = 0
    while pos < len(raw):
        ch = raw[pos]
        if ch == "\\" and pos + 1 < len(raw) and raw[pos + 1] == " ":
            pos += 2
        elif ch == " ":
            break
        else:
            pos += 1

    token = raw[:pos].replace("\\ ", " ")
    remainder = raw[pos:].strip()
    return token, remainder


def _resolve_attachment_path(raw_path: str) -> Optional[Path]:
    """Resolve a user-supplied local attachment path."""
    token = str(raw_path or "").strip()
    if not token:
        return None
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        token = token[1:-1].strip()
    if not token:
        return None

    expanded = os.path.expandvars(os.path.expanduser(token))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(os.getcwd()) / path

    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


class QueuedInput(NamedTuple):
    """One pending-queue payload, classified by who put it there."""

    text: str
    images: List[Path]
    is_btw: bool
    is_relayed: bool


def unpack_queue_payload(item) -> QueuedInput:
    """Classify a pending-queue payload into text plus how it got there.

    ``is_relayed`` marks text nobody typed in this terminal — a peer message or
    a finished job's report, both queued by ``hand_to_agent``. It decides two
    things, and both are one rule: **the input line's affordances belong to
    whoever typed on it.** Relayed text is not echoed as a user turn (its arrival
    was already printed in its own shape) and never dispatches a slash command.

    Single source for the three readers of the queue (the process loop, the TUI
    queue bar, the goal loop's "is real work waiting" check), which each used to
    unpack the tuple forms themselves and disagreed about the markers.
    """
    if isinstance(item, tuple):
        if item and item[0] == "__BTW__":
            return QueuedInput(str(item[1]) if len(item) > 1 else "", [], True, False)
        if item and item[0] == "__RELAYED__":
            return QueuedInput(str(item[1]) if len(item) > 1 else "", [], False, True)
        text, images = item
        return QueuedInput(str(text), list(images or []), False, False)
    return QueuedInput(str(item), [], False, False)


def queue_item_preview(item) -> str:
    """Render one pending-queue payload for the TUI ``Queued (N):`` bar.

    Every queued payload is shown as it was entered, including slash commands
    and skill invocations — hiding them made a queued
    ``/requesting-code-review ...`` look like it never entered the queue.
    """
    queued = unpack_queue_payload(item)
    if queued.is_btw:
        return f"__BTW__: {queued.text}" if queued.text else "__BTW__"
    return queued.text


def peel_image_input(text: str, attached: Optional[list] = None) -> tuple:
    """Split a leading image path (and Ctrl+V attachments) from caption text.

    Idle turns already do this via ``_detect_file_drop`` in the process loop.
    Mid-run Enter has to do it too, or a pasted
    ``/var/folders/...png 如图所示`` steers as a path string instead of a
    vision image plus caption.
    """
    extra = list(attached or [])
    dropped = _detect_file_drop(text)
    if dropped and dropped["is_image"]:
        extra.append(dropped["path"])
        text = dropped["remainder"]
    return text, _deduplicate_image_attachments(extra)


def _detect_file_drop(user_input: str) -> Optional[dict]:
    """Detect if user_input starts with a real local file path."""
    if not isinstance(user_input, str):
        return None
    stripped = user_input.strip()
    if not stripped:
        return None

    starts_like_path = (
        stripped.startswith("/")
        or stripped.startswith("~")
        or stripped.startswith("./")
        or stripped.startswith("../")
        or stripped.startswith('"/')
        or stripped.startswith('"~')
        or stripped.startswith("'/")
        or stripped.startswith("'~")
    )
    if not starts_like_path:
        return None

    first_token, remainder = _split_path_input(stripped)
    drop_path = _resolve_attachment_path(first_token)
    if drop_path is None:
        return None

    return {
        "path": drop_path,
        "is_image": drop_path.suffix.lower() in IMAGE_EXTENSIONS,
        "remainder": remainder,
    }


def _image_content_key(path: Path) -> str:
    """Return a stable content key used to collapse duplicate temp image files."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_image_attachments(paths: List[Path]) -> List[Path]:
    """Keep the first path for each distinct pasted image."""
    unique: List[Path] = []
    seen = set()
    for path in paths:
        key = _image_content_key(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _try_attach_clipboard_image(attached_images: list, image_counter: list) -> bool:
    """Check clipboard for an image and attach it if found."""
    from agentica.cli.clipboard import save_clipboard_image
    from agentica.cli.runtime import CACHE_DIR

    img_dir = Path(CACHE_DIR) / "images"
    image_counter[0] += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = img_dir / f"clip_{ts}_{image_counter[0]}.png"

    if save_clipboard_image(img_path):
        attached_images.append(img_path)
        return True
    image_counter[0] -= 1
    return False


# ==================== Image OCR fallback ====================

# Per-image and total limits for OCR text injection
_OCR_PER_IMAGE_CHARS = 50_000
_OCR_TOTAL_CHARS = 200_000
_OCR_TIMEOUT_SECS = 30


def _ocr_single_image(image_path: str) -> str:
    """OCR a single image, returning extracted text (truncated to limit).

    Extraction itself lives in ``agentica.utils.ocr`` so the ``analyze_image``
    tool and this prompt-injection path cannot drift apart; only the character
    budget differs between them.
    """
    return ocr_image_text(image_path, _OCR_PER_IMAGE_CHARS)


def _ocr_images_parallel(image_paths: list) -> str:
    """OCR multiple images in parallel with timeout. Returns combined text."""
    results = []
    total_len = 0
    with ThreadPoolExecutor(max_workers=min(len(image_paths), 4)) as pool:
        futures = {pool.submit(_ocr_single_image, p): p for p in image_paths}
        for future in futures:
            path = futures[future]
            name = Path(path).name
            try:
                text = future.result(timeout=_OCR_TIMEOUT_SECS)
            except Exception as error:
                logger.warning(f"OCR failed for {path}: {error}")
                continue

            if not text:
                continue

            if total_len + len(text) > _OCR_TOTAL_CHARS:
                remaining = _OCR_TOTAL_CHARS - total_len
                if remaining > 100:
                    text = text[:remaining] + "\n..."
                else:
                    break

            if len(image_paths) > 1:
                results.append(f"[{name}]\n{text}")
            else:
                results.append(text)
            total_len += len(text)

    return "\n\n".join(results)


__all__ = ['PASTE_REF_RE', 'annotate_paste_refs', '_split_path_input', '_resolve_attachment_path', 'queue_item_preview', 'peel_image_input', '_detect_file_drop', '_image_content_key', '_deduplicate_image_attachments', '_try_attach_clipboard_image', '_OCR_PER_IMAGE_CHARS', '_OCR_TOTAL_CHARS', '_OCR_TIMEOUT_SECS', '_ocr_single_image', '_ocr_images_parallel']
