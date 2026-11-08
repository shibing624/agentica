# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deliverable file reference extraction for IM channels.

Adapted from OpenBitFun's ``remote_connect/bot/mod.rs``:
- Extracts markdown links ``[label](ref)``, images ``![alt](ref)``, and
  ``file://`` / ``computer://`` references outside fenced/inline code blocks.
- Excludes source code and config files when referenced by path so citing a
  ``.py`` or ``.ts`` file in an explanation never triggers a file upload.
- Resolves relative paths against the turn's ``work_dir`` without allowing
  ``..`` traversal outside that root.
"""
from __future__ import annotations

import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import unquote, urlparse

MAX_AUTO_PUSH_BYTES = 20 * 1024 * 1024

CODE_FILE_EXTENSIONS: Set[str] = {
    "js", "jsx", "ts", "tsx", "mjs", "cjs", "mts", "cts",
    "py", "pyw", "pyi", "rs", "go", "java", "kt", "kts",
    "scala", "groovy", "c", "cpp", "cc", "cxx", "h", "hpp",
    "hxx", "hh", "cs", "rb", "php", "swift", "vue", "svelte",
    "css", "scss", "less", "sass", "json", "jsonc", "yaml",
    "yml", "toml", "xml", "md", "mdx", "rst", "txt", "sh",
    "bash", "zsh", "fish", "ps1", "bat", "cmd", "sql",
    "graphql", "gql", "proto", "lock", "env", "ini", "cfg",
    "conf", "editorconfig", "gitignore", "log",
}

DOWNLOADABLE_EXTENSIONS: Set[str] = {
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
    "odt", "ods", "odp", "rtf", "pages", "numbers", "key",
    "png", "jpg", "jpeg", "gif", "bmp", "svg", "webp", "ico",
    "tiff", "tif", "zip", "tar", "gz", "bz2", "7z", "rar",
    "dmg", "iso", "xz", "mp3", "wav", "ogg", "flac", "aac",
    "m4a", "wma", "mp4", "avi", "mkv", "mov", "webm", "wmv",
    "flv", "csv", "tsv", "sqlite", "db", "parquet", "epub",
    "mobi", "html", "htm", "apk", "ipa",
}

_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MD_LINK_RE = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_SCHEME_PATH_RE = re.compile(r"(?:computer|file)://[^\s<>()\"']+")


@dataclass(frozen=True)
class AutoPushFile:
    """One resolved deliverable file ready to send over an IM channel."""

    path: str
    name: str
    size: int
    mime_type: str


def format_file_size(num_bytes: int) -> str:
    """Human-readable file size (e.g. ``1.4MB``, ``320KB``, ``512B``)."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024.0 * 1024.0):.1f}MB"
    if num_bytes >= 1024:
        return f"{num_bytes // 1024}KB"
    return f"{num_bytes}B"


def detect_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def is_downloadable_by_extension(file_path: str, *, is_image_syntax: bool = False) -> bool:
    ext = Path(file_path).suffix.lstrip(".").lower()
    if not ext:
        return False
    if is_image_syntax and ext in ("png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"):
        return True
    is_abs = file_path.startswith("/") or (len(file_path) >= 3 and file_path[1] == ":")
    if is_abs:
        return ext not in CODE_FILE_EXTENSIONS and ext in DOWNLOADABLE_EXTENSIONS
    return ext in DOWNLOADABLE_EXTENSIONS


def resolve_workspace_path(raw: str, work_dir: Optional[str]) -> Optional[Path]:
    cleaned = (raw or "").strip()
    if not cleaned:
        return None
    for prefix in ("computer://", "file://"):
        if cleaned.startswith(prefix):
            parsed = urlparse(cleaned)
            cleaned = unquote(parsed.path or cleaned[len(prefix):])
            break
    else:
        cleaned = unquote(cleaned)

    candidate = Path(os.path.expanduser(cleaned))
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except OSError:
            return None

    if not work_dir:
        return None
    try:
        root = Path(os.path.expanduser(work_dir)).resolve()
        resolved = (root / candidate).resolve()
    except OSError:
        return None
    if resolved != root and not str(resolved).startswith(str(root) + os.sep):
        return None
    return resolved


def extract_output_file_references(text: str) -> List[str]:
    """Extract deliverable file references from Markdown outside code blocks."""
    if not text:
        return []
    stripped = _FENCED_CODE_RE.sub(" ", text)
    stripped = _INLINE_CODE_RE.sub(" ", stripped)

    refs: List[str] = []
    seen: Set[str] = set()

    def _add(ref: str, is_img: bool = False) -> None:
        ref = ref.strip().rstrip(".,;:)]")
        if not ref or ref in seen:
            return
        explicit = ref.startswith(("computer://", "file://"))
        if not explicit and (
            "://" in ref
            or ref.startswith(("#", "//", "data:", "mailto:", "javascript:"))
        ):
            return
        check_target = ref
        for pfx in ("computer://", "file://"):
            if check_target.startswith(pfx):
                check_target = check_target[len(pfx):]
                break
        if not explicit and not is_downloadable_by_extension(check_target, is_image_syntax=is_img):
            return
        seen.add(ref)
        refs.append(ref)

    for match in _MD_LINK_RE.finditer(stripped):
        is_image = bool(match.group(1))
        _add(match.group(2), is_img=is_image)

    for match in _SCHEME_PATH_RE.finditer(stripped):
        _add(match.group(0), is_img=False)

    return refs


def collect_auto_push_files(
    text: str,
    work_dir: Optional[str],
    max_size: int = MAX_AUTO_PUSH_BYTES,
) -> List[AutoPushFile]:
    """Resolve existing, size-bounded deliverable files referenced in ``text``."""
    out: List[AutoPushFile] = []
    seen_paths: Set[str] = set()
    for ref in extract_output_file_references(text):
        resolved = resolve_workspace_path(ref, work_dir)
        if resolved is None or not resolved.is_file():
            continue
        real_str = str(resolved)
        if real_str in seen_paths:
            continue
        if not is_downloadable_by_extension(real_str, is_image_syntax=True):
            continue
        try:
            size = resolved.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > max_size:
            continue
        seen_paths.add(real_str)
        out.append(
            AutoPushFile(
                path=real_str,
                name=resolved.name,
                size=size,
                mime_type=detect_mime_type(resolved),
            )
        )
    return out
