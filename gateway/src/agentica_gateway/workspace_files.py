# -*- coding: utf-8 -*-
"""Path resolution for the web workspace file browser.

A relative path, once resolved, must stay inside the session's working
directory. The check is lexical and then canonical (realpath) so ``..`` and
symlink escapes are both refused.
"""
from __future__ import annotations

from pathlib import Path
# 256 KiB preview cap; 50 MiB hard ceiling for a full inline read.
TEXT_PREVIEW_BYTES = 256 * 1024
MAX_READ_BYTES = 50 * 1024 * 1024

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".htm": "text/html; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".json": "application/json",
    ".js": "text/javascript; charset=utf-8",
    ".ts": "text/plain; charset=utf-8",
    ".tsx": "text/plain; charset=utf-8",
    ".jsx": "text/plain; charset=utf-8",
    ".py": "text/plain; charset=utf-8",
    ".sh": "text/plain; charset=utf-8",
    ".yaml": "text/plain; charset=utf-8",
    ".yml": "text/plain; charset=utf-8",
    ".toml": "text/plain; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".csv": "text/plain; charset=utf-8",
    ".log": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}

# Same-origin HTML/SVG would execute in the browser; inline preview falls back
# to text/plain so the SPA can still show source.
SCRIPTABLE = {".html", ".htm", ".svg"}


class WorkspacePathError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _is_inside(target: Path, base: Path) -> bool:
    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def resolve_root(root: str) -> Path:
    if not root or "\0" in root:
        raise WorkspacePathError(400, "root is required")
    base = Path(root).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise WorkspacePathError(404, "workspace root is missing")
    return base


def lexical_target(base: Path, rel: str) -> Path:
    if "\0" in (rel or ""):
        raise WorkspacePathError(400, "path is invalid")
    target = (base / (rel or ".")).resolve()
    if not _is_inside(target, base):
        raise WorkspacePathError(400, "path must be inside the workspace")
    return target


def resolve_existing(root: str, rel: str) -> Path:
    """Canonical path of an existing file or directory inside ``root``."""
    base = resolve_root(root)
    target = lexical_target(base, rel)
    try:
        canonical = target.resolve(strict=True)
    except FileNotFoundError:
        raise WorkspacePathError(404, "path not found") from None
    if not _is_inside(canonical, base):
        raise WorkspacePathError(400, "path must be inside the workspace")
    return canonical


def resolve_write(root: str, rel: str) -> Path:
    """Lexical destination for a write; parent must stay inside ``root``."""
    if not rel or rel.endswith("/") or rel.endswith("\\"):
        raise WorkspacePathError(400, "upload path must be a file")
    base = resolve_root(root)
    dest = lexical_target(base, rel)
    parent = dest.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    parent_real = parent.resolve()
    if not _is_inside(parent_real, base) and parent_real != base:
        raise WorkspacePathError(400, "path must be inside the workspace")
    return dest


def content_type_for(name: str, *, inline: bool) -> str:
    ext = Path(name).suffix.lower()
    if inline and ext in SCRIPTABLE:
        return "text/plain; charset=utf-8"
    return CONTENT_TYPES.get(ext, "application/octet-stream")


def is_text_ext(name: str) -> bool:
    ext = Path(name).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".gz", ".tar"}:
        return False
    ctype = CONTENT_TYPES.get(ext, "text/plain; charset=utf-8")
    return ctype.startswith("text/") or ctype in {"application/json", "image/svg+xml"}


def list_entries(root: str, rel: str) -> list[dict]:
    directory = resolve_existing(root, rel)
    if not directory.is_dir():
        raise WorkspacePathError(400, "path is not a directory")
    base = resolve_root(root)
    rows: list[dict] = []
    try:
        children = list(directory.iterdir())
    except PermissionError:
        return []
    for child in children:
        try:
            canonical = child.resolve()
        except OSError:
            continue
        if not _is_inside(canonical, base):
            continue
        try:
            st = canonical.stat()
        except OSError:
            continue
        is_dir = canonical.is_dir()
        rows.append({
            "name": child.name,
            "kind": "dir" if is_dir else "file",
            "sizeBytes": 0 if is_dir else st.st_size,
            "mtime": _mtime_iso(st.st_mtime),
        })
    rows.sort(key=lambda r: (0 if r["kind"] == "dir" else 1, r["name"].lower()))
    return rows


def _mtime_iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def stat_existing(root: str, paths: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for rel in paths:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        try:
            p = resolve_existing(root, rel)
        except WorkspacePathError:
            continue
        if p.is_file():
            found.append(rel)
    return found


def read_preview_text(path: Path, *, limit: int = TEXT_PREVIEW_BYTES) -> tuple[str, bool]:
    """First ``limit`` bytes only — do not slurp a multi-megabyte file then slice."""
    size = path.stat().st_size
    with path.open("rb") as f:
        data = f.read(limit)
    truncated = size > len(data)
    return data.decode("utf-8", errors="replace"), truncated
