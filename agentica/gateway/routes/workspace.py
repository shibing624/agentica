# -*- coding: utf-8 -*-
"""Workspace file browser: list / preview / download / upload / stat.

Scoped to a client-supplied ``root`` (the session working directory). Every
resolved path must stay inside that root — same containment as the penguin
workspace browser, without a second preview origin.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..workspace_files import (
    CONTENT_TYPES,
    MAX_READ_BYTES,
    TEXT_PREVIEW_BYTES,
    WorkspacePathError,
    content_type_for,
    is_text_ext,
    list_entries,
    read_preview_text,
    resolve_existing,
    resolve_write,
    stat_existing,
)

router = APIRouter()


def _raise(err: WorkspacePathError) -> None:
    raise HTTPException(status_code=err.status, detail=err.detail) from err


class StatRequest(BaseModel):
    root: str
    paths: List[str] = Field(default_factory=list)


@router.get("/api/workspace/files")
async def workspace_list(
    root: str = Query(...),
    path: str = Query(""),
):
    try:
        entries = list_entries(root, path)
    except WorkspacePathError as e:
        _raise(e)
        return
    return {"path": path, "root": str(Path(root).expanduser().resolve()), "entries": entries}


@router.get("/api/workspace/content")
async def workspace_content(
    root: str = Query(...),
    path: str = Query(...),
    download: int = Query(0),
    preview: int = Query(0),
):
    try:
        target = resolve_existing(root, path)
    except WorkspacePathError as e:
        _raise(e)
        return
    if not target.is_file():
        raise HTTPException(status_code=400, detail="path is not a file")
    name = target.name
    if preview:
        if not is_text_ext(name):
            raise HTTPException(status_code=415, detail="not a text file")
        text, truncated = read_preview_text(target)
        return {"name": name, "content": text, "truncated": truncated, "limit": TEXT_PREVIEW_BYTES}
    size = target.stat().st_size
    if size > MAX_READ_BYTES and not download:
        raise HTTPException(status_code=413, detail="file too large to preview")
    media = content_type_for(name, inline=not download)
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
        media = CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")
    return FileResponse(target, media_type=media, headers=headers, filename=name if download else None)


@router.post("/api/workspace/stat")
async def workspace_stat(body: StatRequest):
    if len(body.paths) > 200:
        raise HTTPException(status_code=400, detail="too many paths")
    try:
        existing = stat_existing(body.root, body.paths[:200])
    except WorkspacePathError as e:
        _raise(e)
        return
    return {"existing": existing}


@router.post("/api/workspace/upload")
async def workspace_upload(
    file: UploadFile = File(...),
    root: str = Form(...),
    path: str = Form(""),
):
    """Write ``file`` into ``root`` / ``path`` / filename (path is the directory)."""
    ext = Path(file.filename or "").suffix.lower()
    allowed = settings.upload_allowed_ext_set
    if allowed and ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(allowed))}",
        )
    rel_dir = path.strip().strip("/")
    name = Path(file.filename or "upload").name
    rel = f"{rel_dir}/{name}" if rel_dir else name
    try:
        dest = resolve_write(root, rel)
    except WorkspacePathError as e:
        _raise(e)
        return
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds size limit of {settings.upload_max_size_mb}MB",
            )
        chunks.append(chunk)
    dest.write_bytes(b"".join(chunks))
    return {"status": "ok", "path": rel, "filename": name, "size": total}
