# -*- coding: utf-8 -*-
"""Workspace file browser: list / preview / download / upload / stat.

Session operations resolve their root on the host. Explicit host paths remain
available for the administrator's compatibility and artifact-opening flows.
"""
from __future__ import annotations

import asyncio
import re
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import deps
from ..config import settings
from ..services.uploads import save_upload
from ..workspace_files import (
    MAX_READ_BYTES,
    TEXT_PREVIEW_BYTES,
    WorkspacePathError,
    authorized_workspace_root,
    content_type_for,
    is_text_ext,
    list_entries,
    read_preview_text,
    require_relative_path,
    resolve_existing,
    resolve_write,
    search_workspace,
    stat_existing,
)

router = APIRouter()


def _raise(err: WorkspacePathError) -> None:
    raise HTTPException(status_code=err.status, detail=err.detail) from err


class StatRequest(BaseModel):
    root: str | None = None
    session_id: str | None = None
    binding: str | None = None
    paths: List[str] = Field(default_factory=list)


class WorkspaceDraft(BaseModel):
    session_id: str
    root: str


def _session_id(session_id: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session identity")
    return session_id


def _scope(request: Request, session_id: str, draft_root: str | None = None) -> dict:
    sid = _session_id(session_id)
    principal = request.state.principal
    service = deps.get_agent_service()
    current = service.session_workspace_dir(sid, principal.user_id)
    if current is None and draft_root is None:
        raise HTTPException(status_code=404, detail="Session workspace is not bound")
    try:
        root = str(authorized_workspace_root(
            current if current is not None else draft_root, settings.workspace_path,
            principal.user_id, principal.is_admin,
        ))
    except WorkspacePathError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err
    if current is None:
        service.set_session_work_dir(sid, root, principal.user_id)
    binding = sha256(root.encode()).hexdigest()[:24]
    return {"root": root, "session_id": sid, "binding": binding}


@router.get("/api/workspace/scope")
async def workspace_scope(request: Request, session_id: str):
    return _scope(request, session_id)


@router.post("/api/workspace/scope")
async def bind_draft_workspace(request: Request, body: WorkspaceDraft):
    return _scope(request, body.session_id, body.root)


def _root(request: Request, root: str | None, session_id: str | None, binding: str | None, *paths: str) -> str:
    if session_id is not None:
        if root is not None or not binding:
            raise HTTPException(status_code=400, detail="Provide a session binding instead of a root")
        scope = _scope(request, session_id)
        if scope["binding"] != binding:
            raise HTTPException(status_code=409, detail="Session workspace changed; reload it")
        for path in paths:
            require_relative_path(path)
        return scope["root"]
    if not request.state.principal.is_admin:
        raise HTTPException(status_code=403, detail="Explicit filesystem paths require administrator access")
    if root is None:
        raise HTTPException(status_code=422, detail="A session binding or root is required")
    return root


@router.get("/api/workspace/files")
async def workspace_list(
    request: Request,
    root: str | None = Query(None),
    path: str = Query(""),
    session_id: str | None = Query(None),
    binding: str | None = Query(None),
):
    try:
        root = _root(request, root, session_id, binding, path)
        entries = list_entries(root, path)
    except WorkspacePathError as e:
        _raise(e)
        return
    return {"path": path, "root": str(Path(root).expanduser().resolve()), "entries": entries}


@router.get("/api/workspace/search")
async def workspace_search(request: Request, session_id: str, binding: str, q: str = Query(min_length=1, max_length=160)):
    root = _root(request, None, session_id, binding)
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a filename")
    cancelled = Event()
    task = asyncio.create_task(asyncio.to_thread(search_workspace, root, query, cancelled))
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.1)
            if not task.done() and await request.is_disconnected():
                cancelled.set()
                break
        return await task
    except WorkspacePathError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err
    finally:
        cancelled.set()


@router.get("/api/workspace/content")
async def workspace_content(
    request: Request,
    root: str | None = Query(None),
    path: str = Query(...),
    download: int = Query(0),
    preview: int = Query(0),
    session_id: str | None = Query(None),
    binding: str | None = Query(None),
):
    try:
        root = _root(request, root, session_id, binding, path)
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
        try:
            text, truncated = read_preview_text(target)
        except WorkspacePathError as e:
            _raise(e)
            return
        return {"name": name, "content": text, "truncated": truncated, "limit": TEXT_PREVIEW_BYTES}
    size = target.stat().st_size
    if size > MAX_READ_BYTES and not download:
        raise HTTPException(status_code=413, detail="file too large to preview")
    return FileResponse(
        target,
        media_type=content_type_for(name, inline=not download),
        filename=name if download else None,
    )


@router.post("/api/workspace/stat")
async def workspace_stat(request: Request, body: StatRequest):
    if len(body.paths) > 200:
        raise HTTPException(status_code=400, detail="too many paths")
    try:
        root = _root(request, body.root, body.session_id, body.binding, *body.paths)
        existing = stat_existing(root, body.paths)
    except WorkspacePathError as e:
        _raise(e)
        return
    return {"existing": existing}


@router.post("/api/workspace/upload")
async def workspace_upload(
    request: Request,
    file: UploadFile = File(...),
    root: str | None = Form(None),
    path: str = Form(""),
    session_id: str | None = Form(None),
    binding: str | None = Form(None),
):
    """Write ``file`` into ``root`` / ``path`` / filename (path is the directory)."""
    rel_dir = path.strip().strip("/")
    name = Path(file.filename or "upload").name
    rel = f"{rel_dir}/{name}" if rel_dir else name
    try:
        root = _root(request, root, session_id, binding, path, rel)
        dest = resolve_write(root, rel)
    except WorkspacePathError as e:
        _raise(e)
        return
    total = await save_upload(file, dest, settings.upload_max_size_mb)
    return {"status": "ok", "path": rel, "filename": name, "size": total}
