# -*- coding: utf-8 -*-
"""Worktree outlet for Web / Desktop / IM sessions.

The model gets the same ``worktree`` tool as the CLI. This route is the human
outlet: list, create, bind, merge and remove. ``bind`` is the only operation
that changes a session's working directory, so it also invalidates that
session's cached agent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agentica.git.binding import WorktreeBinder
from agentica.git import worktree as worktrees
from agentica.git.worktree import WorktreeError

from .. import deps
from ..services import live_turn
from ..workspace_files import WorkspacePathError, authorized_workspace_root

router = APIRouter(prefix="/api/worktrees")


def _account(request: Request) -> str:
    return request.state.principal.user_id


def _binder(request: Request, session_id: str) -> tuple[WorktreeBinder, object, str]:
    service = deps.get_agent_service()
    owner = _account(request)
    work_dir = service.get_session_work_dir(session_id, owner)
    if not request.state.principal.is_admin:
        try:
            from ..config import settings

            authorized_workspace_root(work_dir, settings.workspace_path, owner, False)
        except WorkspacePathError as err:
            raise HTTPException(status_code=err.status, detail=err.detail) from err
    binder = WorktreeBinder(agent_config={"work_dir": work_dir})
    return binder, service, owner


def _entry_dict(entry) -> dict:
    return {
        "name": entry.name,
        "path": entry.path,
        "branch": entry.branch_short,
        "head": entry.head,
        "is_main": entry.is_main,
        "exists": entry.exists,
        "locked": entry.locked,
        "lock_reason": entry.lock_reason,
        "describe": entry.describe(),
    }


class WorktreeRequest(BaseModel):
    session_id: str
    action: str = Field(default="status")
    name: str = ""
    base: str = ""


@router.get("")
async def list_worktrees(session_id: str, request: Request):
    binder, _service, _owner = _binder(request, session_id)
    try:
        return {
            "work_dir": binder.work_dir(),
            "worktrees": [_entry_dict(entry) for entry in worktrees.list_worktrees(binder.work_dir())],
            "status": binder.status(),
        }
    except WorktreeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("")
async def worktree_action(body: WorktreeRequest, request: Request):
    binder, service, owner = _binder(request, body.session_id)
    action = body.action.strip().casefold()
    name = body.name.strip()
    base = body.base.strip() or None
    try:
        if action == "status":
            output = binder.status()
        elif action in {"new", "create"}:
            if not name:
                raise WorktreeError("a worktree name is required")
            output = binder.create(name, base=base)
        elif action == "bind":
            if not name:
                raise WorktreeError("a worktree name is required")
            if live_turn.active(body.session_id, owner) or service.is_session_active(body.session_id, owner):
                raise HTTPException(status_code=409, detail="Cannot change the workspace while this session is running")
            entry = worktrees.find(binder.work_dir(), name)
            if entry is None:
                raise WorktreeError(f"no worktree named {name!r}")
            path = Path(entry.path).expanduser().resolve()
            if not entry.exists or not path.is_dir():
                raise WorktreeError(f"{entry.path} is not a usable checkout")
            service.set_session_work_dir(body.session_id, str(path), owner)
            service.invalidate_session_agent(body.session_id, owner)
            output = f"Bound session to {entry.name} at {path}."
        elif action == "merge":
            if not name:
                raise WorktreeError("a worktree name is required")
            output = binder.merge(name, base=base)
        elif action in {"remove", "delete"}:
            if not name:
                raise WorktreeError("a worktree name is required")
            output = binder.remove(name)
        else:
            raise WorktreeError(f"unknown worktree action {action!r}")
        return {"ok": True, "output": output, "work_dir": service.get_session_work_dir(body.session_id, owner)}
    except WorktreeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e