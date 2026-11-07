# -*- coding: utf-8 -*-
"""Commit / commit+push a turn's changes from the review tab.

Runs git as the gateway process, outside the approval gate, so only an admin
gets it — same rule as the terminals.
"""
from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import deps
from ..git_ops import COMMIT_MESSAGE_PROMPT, GitError, commit, diff_for_message, recent_subjects, status

router = APIRouter()


class GitFiles(BaseModel):
    cwd: str
    files: List[str] = Field(default_factory=list)


class GitCommit(GitFiles):
    message: str
    push: bool = False


def _admin(request: Request) -> None:
    if not request.state.principal.is_admin:
        raise HTTPException(status_code=403, detail="Git actions are for the admin account")


def _raise(err: GitError) -> None:
    raise HTTPException(status_code=err.status, detail=err.detail) from err


@router.post("/api/git/status")
async def git_status(body: GitFiles, request: Request):
    _admin(request)
    try:
        return await asyncio.to_thread(status, body.cwd, body.files)
    except GitError as e:
        _raise(e)


@router.post("/api/git/message")
async def git_message(body: GitFiles, request: Request):
    """Draft a commit message for ``files`` on the side-task model
    (auxiliary when configured, else the main model)."""
    _admin(request)
    try:
        diff = await asyncio.to_thread(diff_for_message, body.cwd, body.files)
        history = await asyncio.to_thread(recent_subjects, body.cwd)
    except GitError as e:
        _raise(e)
        return
    if not diff.strip():
        raise HTTPException(status_code=409, detail="nothing to commit in the selected files")
    prompt = diff
    if history:
        prompt = "Recent commit subjects in this repository:\n" + "\n".join(history) + "\n\nDiff:\n" + diff
    try:
        text = await deps.get_agent_service().side_task(COMMIT_MESSAGE_PROMPT, prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    message = text.strip()
    if message.startswith("```"):
        message = message.split("\n", 1)[1] if "\n" in message else ""
        message = message.rsplit("```", 1)[0].strip()
    if not message:
        raise HTTPException(status_code=502, detail="LLM returned an empty message")
    return {"message": message}


@router.post("/api/git/commit")
async def git_commit(body: GitCommit, request: Request):
    _admin(request)
    try:
        return await asyncio.to_thread(commit, body.cwd, body.files, body.message, body.push)
    except GitError as e:
        _raise(e)
