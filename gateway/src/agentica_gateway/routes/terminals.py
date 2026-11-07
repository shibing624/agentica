# -*- coding: utf-8 -*-
"""Right-panel terminals: create / list / close over HTTP, the stream over a
WebSocket per terminal.

A terminal is a shell with the gateway's own OS rights, outside every approval
gate, so only an admin gets one.
"""
from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .. import accounts
from ..auth import auth_enabled, websocket_account, websocket_token_ok
from ..terminals import TerminalError, manager

router = APIRouter()


class CreateTerminal(BaseModel):
    cwd: str
    session_id: str = ""
    cols: int = 80
    rows: int = 24


def _admin(request: Request) -> str:
    principal = request.state.principal
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Terminals are for the admin account")
    return principal.user_id


def _raise(err: TerminalError) -> None:
    raise HTTPException(status_code=err.status, detail=err.detail) from err


@router.get("/api/terminals")
async def list_terminals(request: Request):
    return {"terminals": manager.list(_admin(request))}


@router.post("/api/terminals")
async def create_terminal(request: Request, body: CreateTerminal):
    owner = _admin(request)
    try:
        term = manager.create(owner, body.session_id, body.cwd, body.cols, body.rows)
    except TerminalError as e:
        _raise(e)
        return
    return term.info()


@router.delete("/api/terminals/{term_id}")
async def close_terminal(request: Request, term_id: str):
    try:
        manager.close(_admin(request), term_id)
    except TerminalError as e:
        _raise(e)
    return {"ok": True}


def _same_origin(websocket: WebSocket) -> bool:
    """A page on another site may open a socket here; the browser sends its
    Origin, which the page cannot choose."""
    origin = websocket.headers.get("origin")
    if not origin:
        return True
    return urlsplit(origin).netloc == websocket.headers.get("host", "")


@router.websocket("/api/terminals/{term_id}/stream")
async def terminal_stream(websocket: WebSocket, term_id: str):
    if not _same_origin(websocket) or not websocket_token_ok(websocket):
        await websocket.close(code=4401)
        return
    owner = websocket_account(websocket)
    if auth_enabled() and not accounts.store().is_admin(owner):
        await websocket.close(code=4403)
        return
    try:
        term = manager.get(owner, term_id)
    except TerminalError:
        await websocket.close(code=4404)
        return
    await websocket.accept()

    outbox: asyncio.Queue = asyncio.Queue()
    listener = outbox.put_nowait
    replay = manager.attach(term, listener)
    await websocket.send_json({"type": "snapshot", "data": replay.decode("utf-8", errors="replace"),
                               "info": term.info()})
    if term.exit_code is not None:
        await websocket.send_json({"type": "exit", "code": term.exit_code})

    async def pump() -> None:
        while True:
            event = await outbox.get()
            await websocket.send_json(event)
            if event["type"] == "closed":
                await websocket.close()
                return

    sender = asyncio.create_task(pump())
    try:
        while True:
            message = await websocket.receive_json()
            kind = message.get("type")
            if kind == "input":
                manager.write(term, str(message.get("data", "")))
            elif kind == "resize":
                manager.resize(term, int(message.get("cols", term.cols)), int(message.get("rows", term.rows)))
    except WebSocketDisconnect:
        pass
    finally:
        manager.detach(term, listener)
        sender.cancel()
