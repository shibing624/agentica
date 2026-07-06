# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Chat routes: /api/chat, /api/chat/stream, /api/sessions, /api/upload, /api/memory."""
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from .. import deps
from ..config import settings
from ..models import ChatRequest, ChatResponse, MemoryRequest, RenameRequest, GoalRequest
from ..services.agent_service import AgentService

try:
    from agentica.run_response import AgentCancelledError
except ImportError:
    AgentCancelledError = None

router = APIRouter()


# ============== Non-streaming chat ==============

@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Send a message to the agent (non-streaming)."""
    if request.work_dir:
        await _apply_session_work_dir(svc, request.session_id, request.work_dir)
    svc.set_session_approval_mode(request.session_id, request.approval_mode)

    result = await svc.chat(
        message=request.message,
        session_id=request.session_id,
        user_id=request.user_id,
    )
    return ChatResponse(
        content=result.content,
        session_id=result.session_id,
        user_id=result.user_id,
        tool_calls=result.tool_calls,
    )


# ============== Standing goal ("/goal <objective>") ==============

@router.post("/api/goal")
async def run_goal(
    request: GoalRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Drive a bounded standing-goal loop (Agent.run_goal) for the web UI's
    "/goal <objective>" command. Non-streaming — the loop runs several turns
    internally and returns only the final result."""
    try:
        result = await svc.run_goal(request.objective, request.session_id, request.user_id)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return result


# ============== SSE streaming chat ==============

@router.post("/api/chat/stream")
async def chat_stream(
    request: ChatRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Send a message and stream the response via Server-Sent Events."""
    if request.work_dir:
        await _apply_session_work_dir(svc, request.session_id, request.work_dir)
    svc.set_session_approval_mode(request.session_id, request.approval_mode)

    session_id = request.session_id

    async def event_generator():
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def on_content(delta: str):
            await queue.put({"event": "content", "data": delta})

        async def on_tool_call(name: str, args: dict):
            await queue.put({"event": "tool_call", "data": {"name": name, "args": args}})

        async def on_tool_result(name: str, result: str):
            await queue.put({"event": "tool_result", "data": {"name": name, "result": result}})

        async def on_thinking(delta: str):
            await queue.put({"event": "thinking", "data": delta})

        async def run_agent():
            t0 = time.time()
            try:
                result = await svc.chat_stream(
                    message=request.message,
                    session_id=session_id,
                    user_id=request.user_id,
                    on_content=on_content,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                    on_thinking=on_thinking,
                )
                elapsed = round(time.time() - t0, 2)

                # Build token usage from metrics
                raw_metrics = result.metrics or {}

                def _sum(key):
                    v = raw_metrics.get(key, 0)
                    if isinstance(v, list):
                        return sum(x for x in v if isinstance(x, (int, float)))
                    return v if isinstance(v, (int, float)) else 0

                def _list(key):
                    v = raw_metrics.get(key, [])
                    if isinstance(v, list):
                        return [x for x in v if isinstance(x, (int, float))]
                    return [v] if isinstance(v, (int, float)) else []

                input_tokens = _sum("input_tokens")
                output_tokens = _sum("output_tokens")
                total_tokens = _sum("total_tokens")

                in_list = _list("input_tokens")
                out_list = _list("output_tokens")
                tot_list = _list("total_tokens")
                time_list = _list("time")
                n_requests = max(len(in_list), len(out_list), 1)
                request_entries = []
                for i in range(n_requests):
                    entry = {
                        "request_index": i + 1,
                        "input_tokens": in_list[i] if i < len(in_list) else 0,
                        "output_tokens": out_list[i] if i < len(out_list) else 0,
                        "total_tokens": tot_list[i] if i < len(tot_list) else 0,
                    }
                    if i < len(time_list):
                        entry["response_time"] = round(time_list[i], 3)
                    request_entries.append(entry)

                ctx_window = 128000
                if deps.agent_service:
                    ctx_window = deps.agent_service.get_context_window(session_id)

                await queue.put({"event": "done", "data": {
                    "session_id": result.session_id,
                    "tool_calls": result.tool_calls,
                    "tools_used": result.tools_used,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "requests": n_requests,
                    "response_time": elapsed,
                    "request_entries": request_entries,
                    "context_window": ctx_window,
                }})

            except asyncio.CancelledError:
                pass
            except Exception as e:
                if AgentCancelledError and isinstance(e, AgentCancelledError):
                    pass
                else:
                    await queue.put({"event": "error", "data": str(e)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_agent())

        try:
            while True:
                item = await queue.get()
                if item is None:
                    yield "data: [DONE]\n\n"
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            # User disconnected — cancel the specific session's agent
            svc.cancel_session(session_id)
            task.cancel()
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============== Sessions ==============

@router.get("/api/sessions")
async def list_sessions(svc: AgentService = Depends(deps.get_agent_service)):
    return {"sessions": svc.list_sessions()}


@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    svc: AgentService = Depends(deps.get_agent_service),
):
    success = svc.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.post("/api/sessions/{session_id}/rename")
async def rename_session(
    session_id: str,
    request: RenameRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    svc.rename_session(session_id, name)
    return {"status": "renamed", "session_id": session_id, "name": name}


@router.post("/api/sessions/{session_id}/archive")
async def archive_session(
    session_id: str,
    svc: AgentService = Depends(deps.get_agent_service),
):
    svc.archive_session(session_id, archived=True)
    return {"status": "archived", "session_id": session_id}


@router.post("/api/sessions/{session_id}/unarchive")
async def unarchive_session(
    session_id: str,
    svc: AgentService = Depends(deps.get_agent_service),
):
    svc.archive_session(session_id, archived=False)
    return {"status": "unarchived", "session_id": session_id}


# ============== Memory ==============

@router.post("/api/memory")
async def save_memory(
    request: MemoryRequest,
    svc: AgentService = Depends(deps.get_agent_service),
):
    await svc.save_memory(request.content, user_id=request.user_id, long_term=request.long_term)
    return {"status": "saved", "user_id": request.user_id}


# ============== File upload ==============

@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    target_dir: str = Form(""),
):
    """Upload a file to the working directory.

    Enforces size limit and extension whitelist from settings.
    """
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    allowed = settings.upload_allowed_ext_set
    if allowed and ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(allowed))}",
        )

    # Read in chunks and enforce the size limit while streaming, so an oversized
    # upload is rejected mid-read instead of being fully buffered into memory
    # first (a full read() lets a client OOM the server regardless of the limit).
    max_bytes = settings.upload_max_size_mb * 1024 * 1024
    chunks = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds size limit of {settings.upload_max_size_mb}MB",
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    # Write to destination — enforce that files land inside workspace
    workspace = settings.workspace_path.resolve()
    if target_dir:
        base = Path(target_dir).resolve()
        if not base.is_relative_to(workspace):
            raise HTTPException(
                status_code=400,
                detail="target_dir must be within the workspace directory",
            )
    else:
        base = workspace
    base.mkdir(parents=True, exist_ok=True)
    dest = base / Path(file.filename or "upload").name

    try:
        dest.write_bytes(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file: {e}")

    return {"status": "ok", "path": str(dest), "filename": file.filename, "size": len(content)}


# ============== Helpers ==============

_work_dir_lock = asyncio.Lock()


async def _apply_session_work_dir(svc: AgentService, session_id: str, work_dir: str) -> None:
    """Set per-session work_dir.  Acquires lock to avoid concurrent races."""
    p = Path(work_dir).expanduser()
    if not p.is_dir():
        return
    async with _work_dir_lock:
        current = svc.get_session_work_dir(session_id)
        if str(p) == current:
            return
        svc.set_session_work_dir(session_id, str(p))
