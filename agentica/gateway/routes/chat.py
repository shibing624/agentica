# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Chat routes: /api/chat, /api/chat/stream, /api/sessions, /api/upload, /api/memory."""
import asyncio
import base64
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from agentica.memory.trace import last_completed_round

from .. import deps
from ..channels.base import InboundMedia
from ..config import settings
from ..models import (
    ChatImage, ChatRequest, ChatResponse, CompactRequest, MemoryRequest,
    RenameRequest, GoalRequest, SteerRequest, ApprovalDecisionRequest,
)
from ..services.agent_service import AgentService
from ..services import live_turn

try:
    from agentica.run_response import AgentCancelledError
except ImportError:
    AgentCancelledError = None

router = APIRouter()


def _sse_stream_hooks(publish):
    """Same content/tool/thinking events for chat and /goal."""

    def on_content(delta: str):
        publish({"event": "content", "data": delta})

    def on_tool_call(name: str, args: dict, tool_call_id: str = ""):
        publish({
            "event": "tool_call",
            "data": {"name": name, "args": args, "tool_call_id": tool_call_id},
        })

    def on_tool_result(name: str, result: str, extra: dict | None = None):
        data = {"name": name, "result": result}
        if extra:
            data.update(extra)
        publish({"event": "tool_result", "data": data})

    def on_thinking(delta: str):
        publish({"event": "thinking", "data": delta})

    return on_content, on_tool_call, on_tool_result, on_thinking


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _conflict_if_live(session_id: str, owner: str) -> None:
    if live_turn.active(session_id, owner) is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            ),
        )


_SSE_KEEPALIVE_S = 15.0


async def _sse_from_turn(turn: live_turn.LiveTurn, after: int = 0):
    """Yield SSE bytes. Client disconnect unsubscribes; the agent keeps running."""
    q = turn.subscribe(after=after)
    last_seq = after
    try:
        yield ": keepalive\n\n"
        for ev in turn.replay(after):
            last_seq = ev["seq"]
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        if not turn.done:
            turn.republish_pending_approvals()
        if turn.done:
            yield "data: [DONE]\n\n"
            return
        dropped = getattr(q, "_dropped", None)
        while True:
            get_task = asyncio.create_task(q.get())
            waiters = {get_task}
            drop_task = None
            if dropped is not None:
                drop_task = asyncio.create_task(dropped.wait())
                waiters.add(drop_task)
            done, pending = await asyncio.wait(
                waiters, timeout=_SSE_KEEPALIVE_S, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            if not done:
                yield ": keepalive\n\n"
                continue
            if get_task in done:
                item = get_task.result()
            else:
                return
            if item is None:
                yield "data: [DONE]\n\n"
                break
            if item["seq"] <= last_seq:
                continue
            last_seq = item["seq"]
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    except asyncio.CancelledError:
        return
    finally:
        turn.unsubscribe(q)


def _spawn_live(session_id: str, run_coro, owner: str, kind: str = "chat"):
    """Run ``run_coro(turn)`` in the background. Disconnect does not cancel it."""
    _conflict_if_live(session_id, owner)
    turn = live_turn.start(session_id, owner, kind)

    async def runner():
        try:
            await run_coro(turn)
            turn.finish("completed")
        except asyncio.CancelledError:
            turn.publish({"event": "aborted", "data": {}})
            turn.finish("cancelled")
        except Exception as e:
            if AgentCancelledError and isinstance(e, AgentCancelledError):
                turn.publish({"event": "aborted", "data": {}})
                turn.finish("cancelled")
            else:
                turn.publish({"event": "error", "data": str(e)})
                turn.finish("failed")

    turn.task = asyncio.create_task(runner())
    return turn

# Inline image payloads in JSON; keep well under typical provider inline caps.
_MAX_CHAT_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_CHAT_IMAGES = 8


def _account(request: Request) -> str:
    """The signed-in account, which is also the ``users/<id>/`` partition its
    sessions live in. Read from the credential, never from the payload."""
    return request.state.principal.user_id


def _images_to_media(images: list[ChatImage]) -> list[InboundMedia]:
    """Decode the web UI's base64 images into the same InboundMedia IM uses."""
    if len(images) > _MAX_CHAT_IMAGES:
        raise HTTPException(status_code=400, detail=f"At most {_MAX_CHAT_IMAGES} images per message")
    out: list[InboundMedia] = []
    for img in images:
        try:
            raw = base64.b64decode(img.data, validate=False)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image data")
        if not raw:
            continue
        if len(raw) > _MAX_CHAT_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds {_MAX_CHAT_IMAGE_BYTES // (1024 * 1024)}MB",
            )
        out.append(InboundMedia(kind="image", data=raw, mime=img.mime or "image/png"))
    return out


# ============== Non-streaming chat ==============

@router.post("/api/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Send a message to the agent (non-streaming)."""
    account = _account(request)
    if body.work_dir:
        await _apply_session_work_dir(svc, body.session_id, body.work_dir, account)
    svc.set_session_approval_mode(body.session_id, body.approval_mode, owner=account)

    result = await svc.chat(
        message=body.message,
        session_id=body.session_id,
        user_id=account,
        owner=account,
        media=_images_to_media(body.images) if body.images else None,
    )
    return ChatResponse(
        content=result.content,
        session_id=result.session_id,
        user_id=result.user_id,
        tool_calls=result.tool_calls,
    )


# ============== Mid-run interrupt (CLI steer) ==============

@router.post("/api/chat/steer")
async def steer_chat(
    body: SteerRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Inject guidance into the current run at the next tool-batch boundary.

    ``accepted: false`` is not an error — the run ended between the UI check
    and this call, and the client must queue a fresh turn instead of dropping
    the text (same TOCTOU contract as CLI ``agent.steer``).
    """
    accepted = svc.steer_session(body.session_id, body.message, owner=_account(request))
    return {"accepted": accepted}


@router.post("/api/chat/steer/take")
async def take_steer(
    body: SteerRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Pop steering that outlived the run so the web UI can queue it.

    ``message`` is ignored; the body reuses SteerRequest so the client always
    posts ``session_id``.
    """
    messages = svc.take_undelivered_steer(body.session_id, owner=_account(request))
    return {"messages": messages}


# ============== Standing goal ("/goal <objective>") ==============

@router.post("/api/goal")
async def run_goal(
    body: GoalRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Drive a standing-goal loop for the web UI's ``/goal``.

    Streams ``status`` events (token progress), the same ``content`` /
    ``thinking`` / ``tool_call`` / ``tool_result`` events as ``/api/chat/stream``,
    then a final ``done``. ``token_budget`` of ``-1`` (default) is unlimited.
    """
    session_id = body.session_id
    account = _account(request)

    async def run_goal_turn(turn: live_turn.LiveTurn):
        on_content, on_tool_call, on_tool_result, on_thinking = _sse_stream_hooks(turn.publish)

        def on_event(data: dict):
            turn.publish({"event": "status", "data": data})

        result = await svc.run_goal(
            body.objective, session_id, user_id=account, owner=account,
            token_budget=body.token_budget,
            on_event=on_event,
            on_content=on_content,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_thinking=on_thinking,
        )
        turn.publish({"event": "done", "data": result})

    _touch_session_log(svc, session_id, account)
    turn = _spawn_live(session_id, run_goal_turn, owner=account, kind="goal")
    return StreamingResponse(
        _sse_from_turn(turn),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# ============== SSE streaming chat ==============

async def _start_chat_run(body: ChatRequest, request: Request, svc: AgentService) -> live_turn.LiveTurn:
    account = _account(request)
    session_id = body.session_id
    if body.work_dir:
        await _apply_session_work_dir(svc, session_id, body.work_dir, account)
    svc.set_session_approval_mode(session_id, body.approval_mode, owner=account)
    _touch_session_log(svc, session_id, account)

    async def run_chat_turn(turn: live_turn.LiveTurn):
        on_content, on_tool_call, on_tool_result, on_thinking = _sse_stream_hooks(turn.publish)
        t0 = time.time()
        result = await svc.chat_stream(
            message=body.message,
            session_id=session_id,
            user_id=account,
            owner=account,
            on_content=on_content,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_thinking=on_thinking,
            media=_images_to_media(body.images) if body.images else None,
        )
        elapsed = round(time.time() - t0, 2)

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
            ctx_window = deps.agent_service.get_context_window(session_id, owner=account)

        done = {
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
        }
        if result.usage:
            done["usage"] = result.usage
            if result.usage.get("window"):
                done["context_window"] = result.usage["window"]
        if result.turn_usage:
            done["turn_usage"] = result.turn_usage
            done["cache_read_tokens"] = result.turn_usage["cache_read_tokens"]
            done["cache_hit_percent"] = result.turn_usage["cache_hit_percent"]
            if result.turn_usage.get("cost_usd") is not None:
                done["cost_usd"] = result.turn_usage["cost_usd"]
        log = svc.session_log_for(session_id, owner=account)
        if log.path.exists():
            rd = last_completed_round(log.iter_raw_entries())
            if rd is not None:
                done["duration_ms"] = rd["durationMs"]
                done["llm_ms"] = rd["llmMs"]
                done["tps"] = rd["tps"]
        if result.media_notes:
            done["media_notes"] = result.media_notes
        turn.publish({"event": "done", "data": done})

    return _spawn_live(session_id, run_chat_turn, owner=account, kind="chat")


@router.post("/api/chat/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Create a background run and subscribe to it (one-shot convenience)."""
    turn = await _start_chat_run(body, request, svc)
    return StreamingResponse(
        _sse_from_turn(turn),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/api/chat/runs")
async def create_chat_run(
    body: ChatRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Start a background run and return ``run_id`` immediately."""
    turn = await _start_chat_run(body, request, svc)
    return turn.public()


@router.get("/api/chat/runs/active")
async def active_chat_run(
    request: Request,
    session_id: str,
):
    """The in-flight run for this session, if any (refresh / other tab)."""
    turn = live_turn.active(session_id, owner=_account(request))
    if turn is None:
        return {"run": None}
    return {"run": turn.public()}


@router.get("/api/chat/runs/{run_id}/events")
async def chat_run_events(
    run_id: str,
    request: Request,
    after: int = 0,
):
    """Subscribe or reconnect. Disconnect does not cancel the run."""
    turn = live_turn.get_run(run_id)
    if turn is None or not live_turn.owned(turn, _account(request)):
        raise HTTPException(status_code=404, detail="Run not found")
    return StreamingResponse(
        _sse_from_turn(turn, after=after),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/api/chat/runs/{run_id}/cancel")
async def cancel_chat_run(
    run_id: str,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Stop this run and wait until the session lock is free. Idempotent."""
    try:
        return await live_turn.cancel_and_wait(svc, run_id=run_id, owner=_account(request))
    except PermissionError:
        raise HTTPException(status_code=404, detail="Run not found")


@router.get("/api/chat/stream/{session_id}")
async def attach_chat_stream(
    session_id: str,
    request: Request,
    after: int = 0,
):
    """Reattach by session id (replay ``seq > after``, then live)."""
    turn = live_turn.active(session_id, owner=_account(request))
    if turn is None:
        async def empty():
            yield ": keepalive\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream", headers=_SSE_HEADERS)
    return StreamingResponse(
        _sse_from_turn(turn, after=after),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/api/chat/cancel")
async def cancel_chat(
    body: SteerRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Stop the session's in-flight run (alias of ``POST /api/chat/runs/{id}/cancel``)."""
    try:
        return await live_turn.cancel_and_wait(
            svc, session_id=body.session_id, owner=_account(request),
        )
    except PermissionError:
        raise HTTPException(status_code=404, detail="Run not found")


# ============== Sessions ==============

@router.get("/api/sessions")
async def list_sessions(
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    return {"sessions": svc.list_sessions(owner=_account(request))}


@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    success = svc.delete_session(session_id, owner=_account(request))
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}


@router.post("/api/sessions/{session_id}/rename")
async def rename_session(
    session_id: str,
    body: RenameRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    svc.rename_session(session_id, name, owner=_account(request))
    return {"status": "renamed", "session_id": session_id, "name": name}


@router.post("/api/sessions/{session_id}/archive")
async def archive_session(
    session_id: str,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    svc.archive_session(session_id, archived=True, owner=_account(request))
    return {"status": "archived", "session_id": session_id}


@router.post("/api/sessions/{session_id}/unarchive")
async def unarchive_session(
    session_id: str,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    svc.archive_session(session_id, archived=False, owner=_account(request))
    return {"status": "unarchived", "session_id": session_id}


@router.get("/api/sessions/{session_id}/usage")
async def session_usage(
    session_id: str,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Session-level context occupancy and billing, same shape as CLI ``/usage``."""
    return await svc.session_usage(session_id, owner=_account(request))


@router.post("/api/sessions/{session_id}/approvals/{tool_call_id}")
async def decide_session_approval(
    session_id: str,
    tool_call_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
):
    """Resolve a parked tool call. Unknown id or another account's turn → 404."""
    owner = _account(request)
    turn = live_turn.active(session_id, owner)
    if turn is None or not turn.approvals.decide(tool_call_id, body.decision):
        raise HTTPException(status_code=404, detail="Approval not found")
    return {"status": "ok", "tool_call_id": tool_call_id, "decision": body.decision}


@router.post("/api/sessions/{session_id}/compact")
async def compact_session(
    session_id: str,
    request: Request,
    body: Optional[CompactRequest] = None,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Web ``/compact``: summarise this session the same way the CLI does."""
    try:
        result = await svc.compact_session(
            session_id,
            owner=_account(request),
            instructions=(body.instructions if body else ""),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Compaction failed")
    return result


# ============== Memory (user-level AGENTS.md) ==============

@router.get("/api/memory")
async def get_memory(
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """This account's ``users/<id>/AGENTS.md`` — standing rules in the system prompt."""
    return await svc.read_user_agents_md(_account(request))


@router.put("/api/memory")
async def put_memory(
    body: MemoryRequest,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    try:
        return await svc.write_user_agents_md(_account(request), body.content)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


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


async def _apply_session_work_dir(
    svc: AgentService, session_id: str, work_dir: str, owner: str,
) -> None:
    """Set per-session work_dir.  Acquires lock to avoid concurrent races."""
    p = Path(work_dir).expanduser()
    if not p.is_dir():
        return
    async with _work_dir_lock:
        current = svc.get_session_work_dir(session_id, owner)
        if str(p) == current:
            return
        svc.set_session_work_dir(session_id, str(p), owner)


def _touch_session_log(svc: AgentService, session_id: str, owner: str) -> None:
    """Create an empty jsonl so the sidebar can list a run before the first append."""
    log = svc.session_log_for(session_id, owner=owner)
    path = log.path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
