# -*- coding: utf-8 -*-
"""Read-only session log endpoints: paginated events, whole-file analysis, raw export."""
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from agentica.memory.trace import call_body, search_calls

from .. import deps
from ..services.agent_service import AgentService

router = APIRouter()


@router.get("/api/sessions/{session_id}/trace/events")
async def trace_events(
    session_id: str,
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
    svc: AgentService = Depends(deps.get_agent_service),
):
    log = svc.session_log_for(session_id, owner=request.state.principal.user_id)
    if not log.path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    entries = list(log.iter_raw_entries())
    window = entries[offset:offset + limit]
    return {
        "session_id": session_id,
        "offset": offset,
        "limit": limit,
        "total": len(entries),
        "events": window,
    }


@router.get("/api/sessions/{session_id}/trace/analysis")
async def trace_analysis(
    session_id: str,
    request: Request,
    lite: bool = Query(False),
    svc: AgentService = Depends(deps.get_agent_service),
):
    """``lite``: no bodies — the page loads each one with ``/trace/calls/{index}``."""
    log = svc.session_log_for(session_id, owner=request.state.principal.user_id)
    if not log.path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return await asyncio.to_thread(log.analyze, lite=lite)


@router.get("/api/sessions/{session_id}/trace/calls/{index}")
async def trace_call_body(
    session_id: str,
    index: int,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Arguments, result, thinking or reply of the call at log line ``index``."""
    log = svc.session_log_for(session_id, owner=request.state.principal.user_id)
    if not log.path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    body = await asyncio.to_thread(call_body, log.iter_raw_entries(), index)
    if body is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return body


@router.get("/api/sessions/{session_id}/trace/search")
async def trace_search(
    session_id: str,
    request: Request,
    q: str = Query("", max_length=200),
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Calls whose summary or body contains ``q``, as log-line indices."""
    log = svc.session_log_for(session_id, owner=request.state.principal.user_id)
    if not log.path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return {"indices": await asyncio.to_thread(search_calls, log.iter_raw_entries(), q)}


@router.get("/api/sessions/{session_id}/export")
async def export_session(
    session_id: str,
    request: Request,
    svc: AgentService = Depends(deps.get_agent_service),
):
    """Download the session log as-is: the same JSONL the CLI's ``/export``
    copies out, which ``agentica --resume`` and the Trace page both read."""
    log = svc.session_log_for(session_id, owner=request.state.principal.user_id)
    if not log.path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return FileResponse(
        log.path,
        media_type="application/x-ndjson",
        filename=f"{session_id}.jsonl",
    )
