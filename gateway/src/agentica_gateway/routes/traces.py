# -*- coding: utf-8 -*-
"""Read-only Trace endpoints: paginated events + whole-file analysis."""
from fastapi import APIRouter, Depends, HTTPException, Query, Request

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
    svc: AgentService = Depends(deps.get_agent_service),
):
    log = svc.session_log_for(session_id, owner=request.state.principal.user_id)
    if not log.path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    analysis = log.analyze()
    return analysis
