# -*- coding: utf-8 -*-
"""Scheduler routes: /api/scheduler/* — thin HTTP wrapper over the SDK cron
module (agentica.cron.jobs) and the cronjob tool.

Pydantic bodies for create/update; run history list for execution audit.
The gateway is a single-user local tool, so user_id is fixed to "default"
and not used as an authorization boundary.
"""
import json
from typing import Optional

from fastapi import APIRouter, HTTPException

from agentica.cron.jobs import (
    list_jobs,
    get_job,
    remove_job,
    pause_job,
    resume_job,
    list_task_runs,
    schedule_to_human,
)
from agentica.cron.scheduler import _execute_job
from agentica.tools.cron_tool import cronjob

from .. import deps
from ..config import settings
from ..models import CronJobCreateRequest, CronJobUpdateRequest, PolishPromptRequest

router = APIRouter(prefix="/api/scheduler")


def _job_dict(j) -> dict:
    """Full job representation for list/get responses (key: 'id')."""
    return {
        "id": j.id,
        "name": j.name,
        "prompt": j.prompt,
        "user_id": j.user_id,
        "schedule": schedule_to_human(j.schedule),
        "status": j.status.value,
        "enabled": j.enabled,
        "deliver": j.deliver,
        "next_run_at_ms": j.next_run_at_ms,
        "last_run_at_ms": j.last_run_at_ms,
        "last_status": j.last_status,
        "run_count": j.run_count,
        "timeout_seconds": j.timeout_seconds,
        "max_retries": j.max_retries,
        "retry_count": j.retry_count,
        "retry_delay_ms": j.retry_delay_ms,
        "permissions": j.permissions,
    }


def _run_dict(r) -> dict:
    """Run record for history responses.

    ``result_preview`` mirrors cron_tool._action_runs (200 chars — that one
    is LLM-facing and must stay context-cheap). ``result_full`` is this
    endpoint's own addition for the web UI's hover tooltip: this route is
    browser-only, so it can afford a much larger (but still capped) payload.
    """
    full = r.result or r.error or ""
    return {
        "job_id": r.task_id,
        "status": r.status.value if hasattr(r.status, "value") else str(r.status),
        "started_at_ms": r.started_at_ms,
        "ended_at_ms": r.ended_at_ms,
        "result_preview": (r.result[:200] if r.result else None),
        "result_full": full[:20000],
        "error": r.error,
    }


def _parse_tool_result(result_str: str) -> dict:
    try:
        return json.loads(result_str)
    except Exception:
        return {"success": False, "error": result_str}


def _job_response(result: dict) -> dict:
    """Normalize a cronjob tool result so the embedded job uses the same _job_dict
    shape (key 'id') as list/get, instead of _format_job's 'job_id' key.
    """
    raw = result.get("job")
    if raw and "job_id" in raw:
        j = get_job(raw["job_id"])
        if j:
            result["job"] = _job_dict(j)
    return result


# ============== Job CRUD ==============

@router.get("/jobs")
async def api_list_jobs(include_disabled: bool = True, limit: int = 100):
    jobs = list_jobs(include_disabled=include_disabled, limit=limit)
    return {"jobs": [_job_dict(j) for j in jobs], "total": len(jobs)}


@router.get("/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": _job_dict(job)}


@router.post("/jobs")
async def api_create_job(body: CronJobCreateRequest):
    fields = body.model_dump(exclude={"validate_run"})
    result = _parse_tool_result(cronjob(action="create", **fields))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create job"))
    result = _job_response(result)
    if body.validate_run:
        job = result.get("job") or {}
        job_id = job.get("id")
        job_obj = get_job(job_id) if job_id else None
        if job_obj:
            result["validation_run"] = await _execute_job(job_obj, agent_runner=deps.cron_runner, verbose=False)
            result["job"] = _job_dict(get_job(job_id))
    return result


@router.put("/jobs/{job_id}")
async def api_update_job(job_id: str, body: CronJobUpdateRequest):
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    result = _parse_tool_result(cronjob(action="update", job_id=job_id, **updates))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Update failed"))
    return _job_response(result)


@router.delete("/jobs/{job_id}")
async def api_delete_job(job_id: str):
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    if not remove_job(job_id):
        raise HTTPException(status_code=400, detail="Delete failed")
    return {"status": "deleted", "job_id": job_id}


# ============== Job actions ==============

@router.post("/jobs/{job_id}/pause")
async def api_pause_job(job_id: str):
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    if not pause_job(job_id):
        raise HTTPException(status_code=400, detail="Pause failed")
    return {"status": "paused", "job_id": job_id}


@router.post("/jobs/{job_id}/resume")
async def api_resume_job(job_id: str):
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    updated = resume_job(job_id)
    if not updated:
        raise HTTPException(status_code=400, detail="Resume failed")
    return {"status": "resumed", "job_id": job_id, "next_run_at_ms": updated.next_run_at_ms}


@router.post("/jobs/{job_id}/trigger")
async def api_trigger_job(job_id: str):
    """Run this job once, right now, synchronously — same execution path
    (`_execute_job` + the gateway's AgentRunner) the background ticker uses,
    not just marking it due for the next tick.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    run_result = await _execute_job(job, agent_runner=deps.cron_runner, verbose=False)
    updated = get_job(job_id)
    return {
        "success": run_result.get("status") == "ok",
        "job": _job_dict(updated) if updated else None,
        "run": run_result,
    }


# ============== Prompt polishing (LLM-assisted authoring) ==============

_POLISH_SYSTEM_PROMPT = """You rewrite a rough cron-job idea into a clear, \
self-contained prompt for an unattended AI agent. The agent that receives \
this prompt runs in a fresh session with no chat history and no user present \
to answer follow-up questions, so the prompt must:
- State the exact task and the expected deliverable/output format.
- Include any concrete details the user gave (paths, names, thresholds, etc).
- Avoid vague language like "check things" — be specific and actionable.
Reply with ONLY the rewritten prompt text, no preamble, no quotes, no markdown fences."""


@router.post("/polish_prompt")
async def polish_prompt(body: PolishPromptRequest):
    draft = body.draft.strip()
    if not draft:
        raise HTTPException(status_code=400, detail="draft must not be empty")

    from agentica.model.message import Message
    from ..services.model_factory import create_model

    model = create_model(
        settings.model_provider,
        settings.model_name,
        base_url=settings.model_base_url or None,
        api_key=settings.model_api_key or None,
    )
    messages = [
        Message(role="system", content=_POLISH_SYSTEM_PROMPT),
        Message(role="user", content=draft),
    ]
    try:
        response = await model.response(messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    polished = (response.content or "").strip() if response else ""
    if not polished:
        raise HTTPException(status_code=502, detail="LLM returned an empty response")
    return {"prompt": polished}


# ============== Run history ==============

@router.get("/runs")
async def api_list_runs(limit: int = 50):
    runs = list_task_runs(job_id=None, limit=limit)
    return {"runs": [_run_dict(r) for r in runs], "total": len(runs)}


@router.get("/jobs/{job_id}/runs")
async def api_list_job_runs(job_id: str, limit: int = 50):
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    runs = list_task_runs(job_id=job_id, limit=limit)
    return {"job_id": job_id, "runs": [_run_dict(r) for r in runs], "total": len(runs)}
