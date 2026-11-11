# -*- coding: utf-8 -*-
"""Scheduler routes: /api/scheduler/* — thin HTTP wrapper over the SDK cron
module (agentica.cron.jobs), the cronjob tool, and the gateway's scheduler
service (runs are delivered into conversations).

Jobs are per signed-in account. ``CronJob.user_id`` is stamped from the session
cookie on create, never from the body — a field naming somebody else's jobs
would be a way in. The background ticker still runs every account's due jobs;
this surface only lists and edits the caller's.
"""
import json

from fastapi import APIRouter, HTTPException, Request

from agentica.cron.jobs import (
    AtSchedule,
    CronSchedule,
    EverySchedule,
    JobStatus,
    get_job,
    list_jobs,
    list_task_runs,
    pause_job,
    remove_job,
    resume_job,
    schedule_to_expr,
    schedule_to_human,
    update_job,
)
from agentica.cron.scheduler import is_job_running
from agentica.tools.cron_tool import _do_cronjob

from .. import deps
from ..models import (
    CronJobCreateRequest, CronJobUpdateRequest, PolishPromptRequest, SchedulerToggleRequest,
)

router = APIRouter(prefix="/api/scheduler")


def _account(request: Request) -> str:
    return request.state.principal.user_id


def _owned_job(request: Request, job_id: str):
    """The caller's job, or 404. 404 rather than 403 so the id of someone
    else's job is not confirmed by a different status code."""
    job = get_job(job_id)
    if job is None or job.user_id != _account(request):
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _schedule_spec(schedule) -> dict:
    """The schedule as the editor needs it: its kind and the exact value.

    The human text (``schedule``) does not parse back and the round-trip
    string (``schedule_expr``) loses which picker built it; this is what the
    Automations page reads its form from.
    """
    if isinstance(schedule, CronSchedule):
        return {"kind": "cron", "expression": schedule.expression, "timezone": schedule.timezone}
    if isinstance(schedule, EverySchedule):
        return {"kind": "every", "seconds": schedule.interval_ms // 1000}
    if isinstance(schedule, AtSchedule):
        return {"kind": "at", "at_ms": schedule.at_ms}
    return {"kind": "unknown"}


def _job_dict(j) -> dict:
    """Full job representation for list/get responses (key: 'id')."""
    return {
        "id": j.id,
        "name": j.name,
        "prompt": j.prompt,
        "user_id": j.user_id,
        "schedule": schedule_to_human(j.schedule),
        # What the edit form must send back: the human form does not parse.
        "schedule_expr": schedule_to_expr(j.schedule),
        "schedule_spec": _schedule_spec(j.schedule),
        "timezone": j.timezone,
        "status": j.status.value,
        "enabled": j.enabled,
        "deliver": j.deliver,
        "created_at_ms": j.created_at_ms,
        "next_run_at_ms": j.next_run_at_ms,
        "last_run_at_ms": j.last_run_at_ms,
        "last_status": j.last_status,
        "run_count": j.run_count,
        "timeout_seconds": j.timeout_seconds,
        "max_retries": j.max_retries,
        "retry_count": j.retry_count,
        "retry_delay_ms": j.retry_delay_ms,
        "permission_mode": j.permission_mode,
        "work_dir": j.work_dir,
        "paused_reason": j.paused_reason,
        "target": j.target,
        "session_id": j.session_id,
        "running": deps.scheduler.is_running(j.id) if deps.scheduler else is_job_running(j.id),
    }


def _run_dict(r) -> dict:
    """Run record for history responses.

    ``result_preview`` mirrors cron_tool._action_runs (200 chars — that one
    is LLM-facing and must stay context-cheap). ``result_full`` is this
    endpoint's own addition for the web UI: this route is browser-only, so it
    can afford a much larger (but still capped) payload. ``session_id`` is
    the conversation the run was delivered to, which the UI opens.
    """
    full = r.result or r.error or ""
    return {
        "job_id": r.task_id,
        "status": r.status.value,
        "started_at_ms": r.started_at_ms,
        "ended_at_ms": r.ended_at_ms,
        "result_preview": (r.result[:200] if r.result else None),
        "result_full": full[:20000],
        "error": r.error,
        "duration_ms": r.duration_ms,
        "usage": r.usage,
        "session_id": r.session_id,
    }


def _parse_tool_result(result_str: str) -> dict:
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
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


# ============== Scheduler switch ==============

@router.get("/status")
async def api_status():
    scheduler = deps.get_scheduler()
    return {"enabled": scheduler.enabled, "interval_s": scheduler.interval()}


@router.put("/status")
async def api_set_status(body: SchedulerToggleRequest):
    scheduler = deps.get_scheduler()
    await scheduler.set_enabled(body.enabled)
    return {"enabled": scheduler.enabled, "interval_s": scheduler.interval()}


# ============== Job CRUD ==============

@router.get("/jobs")
async def api_list_jobs(request: Request, include_disabled: bool = True, limit: int = 500):
    jobs = list_jobs(user_id=_account(request), include_disabled=include_disabled, limit=limit)
    return {"jobs": [_job_dict(j) for j in jobs], "total": len(jobs)}


@router.get("/jobs/{job_id}")
async def api_get_job(job_id: str, request: Request):
    return {"job": _job_dict(_owned_job(request, job_id))}


@router.post("/jobs")
async def api_create_job(body: CronJobCreateRequest, request: Request):
    account = _account(request)
    fields = body.model_dump(exclude={"validate_run", "session_id", "work_dir"})
    session_id = body.session_id if body.target == "session" else None
    if session_id and not deps.get_agent_service().session_log_for(session_id, owner=account).path.exists():
        # Only a conversation of this account: its log lives under it.
        raise HTTPException(status_code=400, detail="Conversation not found")
    result = _parse_tool_result(_do_cronjob(
        action="create", user_id=account, caller_session_id=session_id,
        # No calling agent, so no tier ceiling; the directory is the
        # workspace the page picked for the conversations the job creates.
        mandate=("", body.work_dir or None), **fields,
    ))
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create job"))
    result = _job_response(result)
    if body.validate_run:
        result["validation_run"] = deps.get_scheduler().run_now(get_job(result["job"]["id"]))
    return result


@router.put("/jobs/{job_id}")
async def api_update_job(job_id: str, body: CronJobUpdateRequest, request: Request):
    job = _owned_job(request, job_id)
    updates = body.model_dump(exclude_none=True, exclude={"work_dir"})
    if body.work_dir is not None and body.work_dir != (job.work_dir or ""):
        updates_dir = {"work_dir": body.work_dir or None}
    else:
        updates_dir = {}
    if not updates and not updates_dir:
        raise HTTPException(status_code=400, detail="No updates provided")
    if updates:
        result = _parse_tool_result(_do_cronjob(action="update", job_id=job_id, mandate=("", None), **updates))
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Update failed"))
    if updates_dir:
        update_job(job_id, updates_dir)
    return {"success": True, "job": _job_dict(get_job(job_id))}


@router.delete("/jobs/{job_id}")
async def api_delete_job(job_id: str, request: Request):
    _owned_job(request, job_id)
    if not remove_job(job_id):
        raise HTTPException(status_code=400, detail="Delete failed")
    return {"status": "deleted", "job_id": job_id}


@router.post("/jobs/clear_completed")
async def api_clear_completed(request: Request):
    """Delete every completed job of this account; active and paused stay."""
    done = [j for j in list_jobs(user_id=_account(request), include_disabled=True)
            if j.status == JobStatus.COMPLETED]
    for j in done:
        remove_job(j.id)
    return {"deleted": len(done)}


# ============== Job actions ==============

@router.post("/jobs/{job_id}/pause")
async def api_pause_job(job_id: str, request: Request):
    _owned_job(request, job_id)
    if not pause_job(job_id):
        raise HTTPException(status_code=400, detail="Pause failed")
    return {"status": "paused", "job_id": job_id, "job": _job_dict(get_job(job_id))}


@router.post("/jobs/{job_id}/resume")
async def api_resume_job(job_id: str, request: Request):
    _owned_job(request, job_id)
    updated = resume_job(job_id)
    if not updated:
        raise HTTPException(status_code=400, detail="Resume failed")
    return {"status": "resumed", "job_id": job_id, "next_run_at_ms": updated.next_run_at_ms, "job": _job_dict(updated)}


@router.post("/jobs/{job_id}/trigger")
async def api_trigger_job(job_id: str, request: Request):
    """Start one run now, in the job's conversation, without waiting for it.

    Same execution path as the ticker. Returns the conversation the run
    lands in, which the page opens to watch it stream.
    """
    job = _owned_job(request, job_id)
    run = deps.get_scheduler().run_now(job)
    if run["status"] == "busy":
        raise HTTPException(status_code=409, detail=run["error"])
    return {"success": True, "run": run, "job": _job_dict(get_job(job_id))}


# ============== Prompt polishing (LLM-assisted authoring) ==============

_POLISH_SYSTEM_PROMPT = """You rewrite a rough scheduled-task idea into a clear, \
self-contained prompt for an AI agent. The prompt is delivered later as an \
automatic message into a conversation, with nobody present to answer \
follow-up questions, so it must:
- State the exact task and the expected deliverable/output format.
- Include any concrete details the user gave (paths, names, thresholds, etc).
- Avoid vague language like "check things" — be specific and actionable.
Reply with ONLY the rewritten prompt text, no preamble, no quotes, no markdown fences."""


@router.post("/polish_prompt")
async def polish_prompt(body: PolishPromptRequest):
    draft = body.draft.strip()
    if not draft:
        raise HTTPException(status_code=400, detail="draft must not be empty")

    try:
        polished = await deps.get_agent_service().side_task(_POLISH_SYSTEM_PROMPT, draft)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    if not polished:
        raise HTTPException(status_code=502, detail="LLM returned an empty response")
    return {"prompt": polished}


# ============== Run history ==============

@router.get("/runs")
async def api_list_runs(request: Request, limit: int = 50):
    owned = {j.id for j in list_jobs(user_id=_account(request), include_disabled=True)}
    runs = [r for r in list_task_runs(job_id=None, limit=limit * 4) if r.task_id in owned][:limit]
    return {"runs": [_run_dict(r) for r in runs], "total": len(runs)}


@router.get("/jobs/{job_id}/runs")
async def api_list_job_runs(job_id: str, request: Request, limit: int = 50):
    _owned_job(request, job_id)
    runs = list_task_runs(job_id=job_id, limit=limit)
    return {"job_id": job_id, "runs": [_run_dict(r) for r in runs], "total": len(runs)}
