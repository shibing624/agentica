# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unified cron job management tool for agentica agents.

Single compressed action-oriented tool to avoid schema/context bloat.
Agents call cronjob(action="create|list|update|pause|resume|remove|run", ...).

Security: cron prompts are scanned for injection patterns before storage.
"""
import re
from typing import Any, Callable, Optional

from agentica.agent.permissions import PERMISSION_MODES, validate_permission_mode
from agentica.tools.base import Tool
from agentica.tools.decorators import tool
from agentica.tools.helpers import tool_result
from agentica.cron.types import JOB_TARGETS, validate_deliver, validate_target
from agentica.cron.jobs import (
    CronJob,
    create_job,
    get_job,
    list_jobs,
    list_task_runs,
    update_job,
    remove_job,
    pause_job,
    resume_job,
    parse_schedule,
    schedule_to_human,
    CronSchedule,
    compute_next_run_at_ms,
)


# ============== Security ==============

_THREAT_PATTERNS = [
    (r"ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions", "prompt_injection"),
    (r"do\s+not\s+tell\s+the\s+user", "deception_hide"),
    (r"system\s+prompt\s+override", "sys_prompt_override"),
    (r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)", "disregard_rules"),
    (r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_curl"),
    (r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)", "exfil_wget"),
    (r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)", "read_secrets"),
    (r"authorized_keys", "ssh_backdoor"),
    (r"/etc/sudoers|visudo", "sudoers_mod"),
    (r"rm\s+-rf\s+/", "destructive_root_rm"),
]

_INVISIBLE_CHARS = {
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
}


def _scan_cron_prompt(prompt: str) -> str:
    """Scan a cron prompt for critical threats. Returns error string if blocked, else empty."""
    for char in _INVISIBLE_CHARS:
        if char in prompt:
            return f"Blocked: prompt contains invisible unicode U+{ord(char):04X} (possible injection)."
    for pattern, pid in _THREAT_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return f"Blocked: prompt matches threat pattern '{pid}'."
    return ""


# ============== Tool ==============

def _to_json(data: dict) -> str:
    """Convert dict to JSON string. Delegates to tool_result for consistency."""
    return tool_result(data)


def _format_job(job: CronJob) -> dict:
    """Format a CronJob for display."""
    return {
        "job_id": job.id,
        "name": job.name,
        "prompt_preview": job.prompt[:100] + "..." if len(job.prompt) > 100 else job.prompt,
        "schedule": schedule_to_human(job.schedule),
        "status": job.status.value,
        "enabled": job.enabled,
        "deliver": job.deliver,
        "next_run_at_ms": job.next_run_at_ms,
        "last_run_at_ms": job.last_run_at_ms,
        "last_status": job.last_status,
        "run_count": job.run_count,
        "timeout_seconds": job.timeout_seconds,
        "max_retries": job.max_retries,
        "retry_count": job.retry_count,
        "retry_delay_ms": job.retry_delay_ms,
        "permission_mode": job.permission_mode,
        "work_dir": job.work_dir,
        "paused_reason": job.paused_reason,
        "target": job.target,
        "session_id": job.session_id,
    }


_CRONJOB_DESCRIPTION = """Manage the user's scheduled (cron) jobs with a single tool. Prefer this over editing cron files by hand.

Actions:
- action='create'  -> schedule a new job. Requires prompt + schedule. Optional name/target/timeout_seconds/max_retries/permission_mode.
- action='list'    -> list all jobs with their id, schedule, status and next run time.
- action='runs'    -> show recent execution history (optionally for one job_id) to see the effect of past runs.
- action='update'  -> change an existing job's prompt/schedule/name/etc. Requires job_id.
- action='pause' / 'resume' -> disable/enable an existing job. Requires job_id. Pass reason when pausing
                     a job because it cannot proceed unattended; the user is shown it.
- action='remove'  -> delete a job. Requires job_id.
- action='run'     -> run a job ONCE right now (a trial run). Requires job_id. It is queued into the job's
                     conversation and runs once that conversation is idle; read the result there or via 'runs'.

A job runs every time under this session's permission tier and working directory, captured at create.
permission_mode may only make it stricter: 'ask' gives a trial job that reads and reports but has every
write, mutating command and network call denied (nobody is there to approve).

Every run is delivered as an automatic message into a conversation, and the answer stays there for the
user to read and follow up on. target picks the conversation:
- 'session'  -> this conversation: each run sees its history (good for "remind me here", digests).
- 'new_once' -> a conversation created on the first run, reused by every later run.
- 'new'      -> a fresh conversation for every run (no carry-over between runs).
Write the prompt so it still makes sense when it arrives later with nobody present to clarify.
deliver is always 'local' (no push to chat channels).

Not for waiting on a job this session started (training, eval, build, tmux/nohup): a background
execute watcher on its exit marker costs no tokens and reports back here; a cron poll runs a full
agent every tick and cannot message this session. A job that watches for an end state must say in its
prompt to pause itself (action='pause' with its own job_id) once that state is reached.

Schedule formats:
- Cron expression: "30 7 * * *" (daily 7:30), "0 9 * * 1-5" (weekdays 9:00)
- Interval: "30m", "every 2h", "5s", "1d"
- One-shot ISO: "2024-01-15T09:30:00"
"""


def _do_cronjob(
    action: str,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[str] = None,
    name: Optional[str] = None,
    deliver: Optional[str] = None,
    user_id: Optional[str] = None,
    timezone: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay_ms: Optional[int] = None,
    permission_mode: Optional[str] = None,
    reason: Optional[str] = None,
    target: Optional[str] = None,
    mandate: Optional[tuple[str, Optional[str]]] = None,
    caller_session_id: Optional[str] = None,
) -> str:
    """Pure business implementation of the cronjob tool (no immediate-run path).

    Single source of truth for every non-immediate-run action. Both the
    module-level ``@tool``-decorated ``cronjob`` and ``CronTool.cronjob``
    delegate here, so neither depends on the decorator's ``__call__``
    transparency (which was a fragile implicit contract).

    ``mandate`` is ``(permission_mode, work_dir)`` of the agent making the
    call, or None when there is none (HTTP route, SDK call without an agent).
    It is never a model-supplied value. ``caller_session_id`` is that agent's
    conversation, what ``target='session'`` delivers into; also never from the
    model.
    """
    try:
        normalized = (action or "").strip().lower()

        if normalized == "create":
            return _action_create(
                prompt,
                schedule,
                name,
                deliver,
                user_id,
                timezone,
                timeout_seconds,
                max_retries,
                retry_delay_ms,
                permission_mode,
                mandate,
                target or "new_once",
                caller_session_id,
            )
        if normalized == "list":
            return _action_list(user_id)
        if normalized == "runs":
            return _action_runs(job_id)
        if not job_id:
            return _to_json({"success": False, "error": f"job_id is required for action '{normalized}'"})

        job = get_job(job_id)
        if not job:
            return _to_json({"success": False, "error": f"Job '{job_id}' not found. Use cronjob(action='list') to see jobs."})

        if normalized == "remove":
            return _action_remove(job_id, job)
        if normalized == "pause":
            return _action_pause(job_id, reason)
        if normalized == "resume":
            return _action_resume(job_id)
        if normalized in {"run", "run_now", "trigger"}:
            return _action_trigger(job_id)
        if normalized == "update":
            return _action_update(
                job_id,
                prompt,
                schedule,
                name,
                deliver,
                timeout_seconds,
                max_retries,
                retry_delay_ms,
                permission_mode,
                mandate,
                target,
                caller_session_id,
                job,
                timezone,
            )

        return _to_json({"success": False, "error": f"Unknown action '{action}'"})

    except Exception as e:
        return _to_json({"success": False, "error": str(e)})


@tool(name="cronjob", description=_CRONJOB_DESCRIPTION, is_destructive=True)
def cronjob(
    action: str,
    job_id: Optional[str] = None,
    prompt: Optional[str] = None,
    schedule: Optional[str] = None,
    name: Optional[str] = None,
    deliver: Optional[str] = None,
    user_id: Optional[str] = None,
    timezone: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay_ms: Optional[int] = None,
    permission_mode: Optional[str] = None,
    reason: Optional[str] = None,
    target: Optional[str] = None,
) -> str:
    """Unified cron job management tool. See decorator description for actions."""
    return _do_cronjob(
        action=action,
        job_id=job_id,
        prompt=prompt,
        schedule=schedule,
        name=name,
        deliver=deliver,
        user_id=user_id,
        timezone=timezone,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_ms=retry_delay_ms,
        permission_mode=permission_mode,
        reason=reason,
        target=target,
    )


def _resolve_job_mandate(
    requested: Optional[str],
    mandate: Optional[tuple[str, Optional[str]]],
) -> tuple[str, Optional[str]]:
    """The (permission_mode, work_dir) a job will run under.

    The caller's tier is the ceiling: a job is a way to run later what this
    session could run now, not a way to get more. ``requested`` may only lower
    it. Without a calling agent there is no ceiling to hold the request to, so
    the request (validated) stands on its own and work_dir stays unset.
    """
    caller_mode, work_dir = mandate if mandate is not None else ("", None)
    if not requested:
        return caller_mode, work_dir
    validate_permission_mode(requested)
    if caller_mode and PERMISSION_MODES.index(requested) > PERMISSION_MODES.index(caller_mode):
        raise ValueError(
            f"permission_mode={requested!r} is broader than this session's tier {caller_mode!r}; "
            "a job may only run under the same tier or a stricter one."
        )
    return requested, work_dir


def _action_create(
    prompt: Optional[str],
    schedule: Optional[str],
    name: Optional[str],
    deliver: Optional[str],
    user_id: Optional[str],
    timezone: Optional[str],
    timeout_seconds: Optional[float],
    max_retries: Optional[int],
    retry_delay_ms: Optional[int],
    permission_mode: Optional[str],
    mandate: Optional[tuple[str, Optional[str]]],
    target: str,
    caller_session_id: Optional[str],
) -> str:
    if not prompt:
        return _to_json({"success": False, "error": "prompt is required for create"})
    if not schedule:
        return _to_json({"success": False, "error": "schedule is required for create"})

    # Security scan
    scan_error = _scan_cron_prompt(prompt)
    if scan_error:
        return _to_json({"success": False, "error": scan_error})

    job_mode, job_work_dir = _resolve_job_mandate(permission_mode, mandate)
    job = create_job(
        prompt=prompt,
        schedule=schedule,
        name=name,
        user_id=user_id or "default",
        deliver=deliver or "local",
        timezone=timezone or "Asia/Shanghai",
        timeout_seconds=timeout_seconds or 0.0,
        max_retries=max_retries or 0,
        retry_delay_ms=retry_delay_ms or 60000,
        permission_mode=job_mode,
        work_dir=job_work_dir,
        target=target,
        session_id=caller_session_id if target == "session" else None,
    )
    return _to_json({
        "success": True,
        "job": _format_job(job),
        "message": f"Cron job '{job.name}' created. {schedule_to_human(job.schedule)}.",
    })


def _action_list(user_id: Optional[str]) -> str:
    jobs = list_jobs(user_id=user_id, include_disabled=True)
    return _to_json({
        "success": True,
        "count": len(jobs),
        "jobs": [_format_job(j) for j in jobs],
    })


def _action_runs(job_id: Optional[str]) -> str:
    """Recent execution history, newest first. Optional job_id filter."""
    runs = list_task_runs(job_id=job_id, limit=20)
    return _to_json({
        "success": True,
        "count": len(runs),
        "runs": [
            {
                "job_id": r.task_id,
                "status": r.status.value if hasattr(r.status, "value") else str(r.status),
                "started_at_ms": r.started_at_ms,
                "ended_at_ms": r.ended_at_ms,
                "result_preview": (r.result[:200] if r.result else None),
                "error": r.error,
                "usage": r.usage,
            }
            for r in runs
        ],
    })


def _action_remove(job_id: str, job: CronJob) -> str:
    removed = remove_job(job_id)
    if not removed:
        return _to_json({"success": False, "error": f"Failed to remove job '{job_id}'"})
    return _to_json({
        "success": True,
        "message": f"Cron job '{job.name}' removed.",
    })


def _action_pause(job_id: str, reason: Optional[str]) -> str:
    updated = pause_job(job_id, reason=(reason or "").strip())
    if not updated:
        return _to_json({"success": False, "error": "Pause failed"})
    return _to_json({"success": True, "job": _format_job(updated)})


def _action_resume(job_id: str) -> str:
    updated = resume_job(job_id)
    if not updated:
        return _to_json({"success": False, "error": "Resume failed"})
    return _to_json({"success": True, "job": _format_job(updated)})


def _action_trigger(job_id: str) -> str:
    """Mark job as due immediately (will run on next tick)."""
    updated = update_job(job_id, {"next_run_at_ms": 1})
    if not updated:
        return _to_json({"success": False, "error": "Trigger failed"})
    return _to_json({
        "success": True,
        "job": _format_job(updated),
        "message": "Job will run on next scheduler tick.",
    })


def _action_update(
    job_id: str,
    prompt: Optional[str],
    schedule: Optional[str],
    name: Optional[str],
    deliver: Optional[str],
    timeout_seconds: Optional[float],
    max_retries: Optional[int],
    retry_delay_ms: Optional[int],
    permission_mode: Optional[str],
    mandate: Optional[tuple[str, Optional[str]]],
    target: Optional[str],
    caller_session_id: Optional[str],
    job: CronJob,
    timezone: Optional[str] = None,
) -> str:
    updates: dict = {}
    if prompt is not None:
        scan_error = _scan_cron_prompt(prompt)
        if scan_error:
            return _to_json({"success": False, "error": scan_error})
        updates["prompt"] = prompt
    if name is not None:
        updates["name"] = name
    if deliver is not None:
        updates["deliver"] = validate_deliver(deliver)
    if timeout_seconds is not None:
        updates["timeout_seconds"] = timeout_seconds
    if max_retries is not None:
        updates["max_retries"] = max_retries
    if retry_delay_ms is not None:
        updates["retry_delay_ms"] = retry_delay_ms
    if permission_mode is not None:
        # Re-tiering is held to the editing session's ceiling, same as create.
        # work_dir is not moved: an edit of the schedule or prompt from another
        # directory must not silently relocate the job.
        updates["permission_mode"], _ = _resolve_job_mandate(permission_mode, mandate)
    if target is not None and target != job.target:
        # Into a conversation: the one the job already lands in, else the
        # editing one. Away from one: later runs make their own.
        session_id = (job.session_id or caller_session_id or "") if target == "session" else ""
        updates["target"] = validate_target(target, session_id)
        updates["session_id"] = session_id
    if timezone is not None:
        updates["timezone"] = timezone
    if schedule is not None or timezone is not None:
        parsed = parse_schedule(schedule) if schedule is not None else job.schedule
        if isinstance(parsed, CronSchedule):
            parsed = CronSchedule(expression=parsed.expression, timezone=timezone or job.timezone)
        updates["schedule"] = parsed.to_dict()
        next_run = compute_next_run_at_ms(parsed)
        updates["next_run_at_ms"] = next_run or 0

    if not updates:
        return _to_json({"success": False, "error": "No updates provided."})

    updated = update_job(job_id, updates)
    if not updated:
        return _to_json({"success": False, "error": "Update failed"})
    return _to_json({"success": True, "job": _format_job(updated)})


# ============== Tool class wrapper for Agent(tools=[CronTool()]) ==============

class CronTool(Tool):
    """Cron job management tool for Agent integration.

    Usage:
        agent = Agent(tools=[CronTool()])

    Args:
        job_runner: optional ``(CronJob) -> dict`` callback for ``action='run'``.
            It returns an ``execute_job``-shaped dict: the finished run (the CLI
            runs it in place), or ``status='queued'`` when the run was handed to
            the job's conversation and will happen there (the gateway, where the
            conversation may be this very one and busy with this call). When
            omitted (SDK / unattended usage), ``action='run'`` falls back to
            marking the job due on the next scheduler tick.
        owner: when set (gateway web/chat), create/list/edit are scoped to this
            account. The model cannot name a different ``user_id``.
    """

    def __init__(
        self,
        job_runner: Optional[Callable[[CronJob], dict]] = None,
        owner: Optional[str] = None,
    ):
        super().__init__(name="cronjob", description=_CRONJOB_DESCRIPTION)
        self._job_runner = job_runner
        self._owner = owner
        self.register(self.cronjob, is_destructive=True)
        # The immediate-run path spawns a sub-agent in a worker thread via its
        # own asyncio.run() loop, which the outer asyncio.wait_for() cannot
        # cancel — a 120s outer timeout would orphan that thread (still burning
        # tokens) while reporting a timeout to the LLM. execute_job already
        # enforces the job's own timeout_seconds internally, so let the tool
        # manage its own timeout instead of wrapping it.
        self.functions["cronjob"].manages_own_timeout = True

    # No docstring on the method: Tool.register() falls back to self.description
    # (= _CRONJOB_DESCRIPTION, set in __init__) when function.__doc__ is None,
    # so the LLM-facing schema stays in perfect sync with the free-function
    # @tool decorator without duplicating the description text.
    def cronjob(
        self,
        action: str,
        job_id: Optional[str] = None,
        prompt: Optional[str] = None,
        schedule: Optional[str] = None,
        name: Optional[str] = None,
        deliver: Optional[str] = None,
        user_id: Optional[str] = None,
        timezone: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_delay_ms: Optional[int] = None,
        permission_mode: Optional[str] = None,
        reason: Optional[str] = None,
        target: Optional[str] = None,
        agent: Any = None,
    ) -> str:
        normalized = (action or "").strip().lower()
        if self._owner:
            user_id = self._owner
        if job_id and self._owner:
            scoped = get_job(job_id)
            if scoped is not None and scoped.user_id != self._owner:
                return _to_json({"success": False, "error": f"Job '{job_id}' not found. Use cronjob(action='list') to see jobs."})
        # Immediate trial run: only when an executor is wired (interactive CLI).
        if normalized in {"run", "run_now", "trigger"} and self._job_runner is not None:
            if not job_id:
                return _to_json({"success": False, "error": "job_id is required for action 'run'"})
            job = get_job(job_id)
            if not job:
                return _to_json({"success": False, "error": f"Job '{job_id}' not found. Use action='list' to see jobs."})
            return self._run_now(job_id, job)
        # Everything else (and run without an executor) uses the pure impl.
        return _do_cronjob(
            action=action,
            job_id=job_id,
            prompt=prompt,
            schedule=schedule,
            name=name,
            deliver=deliver,
            user_id=user_id,
            timezone=timezone,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_delay_ms=retry_delay_ms,
            permission_mode=permission_mode,
            reason=reason,
            target=target,
            # ``agent`` is injected by the runtime (FunctionCall._build_args),
            # never part of the schema: the job's mandate is the calling
            # session's tier and directory, and target='session' means the
            # calling session — not something the model names.
            mandate=(agent.tool_config.permission_mode, agent.work_dir) if agent is not None else None,
            caller_session_id=agent.session_id if agent is not None else None,
        )

    def _run_now(self, job_id: str, job: CronJob) -> str:
        """Run a job once now: its result, or where the queued run will land."""
        try:
            result = self._job_runner(job) or {}
        except Exception as e:  # surface as tool error, do not crash the turn
            return _to_json({"success": False, "error": f"Immediate run failed: {e}"})
        status = result.get("status")
        if status == "queued":
            return _to_json({
                "success": True,
                "job_id": job_id,
                "run": result,
                "message": (
                    f"Job '{job.name}' is queued to run now in its conversation "
                    f"({result.get('session_id') or 'a new one'}). It starts once that conversation is "
                    "idle; its answer appears there and in action='runs'."
                ),
            })
        # The sub-agent's full response text can be large; returning it verbatim
        # stuffs the parent turn's context. Keep a bounded preview plus length /
        # truncated flags so the LLM knows the run happened and how big it was,
        # without the token bloat.
        text = result.get("result")
        if isinstance(text, str):
            full_len = len(text)
            result = {**result, "result": text[:2000],
                      "result_truncated": full_len > 2000, "result_full_length": full_len}
        return _to_json({
            "success": status == "ok",
            "job_id": job_id,
            "run": result,
            "message": f"Executed job '{job.name}' once now (status={status}).",
        })

    def __repr__(self) -> str:
        return "CronTool()"
