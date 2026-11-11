# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them.
The gateway calls this periodically (e.g. every 60 seconds).

Uses a file-based lock so only one tick runs at a time.
"""
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Protocol

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[assignment]
    try:
        import msvcrt
    except ImportError:
        msvcrt = None  # type: ignore[assignment]

from agentica.cron.jobs import (
    CRON_DIR,
    CronJob,
    compute_next_run_at_ms,
    get_due_jobs,
    mark_job_run,
    now_ms,
    pause_job,
    update_job,
)
from agentica.cron.types import CronRunOutput, RunStatus
from agentica.run.context import RunSource

logger = logging.getLogger(__name__)

LOCK_FILE = CRON_DIR / ".tick.lock"

# Jobs with a run in flight in this process. A run lands in a conversation,
# and a second run of the same job (a tick while "run now" is going, or a
# slow run outliving the next tick) would race the first for that turn.
_running_jobs: set[str] = set()


def is_job_running(job_id: str) -> bool:
    """True while a run of this job is in flight in this process."""
    return job_id in _running_jobs


# ============== Agent Runner Protocol ==============

class AgentRunner(Protocol):
    """Protocol for executing agent tasks from cron jobs."""

    async def run(self, prompt: str, context: dict[str, Any] | None = None) -> str | CronRunOutput:
        """Run agent with a prompt; return the result text, or text plus token usage."""
        ...


def run_usage(response: Any) -> dict[str, Any] | None:
    """``TaskRun.usage`` from a ``RunResponse``'s cost tracker, or None.

    Counts every main-model request of the run, tool rounds included — the
    same numbers as the CLI footer and the gateway's per-turn usage.
    Auxiliary-model calls (memory recall selection, memory extraction) are
    not in the run's cost tracker and are not counted.
    """
    from agentica.model.cost import CostTracker

    tracker = getattr(response, "cost_tracker", None)
    if not isinstance(tracker, CostTracker) or tracker.turns == 0:
        return None
    return {
        "input_tokens": tracker.total_prompt_tokens,
        "output_tokens": tracker.total_output_tokens,
        "cache_read_tokens": tracker.total_cache_read_tokens,
        "cache_write_tokens": tracker.total_cache_write_tokens,
        "api_calls": tracker.turns,
        "cost_usd": round(tracker.total_cost_usd, 6),
    }


def scheduled_turn_text(job_id: str, name: str, prompt: str) -> str:
    """The message a scheduled run delivers into its conversation.

    The first line names the job, which is how a transcript (and the model)
    tells the turn from something the user typed. The job's prompt was written
    before it had an id, so "pause yourself when stuck" needs the id from here.
    """
    return (
        f"[Scheduled task \u201c{name}\u201d ({job_id})]\n{prompt}\n\n"
        "This message was sent by a scheduled task, not typed by the user; nobody may be watching "
        "to answer or approve. If the task cannot go on without a person (a needed permission is "
        "denied, a credential or path is missing, the work is outside what it covers), call "
        f"`cronjob(action='pause', job_id='{job_id}', reason=<one line>)` and stop: the user is "
        "shown the reason, and the task does not fire again until they resume it."
    )


def mandate_problem(job: CronJob) -> str:
    """Why the job's recorded mandate cannot be honored now, or "".

    Checked before every run, so a job whose directory is gone stops instead
    of running somewhere it was never meant to.
    """
    if job.work_dir and not os.path.isdir(job.work_dir):
        return f"work_dir {job.work_dir} no longer exists"
    return ""


# ============== File Lock ==============

class _FileLock:
    """Simple file lock using fcntl (Unix) or msvcrt (Windows)."""

    def __init__(self, path: Path):
        self.path = path
        self._fd = None

    def acquire(self) -> bool:
        """Try to acquire the lock. Returns True if acquired."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = open(self.path, "w")
            if fcntl is not None:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                msvcrt.locking(self._fd.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # No file locking available — proceed without lock
                pass
            return True
        except (OSError, IOError):
            if self._fd:
                self._fd.close()
                self._fd = None
            return False

    def release(self) -> None:
        """Release the lock."""
        if self._fd:
            try:
                if fcntl is not None:
                    fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
            finally:
                self._fd.close()
                self._fd = None


# ============== Tick ==============

async def tick(
    agent_runner: AgentRunner | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Check for due jobs and execute them.

    This is the main scheduler entry point. Call it periodically
    (e.g. every 60 seconds from the gateway).

    Args:
        agent_runner: Implementation that runs agent prompts.
            If None, jobs are marked as failed with "no runner".
        verbose: If True, print execution details.

    Returns:
        List of run results [{job_id, status, result_or_error}].
    """
    lock = _FileLock(LOCK_FILE)
    if not lock.acquire():
        if verbose:
            logger.info("Tick skipped: another tick is running")
        return []

    try:
        due_jobs = get_due_jobs()
        if not due_jobs and verbose:
            logger.info("No due jobs")
            return []

        results = []
        for job in due_jobs:
            result = await execute_job(job, agent_runner, verbose)
            results.append(result)

        return results

    finally:
        lock.release()


def claim_due_jobs(current_ms: int | None = None) -> list[CronJob]:
    """Take the due jobs for a host that runs each one as its own task.

    ``tick`` runs due jobs one after another while holding the tick lock. A
    host that delivers runs into conversations cannot: a run waits for its
    conversation to be idle, and one busy conversation would hold back every
    other job. So the claim is made under the lock instead: each due job's
    next run is moved past now before the lock is released, which is what
    stops another process's tick from taking the same occurrence. The run's
    own ``mark_job_run`` sets the real next run when it ends.
    """
    lock = _FileLock(LOCK_FILE)
    if not lock.acquire():
        return []
    try:
        current = current_ms or now_ms()
        claimed = []
        for job in get_due_jobs(current):
            if job.id in _running_jobs:
                continue
            update_job(job.id, {"next_run_at_ms": compute_next_run_at_ms(job.schedule, current, current) or 0})
            claimed.append(job)
        return claimed
    finally:
        lock.release()


async def execute_job(
    job: CronJob,
    agent_runner: AgentRunner | None,
    verbose: bool,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Execute a single cron job.

    Public: the gateway (a separate distribution) and the CLI both run jobs
    through this, so it is part of the cross-package API.

    A job already running in this process is skipped (status ``"busy"``),
    without a run record: nothing ran.

    ``session_id`` names the conversation this run is delivered into; by
    default the job's own (``CronJob.delivery_session_id``), and "" lets the
    runner start one. A host that must tell the caller where the run will
    land before it starts picks the id itself and passes it here.
    """
    if job.id in _running_jobs:
        return {"job_id": job.id, "status": "busy", "error": "A run of this job is already in progress"}
    _running_jobs.add(job.id)
    try:
        existing_sid = job.delivery_session_id()
        return await _execute_job_once(
            job, agent_runner, verbose,
            existing_sid if session_id is None else session_id,
            fresh_session=not existing_sid,
        )
    finally:
        _running_jobs.discard(job.id)


async def _execute_job_once(
    job: CronJob,
    agent_runner: AgentRunner | None,
    verbose: bool,
    session_id: str,
    *,
    fresh_session: bool = False,
) -> dict[str, Any]:
    logger.info(f"Executing job {job.id}: {job.name}")
    started_at_ms = now_ms()
    attempt = job.retry_count + 1
    if verbose:
        print(f"  Running: {job.name} ({job.id})")

    problem = mandate_problem(job)
    if problem:
        pause_job(job.id, reason=problem)
        mark_job_run(
            job.id,
            RunStatus.FAILED,
            error=f"Not run, job paused: {problem}",
            error_type="MandateError",
            started_at_ms=started_at_ms,
            ended_at_ms=now_ms(),
            attempt=attempt,
        )
        logger.warning(f"Job {job.id} paused: {problem}")
        return {"job_id": job.id, "status": "paused", "error": problem}

    if not agent_runner:
        ended_at_ms = now_ms()
        mark_job_run(
            job.id,
            RunStatus.FAILED,
            error="No agent runner configured",
            error_type="ConfigurationError",
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            attempt=attempt,
        )
        return {"job_id": job.id, "status": "failed", "error": "No agent runner configured"}

    try:
        context = {
            "job_id": job.id,
            "task_id": job.id,
            "user_id": job.user_id,
            "scheduled": True,
            "attempt": attempt,
            "workspace": job.workspace,
            "permission_mode": job.permission_mode,
            "work_dir": job.work_dir,
            "run_source": RunSource.cron.value,
            "job_name": job.name,
            "message": scheduled_turn_text(job.id, job.name, job.prompt),
            "target": job.target,
            # The conversation to deliver into; "" means start a new one.
            "session_id": session_id,
            # True when the job itself starts a new conversation (even if the
            # caller pre-allocated ``session_id`` for immediate UI navigation).
            "fresh_session": fresh_session or not session_id,
        }
        run_coro = agent_runner.run(prompt=context["message"], context=context)
        if job.timeout_seconds > 0:
            output = await asyncio.wait_for(run_coro, timeout=job.timeout_seconds)
        else:
            output = await run_coro
        if isinstance(output, CronRunOutput):
            result_text, usage, session_id = output.content, output.usage, output.session_id or session_id
        else:
            result_text, usage = output, None
        mark_job_run(
            job.id,
            RunStatus.OK,
            result=result_text,
            started_at_ms=started_at_ms,
            ended_at_ms=now_ms(),
            attempt=attempt,
            usage=usage,
            session_id=session_id,
        )

        if verbose:
            preview = result_text[:100] + "..." if len(result_text) > 100 else result_text
            print(f"  OK: {preview}")

        return {"job_id": job.id, "status": "ok", "result": result_text, "usage": usage, "session_id": session_id}

    except asyncio.TimeoutError:
        ended_at_ms = now_ms()
        error_msg = f"Job timed out after {job.timeout_seconds:g}s"
        mark_job_run(
            job.id,
            RunStatus.TIMEOUT,
            error=error_msg,
            error_type="TimeoutError",
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            attempt=attempt,
        )
        logger.error(f"Job {job.id} timed out after {job.timeout_seconds:g}s", exc_info=True)
        if verbose:
            print(f"  TIMEOUT: {error_msg}")
        return {"job_id": job.id, "status": "timeout", "error": error_msg}

    except Exception as e:
        ended_at_ms = now_ms()
        error_msg = str(e)
        mark_job_run(
            job.id,
            RunStatus.FAILED,
            error=error_msg,
            error_type=type(e).__name__,
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            attempt=attempt,
        )
        logger.error(f"Job {job.id} failed: {error_msg}", exc_info=True)
        if verbose:
            print(f"  FAILED: {error_msg}")
        return {"job_id": job.id, "status": "failed", "error": error_msg}


# Pre-1.5.1 name, kept for callers outside this repo.
_execute_job = execute_job
