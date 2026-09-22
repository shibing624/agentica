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
    get_due_jobs,
    mark_job_run,
    now_ms,
)
from agentica.cron.types import RunStatus
from agentica.run.context import RunSource

logger = logging.getLogger(__name__)

LOCK_FILE = CRON_DIR / ".tick.lock"


# ============== Agent Runner Protocol ==============

class AgentRunner(Protocol):
    """Protocol for executing agent tasks from cron jobs."""

    async def run(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        """Run agent with a prompt and return the result text."""
        ...


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
            result = await _execute_job(job, agent_runner, verbose)
            results.append(result)

        return results

    finally:
        lock.release()


async def _execute_job(
    job: CronJob,
    agent_runner: AgentRunner | None,
    verbose: bool,
) -> dict[str, Any]:
    """Execute a single cron job."""
    logger.info(f"Executing job {job.id}: {job.name}")
    started_at_ms = now_ms()
    attempt = job.retry_count + 1
    if verbose:
        print(f"  Running: {job.name} ({job.id})")

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
            "permissions": job.permissions,
            "run_source": RunSource.cron.value,
        }
        run_coro = agent_runner.run(prompt=job.prompt, context=context)
        if job.timeout_seconds > 0:
            result_text = await asyncio.wait_for(run_coro, timeout=job.timeout_seconds)
        else:
            result_text = await run_coro
        mark_job_run(
            job.id,
            RunStatus.OK,
            result=result_text,
            started_at_ms=started_at_ms,
            ended_at_ms=now_ms(),
            attempt=attempt,
        )

        if verbose:
            preview = result_text[:100] + "..." if len(result_text) > 100 else result_text
            print(f"  OK: {preview}")

        return {"job_id": job.id, "status": "ok", "result": result_text}

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
