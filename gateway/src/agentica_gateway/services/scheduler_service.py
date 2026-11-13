# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: The gateway's scheduled tasks: every run is a turn in a conversation.

A job's run is delivered into the conversation its ``target`` names (the one it
was created in, one made on its first run, or a fresh one each time), as an
automated live turn: an open page streams it, the sidebar shows it running,
and the answer stays where the user can read it and reply. IM chats bound to
that conversation get the turn through the IM bot, as with a finished
background command.

Each due job runs as its own task so that one busy conversation (a run waits
for its conversation to be idle) never holds back the others. The ticker can
be switched on and off without a restart; the switch is the same
``cron.enabled`` setting the CLI's ``/cron daemon on`` writes.
"""
import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from agentica.config.profiles import get_setting, set_setting
from agentica.cron.cli_runner import job_turn_meta, live_cli_peer
from agentica.cron.jobs import CronJob
from agentica.cron.scheduler import claim_due_jobs, execute_job, is_job_running
from agentica.cron.types import CronRunOutput
from agentica.utils.log import logger

from ..config import settings

# TaskRun.usage keys, picked out of ChatResult.turn_usage.
_CRON_USAGE_KEYS = (
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "api_calls", "cost_usd",
)
DEFAULT_TICK_S = 60


class GatewayCronRunner:
    """``AgentRunner`` that delivers a run into its job's conversation."""

    def __init__(self, agent_svc: Any, deliver_im: Optional[Any] = None):
        self._svc = agent_svc
        # ``(owner, session_id, text, job_name) -> Optional[ChatResult]``: runs
        # the turn in the IM chats bound to the conversation, None when none is.
        self._deliver_im = deliver_im

    def prepare_session(self, job: CronJob, session_id: str) -> None:
        """Create and title a fresh scheduled conversation synchronously before
        spawning its run, so a client opening ``session_id`` right away finds
        the session and its job title instead of ``Chat`` or 404."""
        from ..routes.chat import _touch_session_log

        owner = job.user_id or settings.default_user_id
        if job.work_dir and Path(job.work_dir).is_dir():
            self._svc.set_session_work_dir(session_id, job.work_dir, owner)
        if job.permission_mode:
            self._svc.set_session_approval_mode(session_id, job.permission_mode, owner)
        _touch_session_log(self._svc, session_id, owner)
        log = self._svc.session_log_for(session_id, owner=owner)
        name = (job.name or "").strip()
        if name and not log.get_name():
            log.set_name(name[:60])

    async def run(self, prompt: str, context: Optional[dict] = None) -> CronRunOutput:
        from ..routes.chat import run_scheduled_turn

        ctx = context or {}
        owner = ctx.get("user_id") or settings.default_user_id
        fresh = bool(ctx["fresh_session"]) if "fresh_session" in ctx else not ctx.get("session_id")
        session_id = ctx.get("session_id") or str(uuid.uuid4())
        name = (ctx.get("job_name") or "").strip()
        peer = live_cli_peer(session_id)
        if peer is not None:
            # The terminal owns this conversation's log; a gateway turn would
            # write it alongside. Ask it to run the turn over the same attach
            # socket the IM path uses, which also hands the answer back — the
            # mailbox handoff this replaces could only report that the prompt
            # had been handed over, so the automation record was always empty.
            from agentica.cli.attach import prompt_live_session
            from agentica.cron.cli_runner import ATTACH_PROMPT_TIMEOUT

            result = await asyncio.to_thread(
                prompt_live_session,
                peer,
                prompt,
                session_id=session_id,
                timeout=ATTACH_PROMPT_TIMEOUT,
                # Nobody at that terminal typed this: keep it out of the
                # user's transcript styling and records, and let jobs it
                # starts name this automation's ticket.
                automated=True,
                ticket_id=ctx.get("ticket_id"),
            )
            if result is not None:
                answer = str(result.get("agenticaAnswer") or "").strip()
                if not answer:
                    if result.get("stopReason") == "cancelled":
                        answer = f"The open terminal [{peer.name}] cancelled this run."
                    else:
                        answer = (
                            f"Handed to the open terminal [{peer.name}]; it has not "
                            "reported a result yet."
                        )
                return CronRunOutput(content=answer, session_id=session_id)
            from agentica.peers import send_message

            send_message(peer, text=prompt, from_name="cron", from_peer_id="cron", from_kind="agent")
            return CronRunOutput(
                content=f"Handed to the open terminal [{peer.name}] serving this conversation.",
                session_id=session_id,
            )
        work_dir = ctx.get("work_dir")
        if work_dir and Path(work_dir).is_dir():
            self._svc.set_session_work_dir(session_id, work_dir, owner)
        if ctx.get("permission_mode"):
            self._svc.set_session_approval_mode(session_id, ctx["permission_mode"], owner)
        log = self._svc.session_log_for(session_id, owner=owner)
        if fresh and name and not log.get_name():
            # A conversation the job starts is named after it. One it was
            # pointed at already belongs to the user and keeps its title.
            log.set_name(name[:60])
        meta = job_turn_meta(ctx)
        result = None
        # The job's dispatch ticket, so the conversation this run creates records
        # the same name as the automation that created it. ``ticket_scope`` is
        # the channel because the Runner resolves the ambient value; nothing
        # downstream needs a new parameter.
        from agentica.ticket import ticket_scope

        with ticket_scope(ctx.get("ticket_id")):
            if self._deliver_im is not None:
                result = await self._deliver_im(owner, session_id, prompt, meta)
            if result is None:
                result = await run_scheduled_turn(self._svc, session_id, owner, prompt, meta)
        if result.error:
            raise RuntimeError(result.error)
        usage = None
        if result.turn_usage:
            usage = {k: result.turn_usage.get(k) for k in _CRON_USAGE_KEYS}
        return CronRunOutput(content=result.content, usage=usage, session_id=session_id)


class GatewayScheduler:
    """Owns the ticker and the in-flight runs of one gateway process."""

    def __init__(self, runner: GatewayCronRunner):
        self.runner = runner
        self._ticker: Optional[asyncio.Task] = None
        self._runs: set[asyncio.Task] = set()
        # Job ids with a run started here and not yet ended. Taken at spawn,
        # before the task first runs: execute_job only marks the job running
        # once the task starts, so two quick "run now"s would both pass.
        self._pending: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._ticker is not None and not self._ticker.done()

    @staticmethod
    def interval() -> int:
        return int(get_setting("cron.interval", DEFAULT_TICK_S) or DEFAULT_TICK_S)

    def start(self) -> None:
        if not self.enabled:
            self._ticker = asyncio.create_task(self._tick_forever())
            logger.info(f"Cron scheduler started ({self.interval()}s tick)")

    async def stop(self) -> None:
        ticker, self._ticker = self._ticker, None
        if ticker is not None:
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

    async def set_enabled(self, enabled: bool) -> None:
        """Turn the ticker on or off now, and remember it for the next start."""
        set_setting("cron.enabled", bool(enabled))
        if enabled:
            self.start()
        else:
            await self.stop()

    async def shutdown(self) -> None:
        await self.stop()
        for task in list(self._runs):
            task.cancel()

    async def _tick_forever(self) -> None:
        while True:
            try:
                self.tick_once()
            except Exception as e:
                logger.error(f"Cron tick error: {e}")
            await asyncio.sleep(self.interval())

    def tick_once(self) -> int:
        """Start every due job; return how many started."""
        jobs = claim_due_jobs()
        started = [job for job in jobs if job.id not in self._pending]
        for job in started:
            self._spawn(job, job.delivery_session_id())
        return len(started)

    def run_now(self, job: CronJob) -> Dict[str, Any]:
        """Start one run of ``job`` now, in the background.

        Returns where it will land without waiting for it: the caller may be
        a turn of that very conversation, which the run waits for.
        """
        if job.id in self._pending or is_job_running(job.id):
            return {"job_id": job.id, "status": "busy", "error": "A run of this job is already in progress"}
        existing_sid = job.delivery_session_id()
        session_id = existing_sid or str(uuid.uuid4())
        if not existing_sid and isinstance(self.runner, GatewayCronRunner):
            self.runner.prepare_session(job, session_id)
        self._spawn(job, session_id)
        return {"job_id": job.id, "status": "queued", "session_id": session_id}

    def _spawn(self, job: CronJob, session_id: str) -> asyncio.Task:
        self._pending.add(job.id)
        task = asyncio.create_task(execute_job(job, self.runner, verbose=False, session_id=session_id))
        self._runs.add(task)

        def _ended(t: asyncio.Task) -> None:
            self._runs.discard(t)
            self._pending.discard(job.id)

        task.add_done_callback(_ended)
        return task

    def is_running(self, job_id: str) -> bool:
        return job_id in self._pending or is_job_running(job_id)
