# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Cron runtime glue for the CLI.

Provides:
- ``CliAgentRunner``: adapts the CLI's agent configuration to the cron
  ``AgentRunner`` protocol. Each run builds a FRESH agent from the same config
  and delivers the prompt as an automated turn of the job's own conversation
  (``CronJob.target``), never the interactive session's in-memory agent.
- ``run_scheduler_loop``: a blocking loop that periodically calls
  ``cron.scheduler.tick``. Designed to run inside a daemon thread (the CLI is
  thread-based, not asyncio-based), where it spins its own event loop. Also
  reused by the standalone ``agentica cron daemon`` subcommand.
"""

import asyncio
import threading
import time
from typing import Any, Callable, Dict, Optional

from agentica.utils.log import logger

from agentica.cron.scheduler import run_usage, tick
from agentica.cron.types import CronRunOutput
from agentica.run.config import RunConfig
from agentica.run.context import RunSource


def job_turn_meta(context: Dict[str, Any]) -> Dict[str, Any]:
    """What a run records on its ``user`` log entry: which job delivered it.

    Lets a transcript show the turn as the job's, with a link back to it,
    instead of as something the user typed.
    """
    return {"cron_job": {"id": context.get("job_id") or "", "name": context.get("job_name") or ""}}


# How long a run waits for the terminal to finish the turn it was handed.
# Bounded on purpose: ``tick`` holds the scheduler's file lock while awaiting a
# run, so waiting out the attach protocol's 600s grace would stall every other
# due job (and make another process's tick report "another tick is running").
# Two minutes covers the short turns this handoff is worth waiting for; a longer
# one gets the honest "still working" line and finishes in the terminal.
ATTACH_PROMPT_TIMEOUT = 120.0


def _prompt_open_terminal(peer: Any, prompt: str, session_id: str) -> Optional[str]:
    """Run this run's turn in the terminal that has the conversation open.

    ``None`` means the terminal could not take it (no socket, no token, RPC
    failed) and the caller should fall back. Any *answer* — including "still
    working on it" or "cancelled there" — is a result: the terminal did accept
    the turn, so falling back would run the same prompt twice.
    """
    from agentica.cli.attach import prompt_live_session

    result = prompt_live_session(
        peer, prompt, session_id=session_id, timeout=ATTACH_PROMPT_TIMEOUT,
    )
    if result is None:
        return None
    answer = str(result.get("agenticaAnswer") or "").strip()
    if answer:
        return answer
    if result.get("stopReason") == "cancelled":
        return f"The open terminal [{peer.name}] cancelled this run."
    return f"Handed to the open terminal [{peer.name}]; it has not reported a result yet."


def live_cli_peer(session_id: str):
    """The live terminal serving ``session_id``, or None.

    A run must not write a conversation's log while a terminal has it open:
    two writers interleave one JSONL. The terminal gets the prompt instead.
    """
    from agentica.peers import list_live_peers

    for peer in list_live_peers():
        if peer.session_id == session_id:
            return peer
    return None

# Keys in a CLI agent_config that name the interactive session's transcript.
# A cron job built from that config must not inherit them.
_SESSION_IDENTITY_KEYS = ("session_id", "session_base_dir", "_resume_at_uuid", "_fork_session")


class CliAgentRunner:
    """Adapt a CLI agent factory to the cron ``AgentRunner`` protocol.

    Args:
        agent_factory: callable returning a fresh agent instance. A new agent
            is built per job run to isolate state. The factory may accept
            optional ``workspace`` / ``user_id`` overrides so a job that targets
            a specific workspace is honored (the scheduler passes these in the
            run context).
    """

    def __init__(self, agent_factory: Callable[..., Any]):
        self._factory = agent_factory

    async def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> CronRunOutput:
        context = context or {}
        fresh = bool(context["fresh_session"]) if "fresh_session" in context else not context.get("session_id")
        session_id = context.get("session_id") or _new_session_id()
        job_name = (context.get("job_name") or "").strip()
        peer = live_cli_peer(session_id)
        if peer is not None:
            # A terminal owns this conversation, so this run must not write its
            # log — two writers interleave one JSONL. Ask that terminal to run
            # the turn over the same attach socket IM uses, which also hands the
            # answer back: the mailbox handoff this replaces could only say the
            # prompt had been handed over, so a scheduled run landing on an open
            # terminal produced an automation record with no output at all.
            answer = await asyncio.to_thread(_prompt_open_terminal, peer, prompt, session_id)
            if answer is not None:
                return CronRunOutput(content=answer, session_id=session_id)
            from agentica.peers import send_message

            send_message(peer, text=prompt, from_name="cron", from_peer_id="cron", from_kind="agent")
            return CronRunOutput(
                content=f"Handed to the open terminal [{peer.name}] serving this conversation.",
                session_id=session_id,
            )
        # Honor a per-job workspace from the cron run context instead of
        # silently dropping it. The scheduler passes a path string (or None);
        # wrap it in a Workspace. None -> the CLI's own default workspace.
        ws = context.get("workspace")
        if isinstance(ws, str) and ws:
            from agentica.workspace import Workspace
            ws = Workspace(path=ws)
        agent = self._factory(
            workspace=ws,
            permission_mode=context.get("permission_mode") or None,
            work_dir=context.get("work_dir") or None,
            session_id=session_id,
        )
        if fresh and job_name and agent._session_log is not None and not agent._session_log.get_name():
            agent._session_log.set_name(job_name[:60])
        response = await agent.run(message=prompt, config=RunConfig(
            source=RunSource.cron, automated_input=True, input_meta=job_turn_meta(context),
            ticket_id=context.get("ticket_id") or None,
        ))
        # RunResponse.content is the final text; fall back to "" for safety.
        return CronRunOutput(
            content=getattr(response, "content", None) or "", usage=run_usage(response), session_id=session_id,
        )


def _new_session_id() -> str:
    from uuid import uuid4

    return str(uuid4())


def build_cli_agent_factory(agent_config: Dict[str, Any], extra_tools=None,
                            workspace=None, skills_registry=None) -> Callable[..., Any]:
    """Return a factory that builds a fresh CLI agent from the given config.

    The returned factory accepts optional ``workspace`` / ``permission_mode`` /
    ``work_dir`` overrides: a job's mandate, so it runs under the tier and
    directory it was created with rather than whatever the host session is in
    when the tick fires (``/permissions`` mutates ``agent_config`` in place,
    and this closure holds that same dict). Without overrides it falls back to
    the defaults captured here. Imported lazily to avoid a circular import
    (cli.config imports tools which may import cron).

    The job gets the interactive session's model and tools, not its session.
    Inheriting the host's ``session_id`` made every tick replay the host's
    transcript as its own history and append its turns to the host's JSONL:
    production saw 419 monitor turns interleaved into a 9-day session, its
    search_session drowned in them, and each tick's startup
    ``seal_incomplete_turn`` stamping "[Session ended before the assistant
    replied]" over the host's live turn. A run writes the conversation the job
    names (``session_id``), and only when no terminal has it open (see
    ``live_cli_peer``).
    """
    workspace_default = workspace

    def _factory(workspace=None, permission_mode=None, work_dir=None, session_id=None):
        from agentica.cli.runtime import create_agent
        job_config = {
            key: value for key, value in agent_config.items()
            if key not in _SESSION_IDENTITY_KEYS
        }
        if work_dir is not None:
            job_config["work_dir"] = work_dir
        if session_id:
            job_config["session_id"] = session_id
        return create_agent(
            job_config,
            extra_tools=extra_tools or [],
            workspace=workspace if workspace is not None else workspace_default,
            skills_registry=skills_registry,
            # Unattended run with nobody to answer: a scheduled job simply does
            # not get the question tool, rather than getting it and having its
            # runtime invent a reply on the user's behalf.
            include_ask_user_question=False,
            # A scheduled job must not recursively kick off further immediate runs.
            enable_cron_immediate_run=False,
            enable_session_log=bool(session_id),
            permission_mode=permission_mode,
        )

    return _factory


def run_scheduler_loop(
    runner: CliAgentRunner,
    interval: int = 60,
    stop_event: Optional[threading.Event] = None,
    verbose: bool = False,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> None:
    """Blocking scheduler loop. Run me inside a daemon thread or a daemon process.

    Every ``interval`` seconds, call ``tick`` (which itself holds a file lock so
    concurrent CLI loops / daemons never double-execute a job). Errors in a tick
    are logged and surfaced via ``on_error`` but never crash the loop.

    Stops promptly when ``stop_event`` is set (checked every second).
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not (stop_event and stop_event.is_set()):
            try:
                loop.run_until_complete(tick(agent_runner=runner, verbose=verbose))
            except Exception as e:  # noqa: BLE001 - surface, do not crash loop
                logger.warning(f"cron tick error: {e}")
                if on_error:
                    on_error(e)
            # Sleep in 1s slices so stop_event is honored quickly.
            slept = 0
            while slept < interval and not (stop_event and stop_event.is_set()):
                time.sleep(1)
                slept += 1
    finally:
        loop.close()


def start_cron_thread(
    runner: CliAgentRunner,
    interval: int = 60,
    verbose: bool = False,
) -> tuple[threading.Thread, threading.Event]:
    """Start the scheduler loop in a daemon thread. Returns (thread, stop_event).

    Call ``stop_event.set()`` to request shutdown; the thread is a daemon so it
    will not block process exit either way.
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_scheduler_loop,
        kwargs={"runner": runner, "interval": interval,
                "stop_event": stop_event, "verbose": verbose},
        name="agentica-cron",
        daemon=True,
    )
    thread.start()
    logger.debug(f"cron scheduler thread started (interval={interval}s)")
    return thread, stop_event