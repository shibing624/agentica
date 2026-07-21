# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Cron runtime glue for the CLI.

Provides:
- ``CliAgentRunner``: adapts the CLI's agent configuration to the cron
  ``AgentRunner`` protocol. Each cron job runs on a FRESH agent instance built
  from the same config, so a scheduled task never pollutes (or is polluted by)
  the interactive session's conversation state.
- ``run_scheduler_loop``: a blocking loop that periodically calls
  ``cron.scheduler.tick``. Designed to run inside a daemon thread (the CLI is
  thread-based, not asyncio-based), where it spins its own event loop. Also
  reused by the standalone ``agentica cron daemon`` subcommand.
"""

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from agentica.utils.log import logger

from agentica.cron.scheduler import tick


def _noninteractive_input_callback(prompt: str, options: Optional[List[str]] = None) -> str:
    """Answer ``ask_user_question`` for an UNATTENDED cron run.

    A scheduled job runs on a background scheduler thread (in-CLI daemon) or a
    headless ``agentica cron daemon`` process — there is no human watching to
    type an answer. The tool's default is a bare ``input()`` that blocks
    forever there (it also deadlocks against prompt_toolkit's stdin ownership
    when the daemon thread lives inside the interactive CLI). That is exactly
    the reported hang.

    Instead of blocking we return a short notice so the agent stops asking and
    proceeds with its best judgment. For ``confirm`` prompts the tool
    normalises any non-yes answer to "no", which is the safe default for an
    unattended run (never take an unapproved irreversible action).
    """
    return (
        "[non-interactive scheduled run: no user is available to answer. "
        "Do not ask further questions — pick a reasonable default and complete "
        "the task, noting any assumptions in your final result.]"
    )


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

    async def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        # Honor a per-job workspace from the cron run context instead of
        # silently dropping it. The scheduler passes a path string (or None);
        # wrap it in a Workspace. None -> the CLI's own default workspace.
        ws = context.get("workspace")
        if isinstance(ws, str) and ws:
            from agentica.workspace import Workspace
            ws = Workspace(path=ws)
        agent = self._factory(workspace=ws)
        response = await agent.run(message=prompt)
        # RunResponse.content is the final text; fall back to "" for safety.
        return getattr(response, "content", None) or ""


def build_cli_agent_factory(agent_config: Dict[str, Any], extra_tools=None,
                            workspace=None, skills_registry=None) -> Callable[..., Any]:
    """Return a factory that builds a fresh CLI agent from the given config.

    The returned factory accepts an optional ``workspace`` override (used by
    cron jobs that target a specific workspace); when not provided it falls back
    to the default captured here. Imported lazily to avoid a circular import
    (cli.config imports tools which may import cron).
    """
    workspace_default = workspace

    def _factory(workspace=None):
        from agentica.cli.runtime import create_agent
        return create_agent(
            agent_config,
            extra_tools=extra_tools or [],
            workspace=workspace if workspace is not None else workspace_default,
            skills_registry=skills_registry,
            # Unattended run: never block on stdin for a human answer.
            ask_user_question_callback=_noninteractive_input_callback,
            # A scheduled job must not recursively kick off further immediate runs.
            enable_cron_immediate_run=False,
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