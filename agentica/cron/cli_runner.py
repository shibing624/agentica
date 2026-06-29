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
from typing import Any, Callable, Dict, Optional

from loguru import logger

from agentica.cron.scheduler import tick


class CliAgentRunner:
    """Adapt a CLI agent factory to the cron ``AgentRunner`` protocol.

    Args:
        agent_factory: zero-arg callable returning a fresh agent instance.
            A new agent is built per job run to isolate state.
    """

    def __init__(self, agent_factory: Callable[[], Any]):
        self._factory = agent_factory

    async def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        agent = self._factory()
        response = await agent.run(message=prompt)
        # RunResponse.content is the final text; fall back to str for safety.
        return getattr(response, "content", None) or ""


def build_cli_agent_factory(agent_config: Dict[str, Any], extra_tools=None,
                            workspace=None, skills_registry=None) -> Callable[[], Any]:
    """Return a factory that builds a fresh CLI agent from the given config.

    Imported lazily to avoid a circular import (cli.config imports tools which
    may import cron).
    """
    def _factory():
        from agentica.cli.config import create_agent
        return create_agent(
            agent_config,
            extra_tools=extra_tools or [],
            workspace=workspace,
            skills_registry=skills_registry,
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