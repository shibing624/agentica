# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Standalone cron daemon entry point (`agentica cron daemon`).

Runs the cron scheduler in the FOREGROUND, decoupled from the interactive CLI.
Useful for keeping scheduled jobs running on a server or in a tmux/screen pane
without an interactive session. The same file lock used by the in-CLI scheduler
prevents double-execution when both run at once.
"""

import threading


def run_cron_daemon(args, console) -> None:
    """Run the cron scheduler loop in the foreground until interrupted.

    Resolves model config from the active config.yaml profile, builds a fresh
    agent per job, and ticks every ``args.interval`` seconds. Ctrl-C stops it.
    """
    if getattr(args, "cron_command", None) != "daemon":
        console.print("[red]Usage: agentica cron daemon [--interval N] [--verbose][/red]")
        return

    interval = getattr(args, "interval", 60) or 60
    verbose = bool(getattr(args, "verbose", False))

    # The standalone daemon reads the active profile directly (no interactive
    # arg parsing). If no profile exists yet, tell the user to run setup.
    from agentica.global_config import get_profile
    profile = get_profile()
    if not profile or not profile.get("model_provider"):
        console.print("[red]No model profile configured. Run `agentica setup` first.[/red]")
        return

    agent_config = {
        "model_provider": profile.get("model_provider"),
        "model_name": profile.get("model_name"),
        "base_url": profile.get("base_url"),
        "api_key": profile.get("api_key"),
        "max_tokens": profile.get("max_tokens"),
        "temperature": profile.get("temperature"),
        "reasoning_effort": profile.get("reasoning_effort"),
        "top_p": profile.get("top_p"),
        "context_window": profile.get("context_window"),
        "debug": verbose,
    }

    from agentica.cron.cli_runner import (
        CliAgentRunner, build_cli_agent_factory, run_scheduler_loop,
    )
    from agentica.cron import jobs as cronjobs

    factory = build_cli_agent_factory(agent_config)
    runner = CliAgentRunner(factory)

    n_jobs = len(cronjobs.list_jobs())
    console.print(f"[bold]Agentica cron daemon[/bold] — interval={interval}s, "
                  f"{n_jobs} job(s) loaded.")
    console.print("[dim]Press Ctrl-C to stop.[/dim]")

    stop_event = threading.Event()
    try:
        run_scheduler_loop(runner, interval=interval, stop_event=stop_event,
                           verbose=verbose)
    except KeyboardInterrupt:
        stop_event.set()
        console.print("\n[dim]cron daemon stopped.[/dim]")