# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Cron slash command and scheduled-job helpers
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from agentica.cli.runtime import (
    get_console,
)
from agentica.run_response import AgentCancelledError

from agentica.cli.commands.context import CommandContext




def _fmt_ms(ms) -> str:
    """Format an epoch-millis timestamp as 'YYYY-MM-DD HH:MM:SS', or '-' if falsy."""

    if not ms:
        return "-"
    try:
        return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ms)



def _fmt_next_run(job) -> str:
    """Human-friendly 'next run' time for a job, or '-' when not scheduled."""
    return _fmt_ms(getattr(job, "next_run_at_ms", None))



def _confirm_via_tui(ctx: CommandContext, question: str) -> bool:
    """TUI-safe yes/no confirmation for a command handler.

    Command handlers run on the background ``process_loop`` thread while
    prompt_toolkit owns the main thread. A nested ``pt_prompt`` there spins up
    a second Application that fights the live one — it deadlocks and leaks CPR
    escapes (``^[[..R``) into the input line, so the user can never answer.
    Instead route through the SAME ``ask_user_question_callback`` the agent's
    ``ask_user_question`` uses: it arms ``state.input_request`` and the main
    thread renders the inline prompt + feeds the typed answer back.

    Returns False when there is no interactive channel (non-TUI / tests) — the
    safe default for a destructive confirmation.
    """
    cb = ctx.ask_user_question_callback
    if cb is None:
        return False
    try:
        ans = cb(f"{question} (yes/no)", ["yes", "no"])
    except AgentCancelledError:
        # Ctrl+C at the prompt == "no" (safe default for a destructive op).
        return False
    return str(ans).strip().lower() in ("yes", "y", "是", "确认", "确定")



def _ask_text_via_tui(ctx: CommandContext, prompt: str, default: str = "") -> str:
    """TUI-safe free-text input for a command handler.

    Same rationale as :func:`_confirm_via_tui`: a nested ``pt_prompt`` on the
    background thread deadlocks the live prompt_toolkit app. Route through the
    ``ask_user_question_callback`` instead. Returns ``default`` when there is no
    interactive channel (non-TUI / tests) or when the user cancels (Ctrl+C).
    """
    cb = ctx.ask_user_question_callback
    if cb is None:
        return default
    try:
        ans = cb(prompt, None)
    except AgentCancelledError:
        return default
    return (ans or "").strip()



_CRON_PROMPT_REFINE_SYSTEM = """You rewrite a user's rough scheduled-task description into a single, concrete, self-contained execution prompt that an autonomous agent will run UNATTENDED on every tick (no human is watching at run time).

Rules:
- Output ONLY the rewritten prompt text — no preamble, no quotes, no explanation.
- Keep the user's original language.
- Make it unambiguous and directly actionable in one run: spell out the exact action, target path/format, and any naming convention implied by the description.
- Do NOT re-implement scheduling/recurrence — the cron system already handles "every N minutes". Describe only what to do in ONE run.
- Never ask the user questions in the prompt; it must be executable without clarification.
- Keep it concise (1-3 sentences)."""



async def _refine_cron_prompt(model, raw_prompt: str, schedule_human: str) -> str:
    """One-shot LLM rewrite of a cron task prompt. Returns "" on any failure."""
    from agentica.model.message import Message

    user = (
        f"Rough task description: {raw_prompt}\n"
        f"Schedule (handled by the system, do not re-implement): {schedule_human}\n\n"
        "Rewrite it into the unattended per-run execution prompt."
    )
    resp = await model.response(
        messages=[
            Message(role="system", content=_CRON_PROMPT_REFINE_SYSTEM),
            Message(role="user", content=user),
        ]
    )
    return (resp.content or "").strip()



def _cmd_cron(ctx: CommandContext, cmd_args: str = ""):
    """Manage scheduled (cron) jobs.

    /cron                          list all jobs
    /cron add "<prompt>" <schedule>  create a job (schedule: cron expr / 'every 5m' / ISO time)
    /cron edit <id> prompt "<text>"  change a job's execution prompt
    /cron edit <id> schedule <sched> change a job's schedule
    /cron pause <id>               pause a job
    /cron resume <id>              resume a job
    /cron remove <id>              delete a job
    /cron runs [<id>]              show recent run history
    /cron run <id>                 run a job once now
    /cron daemon on|off|status     control the in-CLI scheduler thread
    """
    con = get_console()
    from agentica.cron import jobs as cronjobs

    args = cmd_args.strip()
    sub = args.split()[0].lower() if args else "list"
    rest = args[len(sub) :].strip() if args else ""

    if sub in ("list", "ls", ""):
        all_jobs = cronjobs.list_jobs()
        if not all_jobs:
            con.print('[dim]No cron jobs. Add one with: /cron add "<prompt>" <schedule>[/dim]')
            return
        con.print("[bold]Cron jobs[/bold]")
        for j in all_jobs:
            status = getattr(getattr(j, "status", None), "value", getattr(j, "status", "?"))
            human = cronjobs.schedule_to_human(j.schedule)
            con.print(
                f"  [cyan]{j.id}[/cyan]  [yellow]{j.name}[/yellow]  "
                f"[{'green' if status == 'active' else 'dim'}]{status}[/]  "
                f"next: {_fmt_next_run(j)}  [dim]({human})[/dim]"
            )
        return

    if sub == "add":
        # Parse: /cron add "prompt with spaces" <schedule>
        import shlex

        try:
            tokens = shlex.split(rest)
        except ValueError:
            con.print('[red]Could not parse. Use: /cron add "<prompt>" <schedule>[/red]')
            return
        if len(tokens) < 2:
            con.print('[red]Usage: /cron add "<prompt>" <schedule>[/red]')
            con.print("[dim]schedule examples: '0 9 * * *' (9am daily), 'every 30m', '2026-07-01T09:00:00'[/dim]")
            return
        prompt = tokens[0]
        schedule = " ".join(tokens[1:])
        # Validate the schedule up front so we don't burn an LLM refine call on
        # a job that can't be created anyway.
        try:
            parsed = cronjobs.parse_schedule(schedule)
        except Exception as e:
            con.print(f"[red]{e}[/red]")
            return
        human = cronjobs.schedule_to_human(parsed)

        # Add-time refinement + confirmation. A cron job runs UNATTENDED, so the
        # right moment to resolve ambiguity is NOW (the human is here), not at
        # first execution. Refine the rough prompt into a concrete per-run
        # execution prompt with the auxiliary model, then let the user confirm /
        # keep original / hand-write. Interactive path only (needs both a model
        # and the TUI input channel); non-TUI / tests fall straight through.
        final_prompt = prompt
        agent = ctx.current_agent
        if agent is not None and ctx.ask_user_question_callback is not None:
            con.print("[dim]Refining the task prompt with the model…[/dim]")
            refined = ""
            try:
                model = agent.resolve_auxiliary_model("cron_refine")
                refined = asyncio.run(_refine_cron_prompt(model, prompt, human))
            except Exception as e:
                con.print(f"[dim]Refine skipped ({e}); using the original prompt.[/dim]")
            if refined and refined != prompt:
                con.print(f"\n  [bold]Original:[/bold] {prompt}")
                con.print(f"  [bold]Refined :[/bold] {refined}\n")
                opt_refined = "Use the refined prompt (recommended)"
                opt_original = "Keep the original prompt"
                opt_manual = "Write it myself"
                try:
                    choice = str(
                        ctx.ask_user_question_callback(
                            "Which prompt should this scheduled job run?",
                            [opt_refined, opt_original, opt_manual],
                        )
                    ).strip()
                except AgentCancelledError:
                    con.print("[dim]cancelled — job not created[/dim]")
                    return
                if choice == opt_original:
                    final_prompt = prompt
                elif choice == opt_manual:
                    try:
                        typed = ctx.ask_user_question_callback("Enter the execution prompt:", None)
                    except AgentCancelledError:
                        con.print("[dim]cancelled — job not created[/dim]")
                        return
                    final_prompt = (typed or "").strip() or prompt
                else:
                    final_prompt = refined

        try:
            # Keep the user's own words as the display name; run the (possibly
            # refined) prompt under the hood.
            job = cronjobs.create_job(prompt=final_prompt, schedule=schedule, name=prompt[:50])
        except Exception as e:
            con.print(f"[red]Failed to create job: {e}[/red]")
            return
        con.print(f"[green]Created job [cyan]{job.id}[/cyan] '{job.name}'[/green] next: {_fmt_next_run(job)}")
        if final_prompt != prompt:
            con.print(f"[dim]execution prompt: {final_prompt}[/dim]")
        if not (ctx.tui_state and ctx.tui_state.get("cron_is_running", lambda: False)()):
            con.print("[yellow]Scheduler is OFF — job won't run until you enable it: /cron daemon on[/yellow]")
        return

    if sub in ("pause", "resume", "remove", "rm", "delete"):
        if not rest:
            con.print(f"[red]Usage: /cron {sub} <id>[/red]")
            return
        job_id = rest.split()[0]
        try:
            if sub == "pause":
                cronjobs.pause_job(job_id)
                con.print(f"[green]Paused {job_id}[/green]")
            elif sub == "resume":
                cronjobs.resume_job(job_id)
                con.print(f"[green]Resumed {job_id}[/green]")
            else:
                if _confirm_via_tui(ctx, f"Delete cron job {job_id}?"):
                    cronjobs.remove_job(job_id)
                    con.print(f"[green]Removed {job_id}[/green]")
                else:
                    con.print("[dim]cancelled[/dim]")
        except Exception as e:
            con.print(f"[red]{e}[/red]")
        return

    if sub == "edit":
        # /cron edit <id> prompt "<text>"   |   /cron edit <id> schedule <schedule>
        parts = rest.split(maxsplit=2)
        if len(parts) < 3 or parts[1].lower() not in ("prompt", "schedule"):
            con.print('[red]Usage: /cron edit <id> prompt "<text>"  |  /cron edit <id> schedule <schedule>[/red]')
            return
        job_id, field_name, value = parts[0], parts[1].lower(), parts[2].strip()
        if cronjobs.get_job(job_id) is None:
            con.print(f"[red]No job {job_id}[/red]")
            return
        try:
            if field_name == "prompt":
                new_prompt = value.strip('"').strip("'")
                cronjobs.update_job(job_id, {"prompt": new_prompt, "name": new_prompt[:50]})
                con.print(f"[green]Updated prompt for {job_id}[/green]")
                con.print(f"[dim]execution prompt: {new_prompt}[/dim]")
            else:  # schedule
                parsed = cronjobs.parse_schedule(value)
                next_run = cronjobs.compute_next_run_at_ms(parsed)
                cronjobs.update_job(job_id, {"schedule": parsed, "next_run_at_ms": next_run or 0})
                con.print(f"[green]Updated schedule for {job_id}: {cronjobs.schedule_to_human(parsed)}[/green]")
        except Exception as e:
            con.print(f"[red]{e}[/red]")
        return

    if sub == "runs":
        job_id = rest.split()[0] if rest else None
        runs = cronjobs.list_task_runs(job_id=job_id)
        if not runs:
            con.print("[dim]No run history.[/dim]")
            return
        con.print("[bold]Recent runs[/bold]")
        for r in runs[:20]:
            st = r.status.value if hasattr(r.status, "value") else str(r.status)
            when = _fmt_ms(r.started_at_ms)
            color = "green" if st == "ok" else ("yellow" if st == "timeout" else "red")
            con.print(f"  [cyan]{r.task_id}[/cyan]  [{color}]{st}[/]  {when}")
        return

    if sub == "run":
        if not rest:
            con.print("[red]Usage: /cron run <id>[/red]")
            return
        job_id = rest.split()[0]
        job = cronjobs.get_job(job_id)
        if not job:
            con.print(f"[red]No job {job_id}[/red]")
            return
        con.print(f"[dim]Running job {job_id} once now...[/dim]")
        try:
            from agentica.cron.scheduler import _execute_job
            from agentica.cron.cli_runner import CliAgentRunner, build_cli_agent_factory

            factory = build_cli_agent_factory(ctx.agent_config, ctx.extra_tools, ctx.workspace, ctx.skills_registry)
            runner = CliAgentRunner(factory)
            asyncio.run(_execute_job(job, agent_runner=runner, verbose=False))
            con.print(f"[green]Job {job_id} executed.[/green]")
        except Exception as e:
            con.print(f"[red]Run failed: {e}[/red]")
        return

    if sub == "daemon":
        action = rest.split()[0].lower() if rest else "status"
        ts = ctx.tui_state or {}
        is_running = ts.get("cron_is_running", lambda: False)()
        if action == "status":
            from agentica.global_config import get_setting

            enabled = bool(get_setting("cron.enabled", False))
            interval = int(get_setting("cron.interval", 60) or 60)
            con.print(
                f"  this session:   [{'green' if is_running else 'dim'}]{'RUNNING' if is_running else 'STOPPED'}[/]"
            )
            con.print(
                f"  config (persisted): cron.enabled="
                f"[{'green' if enabled else 'red'}]{enabled}[/]  interval={interval}s"
            )
            # The live thread only reflects THIS process. Explain a mismatch so
            # "status can't be seen" never looks like a silent failure.
            if enabled and not is_running:
                con.print(
                    "  [yellow]Enabled in config but no scheduler thread in this "
                    "session[/yellow] — a separate `agentica cron daemon` process may be "
                    "running it (the file lock prevents double execution), or the thread "
                    "failed to start. Run [cyan]/cron daemon on[/cyan] to start it here."
                )
            elif not enabled and is_running:
                con.print(
                    "  [yellow]Running in this session but disabled in config[/yellow] — "
                    "it will not auto-start next launch. Run [cyan]/cron daemon on[/cyan] to persist."
                )
            return
        if action == "on":
            from agentica.global_config import set_setting

            set_setting("cron.enabled", True)  # persist so it survives restart
            start = ts.get("cron_start")
            if start and start():
                con.print("[green]Scheduler started (and enabled in config).[/green]")
            else:
                con.print("[yellow]Enabled in config; could not start thread in this session. Restart CLI.[/yellow]")
            return
        if action == "off":
            from agentica.global_config import set_setting

            set_setting("cron.enabled", False)
            stop = ts.get("cron_stop")
            if stop:
                stop()
            con.print("[green]Scheduler stopped (and disabled in config).[/green]")
            return
        con.print("[red]Usage: /cron daemon on|off|status[/red]")
        return

    con.print(f"[red]Unknown subcommand '{sub}'. See /cron for usage.[/red]")
