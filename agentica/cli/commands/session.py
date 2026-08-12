# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Session slash commands: history, resume, rename, compact, export
"""

from __future__ import annotations

import collections
import json
import os
import shlex
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from agentica.cli.runtime import (
    get_console,
    create_agent,
)
from agentica.cli.display import (
    format_session_summary,
    print_header,
    resumable_session_id,
)
from agentica.cli.session_resume import (
    choose_resume_work_dir,
    enter_work_dir,
    find_sessions_by_id,
)
from agentica.cli.setup import apply_named_profile_to_agent_config
from agentica.global_config import set_project_profile
from agentica.agent.history_filter import strip_tool_artifacts_from_memory
from agentica.goals import GoalManager
from agentica.memory.models import AgentRun
from agentica.memory.session_log import SessionLog
from agentica.model.message import Message
from agentica.run_response import RunResponse
from agentica.utils.log import logger
from agentica.cli.context_usage import measure_context

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands.helpers import _run_async_safe


@dataclass(frozen=True)
class HistoryRenderStats:
    """Counts shared by resume output and the history command."""

    run_count: int = 0
    message_count: int = 0
    tool_call_count: int = 0
    tool_result_count: int = 0
    tool_result_chars: int = 0
    tool_error_count: int = 0



def _canonical_history_runs(agent) -> list[AgentRun]:
    """Return the run history that the next model request actually consumes."""
    working_memory = agent.working_memory
    if working_memory.runs:
        return list(working_memory.runs)
    if working_memory.messages:
        return [
            AgentRun(
                response=RunResponse(
                    messages=[message.model_copy(deep=True) for message in working_memory.messages]
                )
            )
        ]
    return []



def _messages_for_run(run: AgentRun) -> list[Message]:
    if run.response is not None and run.response.messages:
        return list(run.response.messages)
    if run.messages:
        return list(run.messages)
    if run.message is not None:
        return [run.message]
    return []



def _tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    return function.get("name") or tool_call.get("name") or "tool"



def _history_stats(runs: list[AgentRun]) -> HistoryRenderStats:
    message_count = 0
    tool_call_count = 0
    tool_result_count = 0
    tool_result_chars = 0
    tool_error_count = 0
    visible_runs = 0

    for run in runs:
        messages = _messages_for_run(run)
        if not messages:
            continue
        visible_runs += 1
        message_count += len(messages)
        for message in messages:
            if message.role == "assistant":
                tool_call_count += len(message.tool_calls or [])
            elif message.role == "tool":
                tool_result_count += 1
                tool_result_chars += len(message.get_content_string())
                if message.tool_call_error is True:
                    tool_error_count += 1

    return HistoryRenderStats(
        run_count=visible_runs,
        message_count=message_count,
        tool_call_count=tool_call_count,
        tool_result_count=tool_result_count,
        tool_result_chars=tool_result_chars,
        tool_error_count=tool_error_count,
    )



def _format_char_count(char_count: int) -> str:
    if char_count < 1000:
        return f"{char_count} chars"
    return f"{char_count / 1000:.1f}K chars"



def _run_tool_activity(
    messages: list[Message],
) -> tuple[collections.Counter, list[Message], int]:
    call_names: list[str] = []
    tool_results: list[Message] = []
    for message in messages:
        if message.role == "assistant":
            call_names.extend(_tool_call_name(call) for call in (message.tool_calls or []))
        elif message.role == "tool":
            tool_results.append(message)

    result_names = [message.tool_name or "tool" for message in tool_results]
    names = call_names if call_names else result_names
    return collections.Counter(names), tool_results, len(call_names)



def _display_run_tool_summary(con, run_number: int, messages: list[Message]) -> None:
    tool_names, tool_results, tool_call_count = _run_tool_activity(messages)
    if not tool_names and not tool_results:
        return

    name_summary = ", ".join(
        f"{name}x{count}" for name, count in tool_names.most_common()
    )
    call_count = tool_call_count or sum(tool_names.values())
    errors = [message for message in tool_results if message.tool_call_error is True]
    summary = f"[Tools - run {run_number}] {call_count} calls"
    if name_summary:
        summary += f": {name_summary}"
    summary += f" - {len(tool_results)} results hidden"
    if errors:
        summary += f" - {len(errors)} errors"
    con.print(f"\n  {summary}", style="dim", markup=False, highlight=False)

    for message in errors[:3]:
        preview = " ".join(message.get_content_string().split()) or "(empty result)"
        if len(preview) > 160:
            preview = preview[:157] + "..."
        con.print(
            f"    ! {message.tool_name or 'tool'}: {preview}",
            style="yellow",
            markup=False,
            highlight=False,
        )
    if len(errors) > 3:
        con.print(f"    ... {len(errors) - 3} more errors hidden", style="dim")



def display_conversation_history(runs: list[AgentRun], title: str) -> HistoryRenderStats:
    """Render conversation text while collapsing persisted tool activity by run."""
    stats = _history_stats(runs)
    if stats.message_count == 0:
        return stats

    con = get_console()
    con.print(f"\n[bold cyan]{title}[/bold cyan]")
    if stats.tool_result_count:
        summary = (
            f"Conversation view - {stats.tool_result_count} tool results "
            f"({_format_char_count(stats.tool_result_chars)}) collapsed"
        )
        if stats.tool_error_count:
            summary += f" - {stats.tool_error_count} errors"
        summary += " - /history tools [run] for details"
        con.print(summary, style="dim", markup=False, highlight=False)

    for run_number, run in enumerate(runs, start=1):
        messages = _messages_for_run(run)
        if not messages:
            continue
        has_tool_activity = any(
            message.role == "tool" or bool(message.tool_calls)
            for message in messages
        )
        tool_activity_seen = False
        tool_summary_shown = False

        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                tool_activity_seen = True
                continue

            content_text = message.get_content_string()
            if message.role == "user":
                con.print(f"\n[bold cyan]You - run {run_number}[/bold cyan]")
                con.print(content_text, markup=False, highlight=False)
                continue

            if message.role == "assistant":
                if content_text:
                    if tool_activity_seen and not tool_summary_shown:
                        _display_run_tool_summary(con, run_number, messages)
                        tool_summary_shown = True
                    con.print(f"\n[bold green]Agent - run {run_number}[/bold green]")
                    con.print(content_text, markup=False, highlight=False)
                if message.tool_calls:
                    tool_activity_seen = True

        if has_tool_activity and not tool_summary_shown:
            _display_run_tool_summary(con, run_number, messages)

    con.print()
    return stats



def _format_tool_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return json.dumps(arguments, ensure_ascii=False, indent=2)



def format_tool_history(runs: list[AgentRun], run_number: int | None = None) -> str:
    """Build full persisted tool-call and tool-result text for pager display."""
    if run_number is not None:
        selected = [(run_number, runs[run_number - 1])]
    else:
        selected = list(enumerate(runs, start=1))

    sections: list[str] = []
    for current_run, run in selected:
        lines = [f"=== Run {current_run} ==="]
        tool_entries = 0
        for message in _messages_for_run(run):
            if message.role == "assistant":
                for tool_call in message.tool_calls or []:
                    tool_entries += 1
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments", tool_call.get("arguments", {}))
                    call_id = tool_call.get("id") or tool_call.get("tool_call_id") or ""
                    lines.append(f"\nTool call: {_tool_call_name(tool_call)}")
                    if call_id:
                        lines.append(f"Call ID: {call_id}")
                    if arguments not in (None, "", {}):
                        lines.append("Arguments:")
                        lines.append(_format_tool_arguments(arguments))
            elif message.role == "tool":
                tool_entries += 1
                status = "error" if message.tool_call_error is True else "ok"
                lines.append(f"\nTool result: {message.tool_name or 'tool'} [{status}]")
                if message.tool_call_id:
                    lines.append(f"Call ID: {message.tool_call_id}")
                lines.append(message.get_content_string())
        if tool_entries:
            sections.append("\n".join(lines))

    return "\n\n".join(sections)



def _cmd_history(ctx: CommandContext, cmd_args: str = ""):
    """Display conversation history or open full tool history in a pager."""
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation history yet.[/yellow]")
        return
    runs = _canonical_history_runs(agent)
    if not runs:
        con.print("[yellow]No conversation history yet.[/yellow]")
        return

    args = shlex.split(cmd_args)
    if not args:
        display_conversation_history(runs, "Conversation History")
        return
    if args[0].lower() != "tools" or len(args) > 2:
        con.print("[red]Usage: /history [tools [run-number]][/red]")
        return

    run_number = None
    if len(args) == 2:
        try:
            run_number = int(args[1])
        except ValueError:
            con.print("[red]Run number must be an integer.[/red]")
            return
        if run_number < 1 or run_number > len(runs):
            con.print(f"[red]Run number must be between 1 and {len(runs)}.[/red]")
            return

    content = format_tool_history(runs, run_number)
    if not content:
        target = f"run {run_number}" if run_number is not None else "this session"
        con.print(f"[yellow]No tool activity in {target}.[/yellow]")
        return
    if ctx.open_pager_callback is None:
        con.print("[yellow]Tool history pager is only available in interactive mode.[/yellow]")
        return
    title = (
        f"Tool history - run {run_number}"
        if run_number is not None
        else "Tool history"
    )
    ctx.open_pager_callback(title, content)



def _cmd_newchat(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    old_agent = ctx.current_agent
    tui_state = ctx.tui_state or {}
    started_at = tui_state.get("session_started_at", time.monotonic())
    con.print(
        format_session_summary(
            elapsed_seconds=time.monotonic() - started_at,
            usage=old_agent.model.usage,
            session_id=resumable_session_id(old_agent),
        )
    )
    # `agentica resume <id>` pins a session (and possibly another project's
    # storage) into agent_config. A new chat must not inherit either, or it
    # would keep appending to the session it was supposed to leave behind.
    ctx.agent_config.pop("session_id", None)
    ctx.agent_config.pop("session_base_dir", None)
    current_agent = create_agent(
        ctx.agent_config,
        ctx.extra_tools,
        ctx.workspace,
        ctx.skills_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
        background_process_registry=ctx.background_processes,
        peer_session=ctx.peer_session,
    )
    print_header(
        ctx.agent_config.get("model_provider", ""),
        ctx.agent_config.get("model_name", ""),
        work_dir=ctx.agent_config.get("work_dir"),
        extra_tools=ctx.extra_tool_names,
    )
    # Drop any goal manager — the new session has a new SessionLog.
    return {
        "current_agent": current_agent,
        "goal_manager": None,
        "session_started_at": time.monotonic(),
    }



def _resume_base_dir(ctx: CommandContext) -> Optional[str]:
    """Resolve the sessions directory the CLI operates on.

    Prefer the active agent's live ``SessionLog.base_dir`` so the list is scoped
    by the same project (work_dir) + user the agent writes to. Falling back to
    ``None`` lets ``SessionLog`` derive the default from the process cwd, which
    for the CLI equals the current project.
    """
    agent = ctx.current_agent
    log = agent._session_log if agent is not None else None
    return str(log.base_dir) if log is not None else None


def _resume_user_id(ctx: CommandContext) -> Optional[str]:
    """User whose sessions ``/resume`` may search across projects."""
    agent = ctx.current_agent
    return agent.user_id if agent is not None else ctx.agent_config.get("user_id")



def hydrate_resumed_session(agent, resume_at: str | None = None) -> tuple[list[dict[str, Any]], int]:
    """Load a session log into the prompt history used by subsequent runs."""
    session_log = agent._session_log
    if session_log is None or not session_log.exists():
        return [], 0
    _model = getattr(agent, "model", None)
    model_id = getattr(_model, "id", None) if _model is not None else None
    resumed = session_log.load(
        resume_at=resume_at,
        model=model_id,
    )
    # Reasonix CacheState: tell the user whether the resumed prefix is likely
    # still warm in the provider cache before the first request proves it.
    logger.info(
        "resume %s: cache state estimate = %s",
        session_log.session_id,
        session_log.cache_warmth_hint(model_id),
    )
    agent.working_memory.clear()
    runs_built = agent.working_memory.hydrate_runs_from_history(resumed) if resumed else 0
    if runs_built and agent.model is not None and not agent.model.supports_replayed_tool_history:
        strip_tool_artifacts_from_memory(agent.working_memory)
    return resumed, runs_built



def display_resumed_transcript(runs: list[AgentRun], session_label: str) -> HistoryRenderStats:
    """Display resumed conversation text with tool activity collapsed by run."""
    return display_conversation_history(runs, f"Resumed transcript: {session_label}")



def _split_resume_at(args_str: str) -> tuple[str, Optional[str]]:
    """Split ``<session> at <uuid>`` into its two halves."""
    session_target, separator, at_candidate = args_str.rpartition(" at ")
    if not (separator and session_target.strip() and at_candidate.strip()):
        return args_str, None
    try:
        UUID(at_candidate.strip())
    except ValueError:
        return args_str, None
    return session_target.strip(), at_candidate.strip()


def _print_session_list(
    ctx: CommandContext,
    con,
    sessions: list[dict[str, Any]],
    *,
    title: str,
    usage_hint: str,
    show_work_dir: bool = False,
) -> None:
    """Render the resume picker and remember what each number pointed at.

    The numbering has to survive until the follow-up ``/resume <n>``, otherwise
    a list that spans projects would renumber under the user's feet.
    """
    shown = sessions[:10]
    con.print(f"\n[bold]{title}[/bold]\n")
    for i, s in enumerate(shown, 1):
        ts_str = s.get("last_timestamp", "") or ""
        if ts_str:
            ts_str = ts_str[:16].replace("T", " ")
        size_kb = s["size_bytes"] / 1024
        sid = s["session_id"]
        # Show a clean, copy-pasteable 8-char prefix that /resume accepts
        # directly. Avoid the old "abc...wxyz" form which users would copy
        # verbatim (ellipsis included) and which then failed to match.
        short_id = sid if len(sid) <= 12 else sid[:8]
        # Prefer the user-set `/rename` label; otherwise show the first
        # user message that started the session.
        preview = SessionLog.session_preview(s["path"])
        turns = preview["user_count"]
        first_user = preview["first_user"]
        user_name = s.get("name")
        if user_name:
            # Named session: name is the headline, preview is the subline.
            summary = user_name[:80]
            subline = " ".join(first_user.split())[:80] if first_user else "(no messages yet)"
        elif first_user:
            # Unnamed session: keep the legacy single-line preview.
            summary = " ".join(first_user.split())[:80]
            subline = None
        else:
            summary = "(empty session)"
            subline = None
        is_current = ctx.current_agent is not None and sid == ctx.current_agent.session_id
        current_marker = "  [green](current)[/green]" if is_current else ""
        con.print(
            f"  {i}. [cyan]{short_id}[/cyan]  {ts_str}  "
            f"({size_kb:.0f}KB, {turns} turns){current_marker}"
        )
        if user_name:
            con.print(f"     [bold]{summary}[/bold]")
            if subline:
                con.print(f"     [dim]> {subline}[/dim]")
        else:
            con.print(f"     [dim]> {summary}[/dim]")
        if show_work_dir and s.get("work_dir"):
            con.print(f"     [dim]{s['work_dir']}[/dim]")
    con.print(f"\n[dim]{usage_hint}[/dim]")
    if ctx.tui_state is not None:
        ctx.tui_state["resume_picker"] = shown


def _select_session(
    ctx: CommandContext,
    con,
    args_str: str,
    sessions: list[dict[str, Any]],
    visible_sessions: list[dict[str, Any]],
    user_id: Optional[str],
) -> Optional[dict[str, Any]]:
    """Resolve ``/resume <arg>`` to one session, or report why it could not."""
    named_matches = [
        session
        for session in visible_sessions
        if isinstance(session.get("name"), str)
        and session["name"].casefold() == args_str.casefold()
    ]
    if args_str.isdecimal():
        # Numbers refer to the listing the user last saw, which may have spanned
        # every project (`/resume all`), not just this one.
        picker = (ctx.tui_state or {}).get("resume_picker") or visible_sessions
        index = int(args_str) - 1
        if 0 <= index < len(picker):
            return picker[index]
        if len(named_matches) == 1:
            return named_matches[0]
        con.print("[red]Invalid number.[/red]")
        return None
    if len(named_matches) == 1:
        return named_matches[0]
    if len(named_matches) > 1:
        con.print(
            f"[red]Ambiguous: multiple sessions are named '{args_str}'. Use the number or id prefix.[/red]"
        )
        return None

    # Accept the exact id, any unique prefix, or the truncated
    # "7154826e...0358" form printed by the picker. Archived sessions stay
    # reachable by id even though the picker hides them, and a miss here falls
    # through to every other project of the same user.
    matching = find_sessions_by_id(args_str, sessions, user_id=user_id)
    if not matching:
        con.print(f"[red]No session matching '{args_str}'[/red]")
        return None
    if len(matching) > 1:
        con.print(
            f"[red]Ambiguous: '{args_str}' matches {len(matching)} sessions. Use a longer prefix or the number.[/red]"
        )
        return None
    return matching[0]


def _cmd_resume(ctx: CommandContext, cmd_args: str = ""):
    """Resume a previous session from JSONL log."""
    con = get_console()

    # Scope the session list by project (work_dir) + user, exactly like the Web
    # sidebar, so both entrypoints show a consistent set for the same project.
    # The running agent already carries the correctly-scoped work_dir/user_id;
    # fall back to the process cwd (which for the CLI equals the project).
    base_dir = _resume_base_dir(ctx)
    user_id = _resume_user_id(ctx)

    args_str, resume_at_uuid = _split_resume_at((cmd_args or "").strip())

    if args_str.casefold() == "all":
        every = [
            s
            for project in SessionLog.list_projects(user_id=user_id)
            for s in SessionLog.list_sessions(
                base_dir=project["base_dir"], work_dir=project["work_dir"]
            )
            if not s.get("archived")
        ]
        every.sort(key=lambda s: s["mtime"], reverse=True)
        if not every:
            con.print("[yellow]No sessions found to resume.[/yellow]")
            return
        _print_session_list(
            ctx,
            con,
            every,
            title="Sessions across all projects:",
            usage_hint="Usage: /resume <number|id-prefix> — resuming a session from "
            "another directory asks which directory to work in.",
            show_work_dir=True,
        )
        return

    sessions = SessionLog.list_sessions(base_dir=base_dir)
    # Archived sessions are hidden from the picker (same "I don't want to see
    # this anymore" semantic as the Web UI sidebar), but an explicit id/prefix
    # match still searches the full unfiltered `sessions` list so an archived
    # session remains directly resumable by id.
    visible_sessions = [s for s in sessions if not s.get("archived")]

    if args_str:
        chosen = _select_session(ctx, con, args_str, sessions, visible_sessions, user_id)
        if chosen is None:
            return

        current_work_dir = ctx.agent_config.get("work_dir") or os.getcwd()
        choice = choose_resume_work_dir(
            chosen.get("work_dir"),
            current_work_dir,
            asker=ctx.ask_user_question_callback,
            printer=con.print,
        )
        if choice.cancelled:
            con.print("[yellow]Resume cancelled.[/yellow]")
            return

        agent_config = dict(ctx.agent_config)
        agent_config["session_id"] = chosen["session_id"]
        agent_config["_resume_at_uuid"] = resume_at_uuid
        # Pin storage to where the transcript already lives. Without this a
        # session resumed from elsewhere would be looked up in the current
        # project, come up empty, and silently start over.
        agent_config["session_base_dir"] = chosen["base_dir"]
        if choice.work_dir:
            if not enter_work_dir(choice.work_dir):
                con.print(f"[red]Cannot enter {choice.work_dir}; resume aborted.[/red]")
                return
            agent_config["work_dir"] = choice.work_dir
            con.print(f"[dim]Working directory: {choice.work_dir}[/dim]")
        if chosen.get("profile_name"):
            profile_name = chosen["profile_name"]
            try:
                apply_named_profile_to_agent_config(agent_config, profile_name, source="session")
            except ValueError as exc:
                agent_config["_skip_session_profile_persist"] = True
                con.print(
                    f"[yellow]Session profile '{profile_name}' is unavailable: {exc}. "
                    "Using the current profile instead.[/yellow]"
                )
            else:
                set_project_profile(agent_config.get("work_dir") or os.getcwd(), profile_name)
        current_agent = create_agent(
            agent_config,
            ctx.extra_tools,
            ctx.workspace,
            ctx.skills_registry,
            ask_user_question_callback=ctx.ask_user_question_callback,
            background_process_registry=ctx.background_processes,
            peer_session=ctx.peer_session,
        )

        # Eagerly load history into working_memory so /status, /context etc.
        # reflect the resumed state immediately (do not wait for the next _run
        # to lazily replay). Applies to both plain resume and `resume ... at <uuid>`.
        resumed, runs_built = hydrate_resumed_session(current_agent, resume_at_uuid)
        # Keep the model-visible context complete, but render a conversation
        # view so persisted tool payloads do not flood terminal scrollback.
        session_name = chosen.get("name")
        session_label = f"{session_name} ({chosen['session_id']})" if session_name else chosen["session_id"]
        display_stats = display_resumed_transcript(current_agent.working_memory.runs, session_label)
        if resume_at_uuid is None and resumed:
            con.print(
                "[dim]Tip: `/fork` branches this conversation into a new session; "
                "`/fork list` shows the message ids to branch at.[/dim]"
            )

        if resume_at_uuid:
            # create_agent turned `at <uuid>` into a real fork, so the work
            # continues in a new session and the original branch stays intact.
            con.print(
                f"[green]Forked {session_label} at {resume_at_uuid[:8]} → new session "
                f"{current_agent.session_id} — restored {runs_built} runs into context; "
                f"showing conversation only "
                f"({display_stats.tool_result_count} tool results collapsed)[/green]"
            )
        else:
            con.print(
                f"[green]Resumed session: {session_label}"
                f" — restored {runs_built} runs into context; showing conversation only "
                f"({display_stats.tool_result_count} tool results collapsed)[/green]"
            )

        # If the resumed session had an active goal, demote to paused for
        # safety — automatic continuation on resume is too surprising
        # without token-budget guards (P0).
        resumed_goal_manager = None
        if current_agent._session_log is not None:
            judge_model = current_agent.auxiliary_model or current_agent.model
            resumed_goal_manager = GoalManager(current_agent._session_log, judge_model=judge_model)
            state = resumed_goal_manager.load()
            if state is not None:
                if state.status == "active":
                    resumed_goal_manager.force_pause_on_resume()
                    con.print(f"  [yellow]⊙ Standing goal detected and paused for safety:[/yellow] {state.objective}")
                    con.print("  [dim]Use /goal resume to continue working on it.[/dim]")
                elif state.status in ("paused", "complete"):
                    con.print(f"  [dim]⊙ Previous goal ({state.status}): {state.objective}[/dim]")

        result = {"current_agent": current_agent, "goal_manager": resumed_goal_manager}
        if choice.work_dir:
            result["work_dir"] = choice.work_dir
        return result
    else:
        if not visible_sessions:
            con.print(
                "[yellow]No sessions in this directory.[/yellow] "
                "[dim]Use `/resume all` to list sessions from every project.[/dim]"
            )
            return
        _print_session_list(
            ctx,
            con,
            visible_sessions,
            title="Available sessions:",
            usage_hint=(
                f"Usage: /resume <number|name|id-prefix> "
                f"(e.g. /resume {visible_sessions[0]['session_id'][:8]})  ·  "
                f"/resume all lists every project"
            ),
        )
        return



def _cmd_rename(ctx: CommandContext, cmd_args: str = ""):
    """Rename the active session so it is easy to identify in `/resume`."""
    con = get_console()
    agent = ctx.current_agent
    if agent is None or not agent.session_id or agent._session_log is None:
        con.print("[yellow]No active session to rename.[/yellow]")
        return

    new_name = (cmd_args or "").strip()
    if not new_name:
        current_name = agent._session_log.get_name()
        session_label = f"{current_name} ({agent.session_id})" if current_name else agent.session_id
        con.print(f"  Current session: [cyan]{session_label}[/cyan]")
        con.print("  [dim]Usage: /rename <name>[/dim]")
        return

    try:
        agent._session_log.set_name(new_name)
    except OSError as error:
        con.print(f"  [red]Failed to rename session: {error}[/red]")
        return
    con.print(
        f"  [green]Renamed session[/green] [dim]{agent.session_id}[/dim] "
        f"[green]to[/green] [cyan]{new_name}[/cyan]"
    )



def _cmd_clear(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    os.system("clear" if os.name != "nt" else "cls")
    current_agent = create_agent(
        ctx.agent_config,
        ctx.extra_tools,
        ctx.workspace,
        ctx.skills_registry,
        ask_user_question_callback=ctx.ask_user_question_callback,
        background_process_registry=ctx.background_processes,
        peer_session=ctx.peer_session,
    )
    print_header(
        ctx.agent_config["model_provider"],
        ctx.agent_config["model_name"],
        work_dir=ctx.agent_config.get("work_dir"),
        extra_tools=ctx.extra_tool_names,
    )
    con.print("[info]Screen cleared and conversation reset.[/info]")
    return {"current_agent": current_agent, "goal_manager": None}



def _cmd_compact(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    agent = ctx.current_agent
    if not agent or not agent.working_memory:
        con.print("[yellow]No conversation history to compact.[/yellow]")
        return

    messages = agent.working_memory.messages
    msg_count = len(messages)
    if msg_count == 0:
        con.print("[yellow]No messages to compact.[/yellow]")
        return

    custom_instructions = cmd_args.strip() if cmd_args else None
    model = agent.model
    wm = agent.working_memory

    # Same data-loss boundary as the runner: flush memory/experience buffers
    # before the transcript is replaced by a summary.
    hooks = agent._run_hooks
    if hooks is not None:
        _run_async_safe(hooks.on_pre_compact(agent=agent, messages=messages))

    native_compacted = False
    if model.supports_native_compaction:
        con.print(f"[dim]Compacting {msg_count} messages with the provider-native endpoint...[/dim]")
        try:
            result = _run_async_safe(
                model.compact_context(messages, instructions=custom_instructions)
            )
            if result is None:
                raise RuntimeError("model advertised native compaction but returned no checkpoint")
        except Exception as error:
            logger.warning(
                "Native compaction failed (%s); falling back to local compaction", error
            )
        else:
            messages[-1].provider_checkpoint = result.checkpoint
            wm.collapse_runs(messages)
            if agent._session_log is not None:
                agent._session_log.append_provider_checkpoint(result.checkpoint)
            native_compacted = True
            con.print(
                f"[green]Context compacted with {model.id}; portable history remains available.[/green]"
            )

    cm = agent.tool_config.compression_manager if agent.tool_config else None
    if not native_compacted:
        if cm is None:
            con.print("[red]No compression manager on this agent; nothing to compact with.[/red]")
            return
        con.print(f"[dim]Compacting {msg_count} messages with LLM summary...[/dim]")
        compacted = _run_async_safe(
            cm.auto_compact(
                messages,
                model=model,
                force=True,
                working_memory=wm,
                custom_instructions=custom_instructions,
            )
        )
        if not compacted:
            # auto_compact only rewrites the list once it holds a summary, so a
            # False return means nothing moved. Saying so and stopping is the
            # whole answer: the fallback this replaces "succeeded" by clearing
            # the message list — system prompt included — and stitching a
            # 300-char-per-message digest back in its place, which is a worse
            # transcript than the one it destroyed.
            con.print("[red]Compaction failed; conversation left unchanged.[/red]")
            return
        wm.collapse_runs(messages)
        con.print(f"[green]Context compacted: {msg_count} messages -> {len(messages)} summary.[/green]")

    if hooks is not None:
        _run_async_safe(hooks.on_post_compact(agent=agent, messages=messages))

    if ctx.tui_state is not None:
        breakdown = _run_async_safe(measure_context(agent))
        ctx.tui_state["context_tokens"] = breakdown.total
        ctx.tui_state["context_window"] = breakdown.window
    con.print("[dim]Workspace memory preserved.[/dim]")



def _cmd_export(ctx: CommandContext, cmd_args: str = ""):
    """Save conversation history to a JSON file (excludes system prompts)."""
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation to save.[/yellow]")
        return

    messages = agent.working_memory.messages
    export_msgs = []
    for msg in messages:
        if msg.role == "system":
            continue
        content = msg.content or ""
        if isinstance(content, list):
            content = str(content)
        if isinstance(content, str):
            content = content.strip()
        entry = {"role": msg.role, "content": content}
        if msg.tool_calls:
            entry["tool_calls"] = len(msg.tool_calls)
        export_msgs.append(entry)

    if not export_msgs:
        con.print("[yellow]No messages to save.[/yellow]")
        return

    filename = cmd_args.strip() if cmd_args.strip() else f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if not filename.endswith(".json"):
        filename += ".json"

    model_name = f"{ctx.agent_config.get('model_provider', '')}/{ctx.agent_config.get('model_name', '')}"

    data = {
        "model": model_name,
        "session_id": agent.session_id,
        "exported_at": datetime.now().isoformat(),
        "messages": export_msgs,
    }
    Path(filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    con.print(f"  [green]Saved {len(export_msgs)} messages to {filename}[/green]")



def _cmd_retry(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation to retry.[/yellow]")
        return
    wm = agent.working_memory
    last_user_msg = None
    for msg in reversed(wm.messages):
        if msg.role == "user":
            last_user_msg = msg
            break
    if last_user_msg is None or not last_user_msg.content:
        con.print("[yellow]No user message found to retry.[/yellow]")
        return
    user_text = last_user_msg.content if isinstance(last_user_msg.content, str) else str(last_user_msg.content)
    if wm.runs:
        wm.runs.pop()
    while wm.messages and wm.messages[-1].role in ("assistant", "tool"):
        wm.messages.pop()
    if wm.messages and wm.messages[-1].role == "user":
        wm.messages.pop()
    if ctx.pending_queue is not None:
        ctx.pending_queue.put(user_text)
        preview = user_text[:60] + ("..." if len(user_text) > 60 else "")
        con.print(f"  [green]Retrying: {preview}[/green]")



def _cmd_undo(ctx: CommandContext, cmd_args: str = ""):
    con = get_console()
    agent = ctx.current_agent
    if not agent:
        con.print("[yellow]No conversation history.[/yellow]")
        return
    wm = agent.working_memory
    if not wm.messages:
        con.print("[yellow]No messages to undo.[/yellow]")
        return
    if wm.runs:
        wm.runs.pop()
    removed = 0
    while wm.messages and wm.messages[-1].role in ("assistant", "tool"):
        wm.messages.pop()
        removed += 1
    if wm.messages and wm.messages[-1].role == "user":
        wm.messages.pop()
        removed += 1
    if removed > 0:
        con.print(f"  [green]Undone last exchange ({removed} messages removed).[/green]")
    else:
        con.print("[yellow]Nothing to undo.[/yellow]")
