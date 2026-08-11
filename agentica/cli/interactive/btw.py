# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: BTW side questions and background-completion notices
"""

from __future__ import annotations

from typing import List, Optional

from rich.markup import escape as rich_escape

from agentica.cli.commands.context import queue_ahead_of_goal_continuation
from agentica.cli.display.tool_format import _wrap_command_lines
from agentica.cli.runtime import _generate_session_id, get_console
from agentica.memory.models import AgentRun
from agentica.model.message import Message
from agentica.run_response import RunResponse
from agentica.tools.background_processes import (
    BackgroundProcessCompleted,
    read_log_tail,
)

from .console_io import _print_boxed_result
from .session_state import SessionState

def hand_to_agent(state: SessionState, pending_queue, text: str) -> None:
    """Give the agent text nobody typed, without interrupting its work.

    A running agent takes it through ``steer()``, which lands at the next
    tool-batch boundary. An idle one gets it as a queued turn. ``steer()``
    returning False means the run ended between the check and the call — the
    TOCTOU window ``Agent.steer`` documents — so the text falls through to the
    queue instead of being dropped.

    The queued form is tagged ``__RELAYED__`` because a queued turn is echoed as
    if the user had typed it, and nobody typed this. Both callers already print
    their own arrival block (a styled peer message, a finished-command report),
    so the raw model-facing text — headers, log tails and all — would be a
    second, uglier copy of what the user just read.

    ``relayed=True`` threads the same provenance through the steer buffer: if
    the text is accepted during the run's final inference and never drained,
    ``promote_late_steer`` re-queues it with the tag instead of as plain input.
    """
    agent = state.current_agent
    if state.agent_running and agent is not None and agent.steer(text, relayed=True):
        return
    pending_queue.put(("__RELAYED__", text))


def promote_late_steer(state: SessionState, pending_queue) -> List[str]:
    """Re-queue steering that outlived its run; return what was promoted.

    Text accepted by ``steer()`` during a run's final inference was buffered
    after the last drain, so the model never saw it. ``_end_steer_window``
    parks it on the agent rather than dropping it; called right after a run
    finishes, this turns it into ordinary queued input — ahead of any
    goal-continuation prompt, so the correction runs before the next automated
    lap. Provenance rides along: typed lines queue as plain ``str`` (the usual
    queued-turn echo), relayed lines go back tagged ``__RELAYED__`` — without
    the tag a parked peer/bg line could regain slash-command dispatch.
    """
    agent = state.current_agent
    if agent is None:
        return []
    late = agent.pop_undelivered_steer()
    for text, relayed in late:
        queue_ahead_of_goal_continuation(
            pending_queue, ("__RELAYED__", text) if relayed else text
        )
    return [text for text, _ in late]


def _background_result_for_agent(event: BackgroundProcessCompleted) -> str:
    """Render a finished background command as a report for the agent."""
    status = "finished" if event.returncode == 0 else "failed"
    if event.kind == "delegate":
        return _delegate_result_for_agent(event, status)
    lines = [
        f"[Background terminal #{event.num} ({event.id}) {status}: "
        f"exit {event.returncode} after {event.elapsed}]",
        f"Command: {event.command}",
        f"Log: {event.log_path}",
    ]
    tail = read_log_tail(event.log_path, max_lines=20, max_chars=4000)
    if tail:
        lines.extend(["", "Output tail:", tail])
    lines.extend(
        [
            "",
            "This is an automatic report of a command you started in the background. "
            "Pick the work back up where that command left off, or report the outcome "
            "to the user if nothing is left to do.",
        ]
    )
    return "\n".join(lines)


def _delegate_result_for_agent(event: BackgroundProcessCompleted, status: str) -> str:
    """Render a finished delegated session as a report for the agent that sent it.

    The worker ran with ``--print``, so its stdout is its final answer and not a
    command log — it gets a much larger slice than a background command's tail,
    because that answer is the entire deliverable.
    """
    answer = read_log_tail(event.log_path, max_lines=120, max_chars=8000)
    lines = [f'[Delegated task "{event.label}" {status} after {event.elapsed} ({event.id})]']
    if event.returncode == 0:
        lines.extend(["", "Its report:", answer or "(the worker produced no output)"])
        lines.extend(
            [
                "",
                "This is the whole of what that session hands back. Fold it into your "
                "own work, or report it to the user if nothing is left to do.",
            ]
        )
    else:
        lines.extend(
            [
                f"Exit code: {event.returncode}",
                f"Log: {event.log_path}",
                "",
                "Output tail:",
                answer or "(no output)",
            ]
        )
        lines.extend(
            [
                "",
                "The delegated session did not finish its task. Decide whether to do "
                "the work yourself or tell the user why it failed — do not simply "
                "delegate the same task again.",
            ]
        )
    return "\n".join(lines)


def _print_background_completion(event: BackgroundProcessCompleted) -> None:
    """Print a background-terminal completion notice.

    The command and log body are shown in full (wrapped to the terminal width).
    Folding behind Ctrl+O hid the identity of long overnight jobs.
    """
    con = get_console()
    ok = event.returncode == 0
    marker = "[green]✓[/green]" if ok else "[red]✗[/red]"
    status = "finished" if ok else "failed"
    delegated = event.kind == "delegate"
    con.print()
    if delegated:
        con.print(
            f'{marker} Delegated task "{rich_escape(event.label)}" {status} in '
            f"{event.elapsed} (exit {event.returncode})"
        )
    else:
        con.print(
            f"{marker} Background terminal #{event.num} {status} in {event.elapsed} "
            f"(exit {event.returncode})"
        )
    # The delegated command line is a `python -m agentica.cli.main --query <the
    # whole task>` — the label above already says what it was, so showing it
    # again as a wrapped shell command is noise.
    raw_command = "" if delegated else (event.command or "")
    if raw_command:
        try:
            width = int(getattr(con, "width", 80) or 80)
        except (TypeError, ValueError):
            width = 80
        width = max(20, width - 2)
        for line in _wrap_command_lines(raw_command, width):
            con.print(f"  {rich_escape(line)}")
    # Generous body: background jobs are the ones users stare at when they
    # finish, and a 5-line tail was throwing away the run. Still capped so a
    # multi-GB log cannot dump the terminal.
    body = read_log_tail(event.log_path, max_lines=500, max_chars=100_000)
    if body:
        for line in body.splitlines():
            con.print(f"  {rich_escape(line)}")
    con.print(f"  [dim]log: {rich_escape(event.log_path)}[/dim]")


def _run_btw_concurrent(agent, question: str, tui_state: dict):
    """Run a BTW side question in a background thread.

    Uses a fresh agent with NO tools but WITH a snapshot of the main agent's
    conversation history, so it can answer side questions in context.
    """
    try:
        from agentica import Agent

        # Snapshot conversation context from the main agent (same as /bg)
        context_snapshot = []
        if agent and agent.working_memory and agent.working_memory.messages:
            for msg in agent.working_memory.messages:
                if msg.role in ("user", "assistant") and msg.content:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if len(content) > 500:
                        content = content[:500] + "..."
                    context_snapshot.append(
                        Message.model_validate({"role": msg.role, "content": content})
                    )
            context_snapshot = context_snapshot[-10:]

        # Clone the parent model so the BTW agent owns isolated runtime state
        # (HTTP client, usage, metrics, error counters). Sharing the main
        # agent's model instance while it is streaming corrupts that
        # instance's state and breaks the main agent's subsequent turns — the
        # classic "/btw causes follow-up bugs" symptom. Same strategy as
        # Agent.clone() / SubagentRegistry.spawn().
        btw_model = None
        if agent and agent.model:
            from agentica.subagent import SubagentRegistry

            btw_model = SubagentRegistry._clone_parent_model(agent.model)

        btw_agent = Agent(
            model=btw_model,
            tools=[],
            instructions="You are a helpful assistant answering a quick side question. "
            "You have NO tools, NO skills, NO file access. "
            "Answer concisely based on your knowledge and conversation context.",
            session_id=_generate_session_id(),
            debug=False,
            add_history_to_context=True,
        )

        # Inject context snapshot so the BTW agent can see prior conversation
        if context_snapshot:
            synthetic_run = AgentRun(
                response=RunResponse(messages=context_snapshot),
            )
            btw_agent.working_memory.runs.append(synthetic_run)

        response = btw_agent.run_sync(question)
        result_text = str(response.content) if response and response.content is not None else "(no answer)"
    except Exception as e:
        result_text = f"Error: {e}"

    _print_boxed_result("BTW", question, result_text, color="cyan")


__all__ = ['hand_to_agent', 'promote_late_steer', '_background_result_for_agent', '_print_background_completion', '_run_btw_concurrent']
