# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Standing-goal slash commands: /goal and /subgoal
"""

from __future__ import annotations

import shlex
from typing import Any, Dict, List, Optional

from agentica.cli.commands.context import CommandContext
from agentica.cli.runtime import (
    get_console,
)
from agentica.goals import GoalManager
from agentica.run_context import TaskAnchor
from agentica.tools.goal_tool import GoalTool




# ==================== /goal & /subgoal ====================


def _attach_goal_tool(agent: Any) -> None:
    """Idempotently attach the GoalTool so the model can break the loop.

    Delegates to ``Agent.enable_goal_tool()`` (single source of truth shared
    with the SDK entry point ``Agent.run_goal``).
    """
    if agent is None or agent._session_log is None:
        return
    agent.enable_goal_tool()



def _detach_goal_tool(agent: Any) -> None:
    if agent is None or not agent.tools:
        return
    agent.tools = [t for t in agent.tools if not isinstance(t, GoalTool)]



def _sync_goal_budget_tui(tui_state: Optional[dict], mgr: Optional[GoalManager]) -> None:
    """Point the status bar's goal segment at the persisted GoalState.

    ``GoalState`` is the single source of truth for goal token spend; these
    two fields are only a display mirror. Passing ``mgr=None`` (or a manager
    with no live goal) hides the segment.
    """
    if tui_state is None:
        return
    state = mgr.load() if mgr is not None else None
    if state is None or state.status == "cleared":
        tui_state["goal_token_budget"] = None
        tui_state["goal_tokens_used"] = 0
        return
    tui_state["goal_token_budget"] = state.token_budget
    tui_state["goal_tokens_used"] = state.tokens_used



def _ensure_goal_manager(ctx: CommandContext) -> Optional[GoalManager]:
    """Return existing manager, or build one bound to the current agent's
    SessionLog. Returns None if the agent has no session_log (impossible in
    normal CLI flow, but keep defensive).
    """
    if ctx.goal_manager is not None:
        return ctx.goal_manager
    agent = ctx.current_agent
    if agent is None or agent._session_log is None:
        return None
    # Delegate to Agent.get_goal_manager() so CLI and SDK share one
    # construction path; the agent also caches the manager on itself.
    return agent.get_goal_manager()



def _parse_goal_set_args(raw: str) -> tuple[str, Dict[str, Any], Optional[str]]:
    """Parse /goal budget flags while keeping plain text fully compatible."""
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return "", {}, str(exc)

    budgets: Dict[str, Any] = {}
    objective: List[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"--turns", "--turn-budget"}:
            if i + 1 >= len(tokens):
                return "", {}, f"{token} requires an integer value"
            try:
                budgets["turn_budget"] = int(tokens[i + 1])
            except ValueError:
                return "", {}, f"{token} must be an integer"
            i += 2
            continue
        if token.startswith("--turns=") or token.startswith("--turn-budget="):
            value = token.split("=", 1)[1]
            try:
                budgets["turn_budget"] = int(value)
            except ValueError:
                return "", {}, "--turns must be an integer"
            i += 1
            continue
        if token == "--tokens":
            if i + 1 >= len(tokens):
                return "", {}, "--tokens requires an integer value"
            try:
                budgets["token_budget"] = int(tokens[i + 1])
            except ValueError:
                return "", {}, "--tokens must be an integer"
            i += 2
            continue
        if token.startswith("--tokens="):
            try:
                budgets["token_budget"] = int(token.split("=", 1)[1])
            except ValueError:
                return "", {}, "--tokens must be an integer"
            i += 1
            continue
        if token == "--wall":
            if i + 1 >= len(tokens):
                return "", {}, "--wall requires a number of seconds"
            try:
                budgets["wall_clock_budget_sec"] = float(tokens[i + 1])
            except ValueError:
                return "", {}, "--wall must be a number of seconds"
            i += 2
            continue
        if token.startswith("--wall="):
            try:
                budgets["wall_clock_budget_sec"] = float(token.split("=", 1)[1])
            except ValueError:
                return "", {}, "--wall must be a number of seconds"
            i += 1
            continue
        objective.append(token)
        i += 1

    for key, value in budgets.items():
        if value <= 0:
            return "", {}, f"{key} must be positive"
    return " ".join(objective).strip(), budgets, None



def _cmd_goal(ctx: CommandContext, cmd_args: str = ""):
    """
    /goal                  -> show status
    /goal status           -> show status (alias)
    /goal <objective>      -> set new objective + enqueue first turn
    /goal --turns 5 <objective>       -> set turn budget
    /goal --tokens 80000 <objective>  -> set token budget
    /goal --wall 1800 <objective>     -> set wall-clock budget seconds
    /goal pause            -> pause auto-continuation
    /goal resume           -> resume + enqueue continuation
    /goal clear            -> clear current goal
    """
    con = get_console()
    arg = (cmd_args or "").strip()

    mgr = _ensure_goal_manager(ctx)
    if mgr is None:
        con.print("  [yellow]No active agent / session log unavailable.[/yellow]")
        return

    sub = arg.lower()

    # ── status (default) ──
    if not arg or sub == "status":
        con.print(f"  {mgr.status_line()}")
        if mgr.load() is None:
            con.print("  [dim]Usage: /goal <objective>  |  pause | resume | clear[/dim]")
        return {"goal_manager": mgr}

    # ── pause / resume / clear are safe while agent is running ──
    if sub == "pause":
        state = mgr.pause("user")
        if state is None:
            con.print("  [dim]No goal to pause.[/dim]")
        else:
            con.print(f"  ⊙ Goal paused: {state.objective}")
        return {"goal_manager": mgr}

    if sub == "resume":
        if ctx.agent_running:
            con.print("  [yellow]Agent is currently running; goal will continue automatically.[/yellow]")
            return {"goal_manager": mgr}
        state = mgr.resume()
        if state is None:
            con.print("  [dim]No goal to resume.[/dim]")
            return {"goal_manager": mgr}
        if state.status != "active":
            con.print(f"  [dim]Goal status is {state.status}, cannot resume.[/dim]")
            return {"goal_manager": mgr}
        # Re-attach the tool (might have been detached by /clear race).
        _attach_goal_tool(ctx.current_agent)
        # Re-prime the loop with a continuation prompt.
        if ctx.pending_queue is not None:
            ctx.pending_queue.put(mgr.next_continuation_prompt())
        con.print(f"  ↻ Goal resumed: {state.objective}")
        return {"goal_manager": mgr}

    if sub == "clear":
        mgr.clear()
        _detach_goal_tool(ctx.current_agent)
        con.print("  ✗ Goal cleared.")
        return {"goal_manager": mgr}

    # ── set new objective ──
    if ctx.agent_running:
        con.print(
            "  [yellow]Cannot set a new goal while the agent is running. "
            "Wait for the current turn or use /goal pause/clear.[/yellow]"
        )
        return {"goal_manager": mgr}

    objective, budgets, parse_error = _parse_goal_set_args(arg)
    if parse_error:
        con.print(f"  [red]Invalid goal options: {parse_error}[/red]")
        con.print("  [dim]Usage: /goal [--turns N] [--tokens N] [--wall SECONDS] <objective>[/dim]")
        return {"goal_manager": mgr}
    try:
        state = mgr.set(objective, **budgets)
    except ValueError as exc:
        con.print(f"  [red]Invalid goal: {exc}[/red]")
        return {"goal_manager": mgr}

    # Overwrite the session's TaskAnchor so prompts.py + workspace retrieval
    # bind to the standing goal, not the latest user message.
    agent = ctx.current_agent
    if agent is not None:
        agent.task_anchor = TaskAnchor(
            goal=state.objective,
            source_query=state.objective,
            source="goal",
        )
        agent._anchor_session_id = agent.session_id

    # Workspace freeze timing: if workspace was already frozen on an earlier
    # query, retrieval will NOT re-bind to the new goal. Document the limit.
    if agent is not None and agent.workspace is not None:
        try:
            already_frozen = agent.workspace.get_frozen_context() is not None
        except Exception:
            already_frozen = False
        if already_frozen:
            con.print(
                "  [dim]ℹ Workspace memory was already frozen against an earlier query. "
                "Goal-bound retrieval will activate from the next /new session.[/dim]"
            )

    # Attach the goal tool (verify_completion + update_goal) so the model can
    # verify completion with evidence and break the loop when actually done.
    _attach_goal_tool(agent)

    budget_bits = []
    if state.token_budget is not None:
        budget_bits.append(f"{state.token_budget:,} tokens")
    if state.turn_budget is not None:
        budget_bits.append(f"{state.turn_budget} turns")
    if state.wall_clock_budget_sec is not None:
        budget_bits.append(f"{state.wall_clock_budget_sec:.0f}s wall")
    budget_label = ", ".join(budget_bits) if budget_bits else "no caps"
    con.print(f"  ⊙ Goal set ({budget_label}): {state.objective}")

    # Kick off the first turn so the user doesn't need to send a follow-up.
    if ctx.pending_queue is not None:
        ctx.pending_queue.put(state.objective)

    return {"goal_manager": mgr}



def _cmd_subgoal(ctx: CommandContext, cmd_args: str = ""):
    """
    /subgoal               -> list subgoals
    /subgoal <text>        -> add a subgoal to active goal
    /subgoal remove <n>    -> remove the n-th subgoal (1-based)
    /subgoal clear         -> drop all subgoals
    """
    con = get_console()
    arg = (cmd_args or "").strip()

    mgr = _ensure_goal_manager(ctx)
    state = mgr.load() if mgr is not None else None
    if mgr is None or state is None:
        con.print("  [yellow]No active goal — set one with /goal first.[/yellow]")
        return

    if not arg:
        if not state.subgoals:
            con.print("  [dim]No subgoals.[/dim]")
        else:
            con.print(f"  Subgoals ({len(state.subgoals)}):")
            for i, sg in enumerate(state.subgoals, 1):
                con.print(f"    {i}. {sg}")
        con.print("  [dim]Usage: /subgoal <text>  |  remove <n>  |  clear[/dim]")
        return {"goal_manager": mgr}

    parts = arg.split(maxsplit=1)
    sub = parts[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""

    if sub == "remove" or sub == "rm":
        if not rest.isdigit():
            con.print("  [dim]Usage: /subgoal remove <number>[/dim]")
            return {"goal_manager": mgr}
        removed = mgr.remove_subgoal(int(rest))
        if removed is None:
            con.print(f"  [red]Invalid subgoal index: {rest}[/red]")
        else:
            con.print(f"  ✗ Removed subgoal: {removed}")
        return {"goal_manager": mgr}

    if sub == "clear":
        n = mgr.clear_subgoals()
        con.print(f"  ✗ Cleared {n} subgoal(s).")
        return {"goal_manager": mgr}

    # Default: add the whole argument as a subgoal.
    try:
        text = mgr.add_subgoal(arg)
    except ValueError as exc:
        con.print(f"  [red]{exc}[/red]")
        return {"goal_manager": mgr}
    con.print(f"  + Subgoal added: {text}")
    return {"goal_manager": mgr}
