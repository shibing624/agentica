# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Post-turn standing-goal continuation hook
"""

from __future__ import annotations

from agentica.cli.commands.context import PendingQueue
from agentica.cli.commands.goal import _detach_goal_tool, _sync_goal_budget_tui
from agentica.cli.commands.helpers import _run_async_safe
from agentica.goals import is_goal_generated_prompt

from .attachments import unpack_queue_payload
from .console_io import _cprint
from .session_state import SessionState

# ==================== Goal loop hook ====================


def _maybe_continue_goal(
    state: SessionState,
    pending_queue: PendingQueue,
    tui_state: dict,
) -> None:
    """After each agent turn, decide whether to enqueue a continuation prompt.

    Invariants:
    - Real user input ALWAYS preempts the goal loop. If any non-continuation,
      non-internal item is already queued, we defer.
    - A cancelled agent (Ctrl+C) pauses the goal instead of evaluating —
      otherwise the judge sees a half-finished response, judges "not done",
      and the user's cancel immediately gets re-queued.
    - Empty response: skip (nothing to judge).
    - GoalManager.evaluate_after_turn() is async; we bridge with _run_async_safe.
    - token_delta is read from CostTracker totals diffed against the
      pre-turn baseline; elapsed comes from tui_state["last_turn_seconds"].
    """
    mgr = state.goal_manager
    if mgr is None or not mgr.is_active():
        return

    agent = state.current_agent
    if agent is None:
        return

    if agent._cancelled:
        mgr.pause(reason="user-interrupted")
        _cprint("  ⊙ Goal paused (user interrupted).")
        return

    # User real input takes priority.
    loop_prompt_pending = False
    for item, _ts in pending_queue.peek_all_with_timestamps():
        queued = unpack_queue_payload(item)
        # An ephemeral side question runs beside the goal, so the loop does not
        # stand aside for it. Anything else pending — including a peer message or
        # a finished job's report — outranks another lap.
        text = "" if queued.is_btw else queued.text
        if not text or text.startswith("__"):
            continue
        if is_goal_generated_prompt(text):
            loop_prompt_pending = True
            continue
        return  # real user message waiting — let it run first

    # Extract per-turn signals (final text, token delta, tool pairs) via the
    # SAME shared helper the SDK ``run_goal_step()`` uses, so the CLI /goal
    # loop and ``Agent.run_goal()`` never drift on how a turn is measured.
    # The CLI keeps its own outer shell (user-input preemption above, Ctrl+C
    # pause, continuation queueing below) — only the per-turn evaluation is
    # shared. ``goal_tokens_baseline`` persists the accumulated total across
    # turns of this /goal session (each turn's per-run delta added on).
    final_text, token_delta, new_baseline, tool_pairs = mgr.extract_turn_signals(
        agent.run_response, state.goal_tokens_baseline
    )
    if not final_text.strip():
        return
    state.goal_tokens_baseline = new_baseline

    elapsed_sec = float(tui_state.get("last_turn_seconds", 0.0) or 0.0)

    with state.goal_lock:
        try:
            decision = _run_async_safe(
                mgr.evaluate_after_turn(
                    final_text,
                    token_delta=token_delta,
                    elapsed_sec=elapsed_sec,
                    tool_calls=tool_pairs or None,
                )
            )
        except Exception as exc:
            _cprint(f"  [goal] evaluator failed: {exc}")
            return

    # Replace the live mid-turn estimate with the charged total the manager
    # just persisted, so the status bar settles on the authoritative number.
    _sync_goal_budget_tui(tui_state, mgr)

    if decision.message:
        _cprint(f"  {decision.message}")

    # If the loop ended (complete / paused / budget_limited), detach the
    # tool — otherwise it lingers on a goal that no longer auto-continues.
    if decision.status in ("complete", "paused", "budget_limited"):
        _detach_goal_tool(agent)
        if decision.status == "complete":
            _sync_goal_budget_tui(tui_state, None)

    # ``loop_prompt_pending`` guards the case where a /steer (or /goal resume)
    # jumped a turn in front of an already-queued continuation: this turn was
    # the interjection, and the continuation behind it still stands. Enqueuing
    # a second one would run the same next step twice.
    if decision.should_continue and decision.continuation_prompt and not loop_prompt_pending:
        pending_queue.put(decision.continuation_prompt)


__all__ = ['_maybe_continue_goal']
