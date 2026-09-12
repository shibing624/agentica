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
    agent = state.current_agent

    # This hook is the one place that knows whether another lap is coming, so it
    # is also the place that can say "the goal has stopped, the work is over".
    # The notify sink defers a goal-driven ``run.completed`` (a lap ending is
    # not the work ending) and needs to be told when the stopping point is
    # reached — see ``_release_deferred_completion``. Everything below can
    # return early; all of those returns mean "no more laps", so the release
    # must happen on every one of them.
    goal_was_active = mgr is not None and mgr.is_active()

    def _release_deferred_completion() -> None:
        if not goal_was_active:
            return
        try:
            from agentica.notify import goal_finished

            goal_finished(
                agent,
                session_id=getattr(agent, "session_id", None),
                work_dir=getattr(agent, "work_dir", None),
            )
        except Exception as exc:
            # Observation only: never let the sink's bookkeeping break the loop.
            from agentica.utils.log import logger

            logger.debug(f"notify sink: completion release failed: {exc}")

    if mgr is None or not goal_was_active:
        return

    if agent is None:
        return

    if agent._cancelled:
        mgr.pause(reason="user-interrupted")
        _cprint("  ⊙ Goal paused (user interrupted).")
        _release_deferred_completion()
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
        # A real user message outranks the next lap, so the goal is not driving
        # this session any more: the deferred "come back" is now true.
        _release_deferred_completion()
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
        # Nothing to judge, so no continuation is queued either.
        _release_deferred_completion()
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
            _release_deferred_completion()
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

    # Has the goal stopped for good? ``decision.status`` answers it, and nothing
    # else here can: "is another lap queued" is unreliable, because a queued
    # continuation still has to survive its own evaluation. Deliberately
    # pessimistic — the status is treated as "the goal may continue" unless it
    # is one of the terminal values, so an unrecognised status leaves the
    # completion held rather than announcing "you can come back" mid-goal.
    goal_has_ended = decision.status != "active"
    if goal_has_ended or not decision.should_continue:
        _release_deferred_completion()


__all__ = ['_maybe_continue_goal']
