# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: ``update_goal`` model tool for the standing-goal loop.

When the standing-goal loop is active (see ``agentica/goals.py``), the
external judge runs after every turn to decide whether the agent's last
response satisfies the user's objective. That extra LLM call costs tokens
and can be wrong in two ways:

1. The agent already knows it is done — the judge call is pure overhead.
2. The agent is BLOCKED (needs user input). The judge sees "I'm waiting
   for you to clarify X" and rules ``not done``, causing the runtime to
   auto-loop forever.

``update_goal`` lets the model itself break the loop. It is intentionally
narrow — the model can ONLY mark ``complete`` or ``paused``; it cannot
rewrite the objective, change subgoals, or clear the goal. Those remain
under user control.

The tool writes through ``SessionLog.append_goal()`` so the persisted
state is authoritative; the caller's ``GoalManager.evaluate_after_turn()``
re-reads from disk and respects the tool's decision (skipping the judge).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentica.goals import GoalState
from agentica.tools.base import Tool
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.memory.session_log import SessionLog


class GoalTool(Tool):
    """Receive-only model tool: lets the agent signal goal completion or
    block on user input. Bound to a specific ``SessionLog`` instance.
    """

    def __init__(self, session_log: "SessionLog"):
        super().__init__(name="goal_tool")
        self._session_log = session_log
        self.register(self.update_goal)

    async def update_goal(self, status: str, reason: str = "") -> str:
        """Mark the user's standing goal as complete or paused.

        Use ``status="complete"`` ONLY when the goal is actually finished
        and evidence is in your last message. Do NOT call for partial
        progress.

        Use ``status="paused"`` when you are blocked and need new input
        from the user (e.g. ambiguous requirements, missing credentials).
        This stops the auto-continuation loop without losing progress.

        Args:
            status: ``"complete"`` or ``"paused"``.
            reason: One-sentence rationale shown to the user.
        """
        status = (status or "").strip().lower()
        if status not in ("complete", "paused"):
            return (
                f"Invalid status '{status}'. Must be 'complete' or 'paused'. "
                f"You cannot rewrite the objective; only the user can change it."
            )

        payload = self._session_log.load_goal()
        if payload is None:
            return "No standing goal is set — nothing to update."
        if payload.get("status") != "active":
            return (
                f"Goal is not active (current status: {payload.get('status')}). "
                f"Cannot update."
            )

        payload["status"] = status
        payload["last_verdict"] = "tool_signal"
        payload["last_reason"] = reason or f"agent-marked {status}"
        payload["updated_at"] = time.time()
        if status == "paused":
            payload["paused_reason"] = "agent-tool"

        try:
            state = GoalState.from_dict(payload)
            self._session_log.append_goal(state)
        except (TypeError, ValueError) as exc:
            logger.warning("GoalTool.update_goal serialization failed: %s", exc)
            return f"Failed to update goal: {exc}"

        if status == "complete":
            return f"Goal marked complete: {reason or 'agent confirmed done'}"
        return f"Goal paused: {reason or 'agent reported blocked'}"
