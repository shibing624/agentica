# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Standing-goal model tools: ``update_goal`` (pause/block) and
``verify_completion`` (agent-driven, evidence-backed completion check).

When the standing-goal loop is active (see ``agentica/goals.py``), the loop
keeps handing the objective back to the agent until something declares the
goal complete. Historically that "something" was an external LLM judge that
ran **after every turn** — costing tokens on turns where the agent was
obviously nowhere near done, and deciding completion by optimistic
inspection rather than objective evidence.

This module replaces that with an **agent-driven, on-demand** verification
tool:

- ``verify_completion`` — the agent calls this **only when it believes the
  goal may be done**. The tool runs a real acceptance check inside itself:
    * ``mode="test"``     — run ``verify_command`` (e.g. ``pytest -q``);
                            exit code 0 == green == complete. stdout/stderr
                            are returned so a red run feeds the failure back
                            to the agent to keep fixing.
    * ``mode="criteria"`` — ask the judge model whether ``acceptance_criteria``
                            are satisfied by ``summary`` (the old judge, now
                            invoked by the agent's choice instead of every
                            turn).
  Only a green result marks the goal complete; a red result leaves the goal
  active and returns the gap so the loop continues.

- ``update_goal`` — narrow control channel retained for the **paused**
  case: the agent is blocked and needs user input. It can still mark
  ``complete`` directly as an escape hatch, but ``verify_completion`` is the
  preferred, evidence-backed path.

Both tools write through ``SessionLog.append_goal()`` so the persisted state
is authoritative; ``GoalManager.evaluate_after_turn()`` re-reads from disk
and respects the tool's decision (skipping the judge).
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from agentica.goals import GoalState, judge_goal
from agentica.tools.base import Tool
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.memory.session_log import SessionLog
    from agentica.model.base import Model

# Cap how much command output we feed back to the agent. Full pytest logs can
# be huge; the tail almost always carries the failing assertions.
_OUTPUT_TAIL_CHARS = 4000
# Default wall-clock cap for a verify command. A test suite that hangs should
# surface as a red result, not stall the whole goal loop.
_DEFAULT_VERIFY_TIMEOUT_SEC = 300.0


def _tail(text: str, limit: int = _OUTPUT_TAIL_CHARS) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return "…(truncated)…\n" + text[-limit:]


class GoalTool(Tool):
    """Model tools for the standing-goal loop, bound to one ``SessionLog``.

    Exposes ``verify_completion`` (evidence-backed, agent-driven completion
    check) and ``update_goal`` (pause-on-block / manual escape hatch).
    """

    def __init__(
        self,
        session_log: "SessionLog",
        judge_model: Optional["Model"] = None,
        *,
        work_dir: Optional[str] = None,
    ):
        super().__init__(name="goal_tool")
        self._session_log = session_log
        # Needed for ``verify_completion(mode="criteria")``. When None, the
        # criteria mode returns an explicit error telling the agent to use
        # ``mode="test"`` with a concrete verify_command instead.
        self._judge_model = judge_model
        self._work_dir = work_dir
        self.register(self.verify_completion)
        self.register(self.update_goal)

    # ------------------------------------------------------------------ helpers
    def _load_active_payload(self):
        """Return the active goal payload, or an error string if none active."""
        payload = self._session_log.load_goal()
        if payload is None:
            return None, "No standing goal is set — nothing to verify."
        if payload.get("status") != "active":
            return None, f"Goal is not active (current status: {payload.get('status')}). Cannot update."
        return payload, None

    def _mark_complete(self, payload: dict, reason: str, final_answer: str = "") -> None:
        payload["status"] = "complete"
        payload["last_verdict"] = "verify_tool"
        payload["last_reason"] = reason
        payload["updated_at"] = time.time()
        if final_answer and final_answer.strip():
            payload["final_answer"] = final_answer.strip()
        state = GoalState.from_dict(payload)
        self._session_log.append_goal(state)

    # ------------------------------------------------------------------ verify
    async def verify_completion(
        self,
        mode: str,
        summary: str = "",
        verify_command: str = "",
        acceptance_criteria: str = "",
        final_answer: str = "",
    ) -> str:
        """Verify the standing goal is actually complete, then mark it done.

        Call this ONLY when you believe the goal may be finished — not every
        turn. The tool runs a real acceptance check and only marks the goal
        complete if that check passes. A failing check returns the concrete
        gap so you can keep working.

        Choose ``mode`` by the task type:

        - ``mode="test"`` (preferred for code tasks): pass ``verify_command``
          — a shell command that PASSES (exit code 0) exactly when the goal
          is met, e.g. ``"pytest tests/test_foo.py -q"``. WRITE the test
          cases first so there is something to run. If the command exits
          non-zero the goal stays active and you get stdout/stderr back to
          fix the failures.

        - ``mode="criteria"`` (for non-code deliverables like reports):
          pass ``acceptance_criteria`` (the checklist the output must meet)
          and ``summary`` (what you produced / where it is). A judge model
          checks the criteria against your summary.

        Always pass ``final_answer`` with the complete substantive
        deliverable — it is captured as the goal's result and survives any
        chat text you write afterwards.

        Args:
            mode: ``"test"`` or ``"criteria"``.
            summary: What you did / the deliverable, used by criteria mode
                and shown to the user.
            verify_command: Shell command for test mode (exit 0 == pass).
            acceptance_criteria: Checklist for criteria mode.
            final_answer: The complete result to capture as the goal output.

        Returns:
            A message. On pass: confirms the goal is marked complete. On
            fail: reports why (command output or unmet criteria) — the goal
            stays active so the loop continues.
        """
        mode = (mode or "").strip().lower()
        if mode not in ("test", "criteria"):
            return (
                f"Invalid mode '{mode}'. Use mode='test' with a verify_command "
                f"(exit 0 == pass), or mode='criteria' with acceptance_criteria."
            )

        payload, err = self._load_active_payload()
        if err is not None:
            return err

        if mode == "test":
            return await self._verify_test(payload, verify_command, summary, final_answer)
        return await self._verify_criteria(payload, acceptance_criteria, summary, final_answer)

    async def _verify_test(self, payload: dict, verify_command: str, summary: str, final_answer: str) -> str:
        verify_command = (verify_command or "").strip()
        if not verify_command:
            return (
                "mode='test' requires verify_command — a shell command that exits 0 "
                "only when the goal is met (e.g. 'pytest tests/ -q'). Write the tests first."
            )
        try:
            proc = await asyncio.create_subprocess_shell(
                verify_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._work_dir,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=_DEFAULT_VERIFY_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return (
                    f"Verification command timed out after {_DEFAULT_VERIFY_TIMEOUT_SEC:.0f}s: "
                    f"`{verify_command}`. Goal stays active — make the check terminate faster "
                    f"or narrow it, then verify again."
                )
        except Exception as exc:
            logger.warning("verify_completion(test) failed to run command: %s", exc)
            return (
                f"Could not run verify_command `{verify_command}`: {exc}. "
                f"Goal stays active. Fix the command and verify again."
            )

        exit_code = proc.returncode
        stdout = _tail(stdout_b.decode("utf-8", errors="replace"))
        stderr = _tail(stderr_b.decode("utf-8", errors="replace"))

        if exit_code == 0:
            reason = f"verify_command passed (exit 0): {verify_command}"
            self._mark_complete(payload, reason, final_answer or summary)
            return f"✓ Verification passed — goal marked complete. Command: `{verify_command}` (exit 0)."

        return (
            f"✗ Verification FAILED — goal stays active. Command `{verify_command}` "
            f"exited {exit_code}. Fix the failures below and call verify_completion again.\n"
            f"--- stdout ---\n{stdout or '(empty)'}\n"
            f"--- stderr ---\n{stderr or '(empty)'}"
        )

    async def _verify_criteria(self, payload: dict, acceptance_criteria: str, summary: str, final_answer: str) -> str:
        acceptance_criteria = (acceptance_criteria or "").strip()
        summary = (summary or "").strip()
        if not acceptance_criteria:
            return (
                "mode='criteria' requires acceptance_criteria — the checklist the "
                "deliverable must satisfy. For code tasks prefer mode='test' instead."
            )
        if self._judge_model is None:
            return (
                "No judge model is configured, so criteria mode is unavailable. "
                "Use mode='test' with a concrete verify_command instead."
            )

        objective = payload.get("objective", "")
        # Reuse the standard judge. Feed the acceptance criteria as subgoals
        # (each must have concrete evidence) and the agent's summary as the
        # 'response' being judged.
        criteria_list = [c.strip("-• \t") for c in acceptance_criteria.splitlines() if c.strip()]
        verdict, reason, parse_failed = await judge_goal(
            self._judge_model,
            objective,
            final_response=summary or final_answer or "(no summary provided)",
            subgoals=criteria_list or [acceptance_criteria],
        )
        if parse_failed:
            return (
                f"Could not get a clear verdict from the judge ({reason}). "
                f"Goal stays active. Provide a more concrete summary with evidence, then verify again."
            )
        if verdict == "done":
            self._mark_complete(payload, f"criteria satisfied: {reason}", final_answer or summary)
            return f"✓ Verification passed — goal marked complete. {reason}"
        return (
            f"✗ Verification FAILED — goal stays active. Unmet criteria: {reason}. "
            f"Address the gap and call verify_completion again."
        )

    # ------------------------------------------------------------------ update
    async def update_goal(self, status: str, reason: str = "", final_answer: str = "") -> str:
        """Pause the goal when blocked, or force-complete as an escape hatch.

        Prefer ``verify_completion`` to finish a goal — it backs completion
        with real evidence. Use ``update_goal`` for:

        - ``status="paused"``: you are blocked and need new input from the
          user (ambiguous requirements, missing credentials). Stops the
          auto-continuation loop without losing progress.
        - ``status="complete"``: escape hatch for goals with no runnable
          check (exit 0) and no judgeable criteria. Pass the full
          deliverable in ``final_answer``.

        Args:
            status: ``"complete"`` or ``"paused"``.
            reason: One-sentence rationale shown to the user.
            final_answer: The complete substantive result for the user.
        """
        status = (status or "").strip().lower()
        if status not in ("complete", "paused"):
            return (
                f"Invalid status '{status}'. Must be 'complete' or 'paused'. "
                f"You cannot rewrite the objective; only the user can change it."
            )

        payload, err = self._load_active_payload()
        if err is not None:
            return err

        payload["status"] = status
        payload["last_verdict"] = "tool_signal"
        payload["last_reason"] = reason or f"agent-marked {status}"
        payload["updated_at"] = time.time()
        if status == "paused":
            payload["paused_reason"] = "agent-tool"
        # Persist the substantive deliverable separately from chat content so
        # it survives any closing chatter the model emits after this call.
        if final_answer and final_answer.strip():
            payload["final_answer"] = final_answer.strip()

        try:
            state = GoalState.from_dict(payload)
            self._session_log.append_goal(state)
        except (TypeError, ValueError) as exc:
            logger.warning("GoalTool.update_goal serialization failed: %s", exc)
            return f"Failed to update goal: {exc}"

        if status == "complete":
            return f"Goal marked complete: {reason or 'agent confirmed done'}"
        return f"Goal paused: {reason or 'agent reported blocked'}"
