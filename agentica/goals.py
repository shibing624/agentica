# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Persistent session goals — the standing-goal loop for Agentica.

A goal is a free-form user objective that stays active across turns. After
each turn completes, an external judge (via ``Model.response()``) decides
whether the goal is satisfied by the assistant's last response. If not,
the CLI feeds a continuation prompt back into the same session and keeps
working until the goal is done, the turn budget is exhausted, the user
pauses/clears it, or the user sends a new message (which takes priority
and preempts the goal loop).

State is persisted in ``SessionLog`` as a ``type="goal"`` entry so
``/resume`` can read it back. Nothing in this module touches the agent's
system prompt or toolset — continuation prompts are plain user messages.

Design constraints (see docs/learn_cc/goal.md):
- SDK-first: GoalManager is a plain async controller, not tied to CLI.
- Async-first: ``evaluate_after_turn`` is ``async def``; judge calls
  ``await model.response(messages=[...])``.
- UI-neutral: GoalManager NEVER prints; it returns ``GoalDecision`` and
  lets the caller render the message.
- No defensive ``getattr`` — fields are formally declared on dataclasses.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from agentica.run_events import RunEventType
from agentica.utils.log import logger

if TYPE_CHECKING:
    from agentica.memory.session_log import SessionLog
    from agentica.model.base import Model
    from agentica.run_response import RunResponse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default turn cap for a standing goal. NOTE: this is the safety-net cap
# against runaway loops, NOT the primary cost budget — ``token_budget`` and
# ``wall_clock_budget_sec`` (when set) are the real hard caps and take
# precedence in ``evaluate_after_turn``. Empirically:
#   - one-shot tasks (compute X, draft a line): 1–3 turns
#   - bug fixes (search → edit → test → verify):  5–15 turns
#   - feature + tests: 20–50 turns
#   - multi-step refactor / migration: 50–100 turns
# A default of 100 keeps the safety net loose enough that almost no real
# workflow trips it accidentally — runaway loops are still caught by
# token / wall-clock budgets (the real cost gate) or by the
# consecutive-parse-failure pause.
DEFAULT_TURN_BUDGET = 100
MAX_CONSECUTIVE_PARSE_FAILURES = 3

JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. Consider the agent's last response "
    "only. Be conservative: only mark done=true when the response shows "
    "the goal is actually complete (not merely progress).\n"
    "Reply ONLY with a single JSON object on one line, no prose:\n"
    '{"done": <true|false>, "reason": "<one-sentence rationale>"}'
)

CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {objective}\n"
    "{subgoals_block}\n"
    "Continue working toward this goal. Take the next concrete step. "
    "Do not stop merely because you made partial progress. "
    "If the goal is complete, state the evidence clearly and stop. "
    "If blocked and needing user input, explain the blocker and stop."
)

CONTINUATION_PROMPT_PREFIX = "[Continuing toward your standing goal]"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GoalState:
    """Persistent per-session goal state.

    Serialized into SessionLog as a ``type="goal"`` entry; never injected
    into model history (SessionLog._build_messages() only whitelists
    user/assistant/system/tool).

    Budgets (P1 S2):
        ``turn_budget``       hard cap on continuation turns
        ``token_budget``      hard cap on accumulated input+output tokens
                              (None = unlimited)
        ``wall_clock_budget_sec``  hard cap on agent wall-clock seconds
                              (None = unlimited)

    When ANY budget is exhausted the state goes to ``budget_limited``
    (semantically distinct from ``paused``: budget-limited means the
    user must decide to extend the cap or accept the partial result).
    """

    session_id: str
    objective: str
    status: str = "active"            # active | paused | complete | cleared | budget_limited
    turn_budget: int = DEFAULT_TURN_BUDGET
    turns_used: int = 0
    token_budget: Optional[int] = None
    tokens_used: int = 0
    wall_clock_budget_sec: Optional[float] = None
    wall_clock_used_sec: float = 0.0
    subgoals: List[str] = field(default_factory=list)
    consecutive_parse_failures: int = 0
    last_verdict: Optional[str] = None      # "done" | "continue" | "parse_failed" | "tool_signal"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None     # user | interrupted | budget | judge-broken | resume-safety | agent-tool
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalState":
        # Defensive against older payloads — only pass known fields.
        known = {f for f in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class GoalRunResult:
    """Return value of ``Agent.run_goal()``.

    A flat, ergonomic summary of a standing-goal loop execution.

    Attributes:
        status:       One of ``complete`` / ``paused`` / ``budget_limited``.
        reason:       Human-readable rationale (judge verdict, budget
                      message, or tool reason).
        run_response: The agent's last ``RunResponse`` (same shape as
                      ``Agent.run_response``). Full access to ``content``,
                      ``cost_tracker``, ``messages``, ``tool_calls`` for
                      callers that need them. May be ``None`` only if the
                      loop terminated before the first ``agent.run()``.
        goal:         Final ``GoalState`` snapshot (objective, counters,
                      subgoals, last_verdict).
        turns_used:   Convenience copy of ``goal.turns_used``.

    Convenience accessor: ``result.response_content`` returns
    ``run_response.content or ""`` so the 90% case stays a one-liner.
    """

    status: str
    reason: str
    run_response: Optional["RunResponse"]
    goal: "GoalState"
    turns_used: int

    @property
    def response_content(self) -> str:
        """Last assistant message content, or "" if no response was produced."""
        if self.run_response is None:
            return ""
        return self.run_response.content or ""


@dataclass
class GoalDecision:
    """Returned by ``GoalManager.evaluate_after_turn()``. UI-neutral."""

    status: str                               # active | paused | complete
    should_continue: bool
    continuation_prompt: Optional[str]
    verdict: str                              # done | continue | parse_failed
    reason: str
    message: str                              # human-readable summary for UI


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> Tuple[str, str, bool]:
    """Parse the judge's JSON output.

    Returns:
        (verdict, reason, parse_failed)
        verdict: "done" | "continue"
        parse_failed: True if JSON could not be extracted/parsed.
    """
    if not raw:
        return "continue", "empty judge response", True

    candidate = raw.strip()
    # Strip ```json fences if present.
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(candidate)
        if not match:
            return "continue", f"unparseable judge output: {candidate[:120]}", True
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return "continue", f"unparseable judge output: {candidate[:120]}", True

    if not isinstance(payload, dict) or "done" not in payload:
        return "continue", "judge JSON missing 'done' field", True

    done = bool(payload.get("done"))
    reason = str(payload.get("reason", "")).strip() or ("done" if done else "continue")
    return ("done" if done else "continue"), reason, False


def _build_judge_user_prompt(
    objective: str,
    final_response: str,
    subgoals: Optional[List[str]],
) -> str:
    parts = [f"Goal: {objective}"]
    if subgoals:
        parts.append("Acceptance criteria:")
        for i, sg in enumerate(subgoals, 1):
            parts.append(f"  {i}. {sg}")
    parts.append("")
    parts.append("Agent's last response:")
    parts.append("---")
    parts.append(final_response.strip() or "(empty)")
    parts.append("---")
    parts.append("Is the goal complete? Reply with the JSON object only.")
    return "\n".join(parts)


async def judge_goal(
    model: "Model",
    objective: str,
    final_response: str,
    subgoals: Optional[List[str]] = None,
) -> Tuple[str, str, bool]:
    """Ask the judge model whether ``final_response`` satisfies ``objective``.

    Uses Agentica's standard async entrypoint ``Model.response()``.
    Fail-open on transport errors (caller treats this as ``continue`` and
    does NOT count it as a parse failure).

    Returns:
        (verdict, reason, parse_failed)
    """
    from agentica.model.message import Message

    user_prompt = _build_judge_user_prompt(objective, final_response, subgoals)
    messages = [
        Message(role="system", content=JUDGE_SYSTEM_PROMPT),
        Message(role="user", content=user_prompt),
    ]
    try:
        resp = await model.response(messages=messages)
    except Exception as exc:
        logger.debug("judge_goal: model.response() failed, fail-open: %s", exc)
        return "continue", f"judge unavailable: {exc}", False

    return _parse_judge_response(resp.content or "")


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class GoalManager:
    """Per-session goal lifecycle controller.

    Owned by the CLI layer (or any external orchestrator). Persists state
    through ``SessionLog.append_goal()`` and reads it back via
    ``SessionLog.load_goal()``. The manager itself is in-memory cached so
    the same instance can be reused across turns without re-parsing JSONL.
    """

    def __init__(
        self,
        session_log: "SessionLog",
        *,
        default_turn_budget: int = DEFAULT_TURN_BUDGET,
        judge_model: Optional["Model"] = None,
        event_callback: Optional[Callable[[RunEventType, Dict[str, Any]], None]] = None,
    ):
        self.session_log = session_log
        self.default_turn_budget = default_turn_budget
        self.judge_model = judge_model
        # Optional emit-site for RunEventType.goal_* (A2). Caller wires it to
        # tracing exporter / hooks / printer. Must NOT raise.
        self.event_callback = event_callback
        self._state: Optional[GoalState] = None
        self._loaded = False

    def _emit(self, event: RunEventType, **payload: Any) -> None:
        if self.event_callback is None or self._state is None:
            return
        try:
            self.event_callback(event, {
                "session_id": self._state.session_id,
                "objective": self._state.objective,
                "status": self._state.status,
                "turns_used": self._state.turns_used,
                "turn_budget": self._state.turn_budget,
                "tokens_used": self._state.tokens_used,
                "token_budget": self._state.token_budget,
                "wall_clock_used_sec": self._state.wall_clock_used_sec,
                "wall_clock_budget_sec": self._state.wall_clock_budget_sec,
                **payload,
            })
        except Exception as exc:  # pragma: no cover - callback bug isolation
            logger.warning("GoalManager event_callback raised: %s", exc)

    # ------------------------------------------------------------------ load/save
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        payload = self.session_log.load_goal()
        if payload is not None:
            try:
                self._state = GoalState.from_dict(payload)
            except (TypeError, ValueError) as exc:
                logger.warning("GoalManager: failed to deserialize goal entry: %s", exc)
                self._state = None
        self._loaded = True

    def _persist(self) -> None:
        if self._state is None:
            return
        self._state.updated_at = time.time()
        self.session_log.append_goal(self._state)

    # ------------------------------------------------------------------ public API
    def load(self) -> Optional[GoalState]:
        self._ensure_loaded()
        return self._state

    def is_active(self) -> bool:
        self._ensure_loaded()
        return self._state is not None and self._state.status == "active"

    def set(
        self,
        objective: str,
        *,
        turn_budget: Optional[int] = None,
        token_budget: Optional[int] = None,
        wall_clock_budget_sec: Optional[float] = None,
    ) -> GoalState:
        objective = (objective or "").strip()
        if not objective:
            raise ValueError("Goal objective cannot be empty.")
        self._ensure_loaded()
        now = time.time()
        self._state = GoalState(
            session_id=self.session_log.session_id,
            objective=objective,
            status="active",
            turn_budget=turn_budget if turn_budget is not None else self.default_turn_budget,
            turns_used=0,
            token_budget=token_budget,
            tokens_used=0,
            wall_clock_budget_sec=wall_clock_budget_sec,
            wall_clock_used_sec=0.0,
            subgoals=[],
            consecutive_parse_failures=0,
            last_verdict=None,
            last_reason=None,
            paused_reason=None,
            created_at=now,
            updated_at=now,
        )
        self._persist()
        self._emit(RunEventType.goal_set, reason="user")
        return self._state

    def pause(self, reason: str = "user") -> Optional[GoalState]:
        self._ensure_loaded()
        if self._state is None or self._state.status not in ("active",):
            return self._state
        self._state.status = "paused"
        self._state.paused_reason = reason
        self._persist()
        self._emit(RunEventType.goal_paused, paused_reason=reason)
        return self._state

    def _check_budget_exhausted(self) -> Optional[str]:
        """Return a human-readable reason if any hard budget is exhausted,
        else None. ``turn_budget`` is checked separately AFTER the judge so
        we don't waste a judge call when token/time budget already blew.
        """
        s = self._state
        if s is None:
            return None
        if s.token_budget is not None and s.tokens_used >= s.token_budget:
            return f"token budget exhausted ({s.tokens_used}/{s.token_budget})"
        if (
            s.wall_clock_budget_sec is not None
            and s.wall_clock_used_sec >= s.wall_clock_budget_sec
        ):
            return (
                f"wall-clock budget exhausted "
                f"({s.wall_clock_used_sec:.0f}s/{s.wall_clock_budget_sec:.0f}s)"
            )
        return None

    def _reload_from_disk(self) -> None:
        """Refresh in-memory cache from SessionLog.

        Used by ``evaluate_after_turn`` to pick up any tool-driven mutation
        of GoalState that happened during the just-finished turn (see
        ``GoalTool.update_goal``). Without this, the manager and disk
        could diverge for one turn.
        """
        payload = self.session_log.load_goal()
        if payload is not None:
            try:
                self._state = GoalState.from_dict(payload)
            except (TypeError, ValueError) as exc:
                logger.warning("GoalManager: failed to deserialize goal entry: %s", exc)
        self._loaded = True

    def resume(self) -> Optional[GoalState]:
        self._ensure_loaded()
        if self._state is None:
            return None
        if self._state.status not in ("paused", "budget_limited"):
            return self._state
        self._state.status = "active"
        self._state.paused_reason = None
        self._state.consecutive_parse_failures = 0
        self._persist()
        return self._state

    def clear(self) -> None:
        self._ensure_loaded()
        if self._state is None:
            return
        self._state.status = "cleared"
        self._persist()
        self._state = None

    def force_pause_on_resume(self) -> Optional[GoalState]:
        """Called by /resume <sid> to demote an active goal to paused for safety.

        Does nothing if the goal is already non-active.
        """
        self._ensure_loaded()
        if self._state is None or self._state.status != "active":
            return self._state
        self._state.status = "paused"
        self._state.paused_reason = "resume-safety"
        self._persist()
        return self._state

    # --- tool-driven mutations (called from GoalTool) -----------------------
    def mark_complete_from_tool(self, reason: str) -> Optional[GoalState]:
        """The agent's ``update_goal`` tool said the goal is complete."""
        self._reload_from_disk()
        if self._state is None or self._state.status != "active":
            return self._state
        self._state.status = "complete"
        self._state.last_verdict = "tool_signal"
        self._state.last_reason = reason or "agent-marked complete"
        self._persist()
        self._emit(RunEventType.goal_completed, verdict="tool_signal", reason=reason)
        return self._state

    def mark_paused_from_tool(self, reason: str) -> Optional[GoalState]:
        """The agent's ``update_goal`` tool said it is blocked / paused."""
        self._reload_from_disk()
        if self._state is None or self._state.status != "active":
            return self._state
        self._state.status = "paused"
        self._state.paused_reason = "agent-tool"
        self._state.last_verdict = "tool_signal"
        self._state.last_reason = reason or "agent-marked paused"
        self._persist()
        self._emit(RunEventType.goal_paused, paused_reason="agent-tool", reason=reason)
        return self._state

    # --- subgoals -----------------------------------------------------------
    def add_subgoal(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Subgoal text cannot be empty.")
        self._ensure_loaded()
        if self._state is None:
            raise ValueError("No active goal — set one with /goal first.")
        self._state.subgoals.append(text)
        self._persist()
        return text

    def remove_subgoal(self, index: int) -> Optional[str]:
        """``index`` is 1-based to match the CLI display."""
        self._ensure_loaded()
        if self._state is None:
            return None
        i = index - 1
        if i < 0 or i >= len(self._state.subgoals):
            return None
        removed = self._state.subgoals.pop(i)
        self._persist()
        return removed

    def clear_subgoals(self) -> int:
        self._ensure_loaded()
        if self._state is None or not self._state.subgoals:
            return 0
        n = len(self._state.subgoals)
        self._state.subgoals.clear()
        self._persist()
        return n

    # --- status -------------------------------------------------------------
    def status_line(self) -> str:
        self._ensure_loaded()
        if self._state is None:
            return "No active goal."
        s = self._state
        head = f"Goal [{s.status}] ({s.turns_used}/{s.turn_budget} turns): {s.objective}"
        extras = []
        budget_bits = []
        if s.token_budget is not None:
            budget_bits.append(f"tokens {s.tokens_used:,}/{s.token_budget:,}")
        elif s.tokens_used:
            budget_bits.append(f"tokens {s.tokens_used:,}")
        if s.wall_clock_budget_sec is not None:
            budget_bits.append(
                f"wall {s.wall_clock_used_sec:.0f}s/{s.wall_clock_budget_sec:.0f}s"
            )
        elif s.wall_clock_used_sec:
            budget_bits.append(f"wall {s.wall_clock_used_sec:.0f}s")
        if budget_bits:
            extras.append("  Budget: " + " | ".join(budget_bits))
        if s.subgoals:
            extras.append(f"  Subgoals ({len(s.subgoals)}):")
            for i, sg in enumerate(s.subgoals, 1):
                extras.append(f"    {i}. {sg}")
        if s.paused_reason and s.status in ("paused", "budget_limited"):
            extras.append(f"  Paused reason: {s.paused_reason}")
        if s.last_reason:
            extras.append(f"  Last verdict: {s.last_verdict} — {s.last_reason}")
        return "\n".join([head] + extras)

    # --- per-turn loop ------------------------------------------------------
    def next_continuation_prompt(self) -> str:
        self._ensure_loaded()
        if self._state is None:
            return ""
        subgoals_block = ""
        if self._state.subgoals:
            lines = ["Acceptance criteria still to satisfy:"]
            for i, sg in enumerate(self._state.subgoals, 1):
                lines.append(f"  {i}. {sg}")
            subgoals_block = "\n" + "\n".join(lines) + "\n"
        return CONTINUATION_PROMPT_TEMPLATE.format(
            objective=self._state.objective,
            subgoals_block=subgoals_block,
        )

    async def evaluate_after_turn(
        self,
        final_response: str,
        *,
        token_delta: int = 0,
        elapsed_sec: float = 0.0,
    ) -> GoalDecision:
        """Drive one round of the goal loop after an agent turn finishes.

        Mutates ``GoalState`` (turns_used, tokens_used, wall_clock_used_sec,
        status, counters) and persists.

        Args:
            final_response: Agent's last assistant message content.
            token_delta:    Tokens consumed by the just-finished turn
                            (input + output). Caller reads from
                            ``agent.run_response.cost_tracker``.
            elapsed_sec:    Wall-clock seconds the turn took.

        Returns:
            A UI-neutral ``GoalDecision``.
        """
        # Re-read disk first so a GoalTool call during the turn takes effect.
        self._reload_from_disk()

        if self._state is None:
            return GoalDecision(
                status="cleared",
                should_continue=False,
                continuation_prompt=None,
                verdict="continue",
                reason="no active goal",
                message="",
            )

        # Charge the turn that just ran. This is independent of the
        # eventual decision (tool short-circuit / budget / judge) — the
        # LLM work has already happened and the cost is real. Doing this
        # in ONE place (instead of duplicating below) avoids drift.
        if self._state.status != "cleared":
            self._state.turns_used += 1
            if token_delta > 0:
                self._state.tokens_used += int(token_delta)
            if elapsed_sec > 0:
                self._state.wall_clock_used_sec += float(elapsed_sec)

        # Hard budget caps (P1 S2) take precedence over EVERYTHING else,
        # including tool short-circuit and judge. Rationale: the user set
        # the cap to bound resource consumption; once we've blown past it
        # that's the primary signal to report, regardless of what the
        # model claimed via update_goal.
        budget_msg = self._check_budget_exhausted()
        if budget_msg is not None:
            self._state.status = "budget_limited"
            self._state.paused_reason = "budget"
            self._persist()
            self._emit(RunEventType.goal_paused, paused_reason="budget",
                       budget_message=budget_msg)
            return GoalDecision(
                status="budget_limited",
                should_continue=False,
                continuation_prompt=None,
                verdict="continue",
                reason=budget_msg,
                message=f"⊙ Goal budget-limited: {budget_msg}. Use /goal resume to continue.",
            )

        if self._state.status != "active":
            # Tool may have marked complete/paused mid-turn.
            if self._state.status == "complete":
                self._persist()
                self._emit(
                    RunEventType.goal_completed,
                    verdict="tool_signal",
                    reason=self._state.last_reason or "agent marked complete",
                )
                return GoalDecision(
                    status="complete",
                    should_continue=False,
                    continuation_prompt=None,
                    verdict="tool_signal",
                    reason=self._state.last_reason or "agent marked complete",
                    message=f"✓ Goal complete (agent-marked): {self._state.last_reason or ''}",
                )
            if self._state.status == "paused":
                self._persist()
                self._emit(
                    RunEventType.goal_paused,
                    paused_reason=self._state.paused_reason or "agent-tool",
                )
                return GoalDecision(
                    status="paused",
                    should_continue=False,
                    continuation_prompt=None,
                    verdict="tool_signal",
                    reason=self._state.last_reason or "agent paused goal",
                    message=f"⊙ Goal paused (agent-marked): {self._state.last_reason or ''}",
                )
            return GoalDecision(
                status=self._state.status,
                should_continue=False,
                continuation_prompt=None,
                verdict="continue",
                reason="no active goal",
                message="",
            )

        if self.judge_model is None:
            # Caller forgot to pass a model — fail-open: treat as continue
            # but do NOT auto-loop because we have no way to ever say done.
            self._state.status = "paused"
            self._state.paused_reason = "judge-broken"
            self._persist()
            return GoalDecision(
                status="paused",
                should_continue=False,
                continuation_prompt=None,
                verdict="parse_failed",
                reason="no judge model configured",
                message="⊙ Goal paused: no judge model configured.",
            )

        verdict, reason, parse_failed = await judge_goal(
            self.judge_model,
            self._state.objective,
            final_response,
            subgoals=self._state.subgoals or None,
        )
        self._state.last_verdict = "parse_failed" if parse_failed else verdict
        self._state.last_reason = reason

        if parse_failed:
            self._state.consecutive_parse_failures += 1
            if self._state.consecutive_parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                self._state.status = "paused"
                self._state.paused_reason = "judge-broken"
                self._persist()
                return GoalDecision(
                    status="paused",
                    should_continue=False,
                    continuation_prompt=None,
                    verdict="parse_failed",
                    reason=reason,
                    message=(
                        f"⊙ Goal paused: judge JSON unparseable "
                        f"{self._state.consecutive_parse_failures}x in a row."
                    ),
                )
            # Soft retry — keep going.
            self._persist()
            return GoalDecision(
                status="active",
                should_continue=True,
                continuation_prompt=self.next_continuation_prompt(),
                verdict="parse_failed",
                reason=reason,
                message=f"⊙ Goal: judge unparseable ({self._state.consecutive_parse_failures}/"
                        f"{MAX_CONSECUTIVE_PARSE_FAILURES}); continuing.",
            )

        self._state.consecutive_parse_failures = 0

        if verdict == "done":
            self._state.status = "complete"
            self._persist()
            self._emit(RunEventType.goal_completed, verdict="done", reason=reason)
            return GoalDecision(
                status="complete",
                should_continue=False,
                continuation_prompt=None,
                verdict="done",
                reason=reason,
                message=f"✓ Goal complete: {reason}",
            )

        if self._state.turns_used >= self._state.turn_budget:
            self._state.status = "budget_limited"
            self._state.paused_reason = "budget"
            self._persist()
            self._emit(RunEventType.goal_paused, paused_reason="budget",
                       budget_message="turn budget exhausted")
            return GoalDecision(
                status="budget_limited",
                should_continue=False,
                continuation_prompt=None,
                verdict="continue",
                reason=reason,
                message=(
                    f"⊙ Goal budget-limited: turn budget exhausted "
                    f"({self._state.turns_used}/{self._state.turn_budget}). "
                    f"Use /goal resume to continue."
                ),
            )

        self._persist()
        self._emit(RunEventType.goal_continuing, verdict="continue", reason=reason)
        return GoalDecision(
            status="active",
            should_continue=True,
            continuation_prompt=self.next_continuation_prompt(),
            verdict="continue",
            reason=reason,
            message=f"↻ Goal continuing ({self._state.turns_used}/{self._state.turn_budget}): {reason}",
        )
