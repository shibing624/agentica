# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Goal loop mixin for Agent — standing-goal drive + per-turn step.

Extracted from Agent so base.py keeps only the public run API surface and
thin delegation. The goal loop (set objective → run turns → evaluate → stop)
lives here, mirroring the existing PromptsMixin / ToolsMixin tradition.
"""

import inspect
import time
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
    TYPE_CHECKING,
)
from uuid import uuid4

from agentica.model.message import Message
from agentica.run_response import RunResponse
from agentica.run_context import TaskAnchor
from agentica.memory.session_log import SessionLog

if TYPE_CHECKING:
    from agentica.goals import GoalRunResult, GoalStepResult


class GoalMixin:
    """Standing-goal loop methods, mixed into :class:`Agent`.

    Provides ``get_goal_manager`` / ``enable_goal_tool`` / ``run_goal`` /
    ``run_goal_step``. All state lives on the host Agent instance
    (``self.goal_manager``, ``self._session_log``, ``self.task_anchor`` …),
    so the mixin needs no constructor of its own.
    """

    def get_goal_manager(
        self,
        *,
        default_turn_budget: Optional[int] = None,
        default_token_budget: Optional[int] = None,
        event_callback: Optional[Callable[..., None]] = None,
        verifier: Optional[Callable[..., Any]] = None,
        auto_judge: bool = False,
    ) -> Any:
        """Return (and lazily create) this agent's ``GoalManager``.

        Creates a ``SessionLog`` on the fly if the agent was built without
        a ``session_id``. Idempotent: subsequent calls return the same
        manager so any persisted GoalState stays consistent.

        Args:
            default_turn_budget: Optional turn cap for newly set goals
                (``None`` = no turn limit). Ignored if a manager already exists.
            default_token_budget: Default token cap for newly set goals
                (``None`` = unlimited, ``agentica.goals.DEFAULT_TOKEN_BUDGET``).
                Ignored if a manager already exists.
            event_callback: ``(RunEventType, dict) -> None`` hook for
                ``goal.set / continuing / completed / paused`` events.

        Returns:
            ``agentica.goals.GoalManager``.
        """
        # Local import keeps module import graph cheap for users that
        # never touch the goal loop.
        from agentica.goals import GoalManager, DEFAULT_TOKEN_BUDGET

        if self._session_log is None:
            if self.session_id is None:
                self.session_id = str(uuid4())
            self._session_log = SessionLog(
                session_id=self.session_id,
                work_dir=self.work_dir,
                user_id=self.user_id,
            )

        if self.goal_manager is None:
            self.goal_manager = GoalManager(
                self._session_log,
                default_turn_budget=default_turn_budget,
                default_token_budget=(
                    default_token_budget if default_token_budget is not None else DEFAULT_TOKEN_BUDGET
                ),
                judge_model=self.resolve_auxiliary_model("goal_judge"),
                event_callback=event_callback,
                verifier=verifier,
                auto_judge=auto_judge,
            )
            # Load any persisted state from a previous session.
            self.goal_manager.load()
        else:
            # Allow re-binding the callback / verifier on a pre-existing
            # manager (cheap, no-mutation otherwise).
            if event_callback is not None:
                self.goal_manager.event_callback = event_callback
            if verifier is not None:
                self.goal_manager.verifier = verifier

        return self.goal_manager

    def enable_goal_tool(self) -> None:
        """Attach ``GoalTool`` so the model can drive goal completion itself.

        Exposes two tools to the model:

        - ``verify_completion`` — the primary, evidence-backed completion
          check. The model calls it only when it believes the goal is done;
          the tool runs tests / checks criteria and marks the goal complete
          only when green (short-circuiting any judge).
        - ``update_goal`` — narrow control channel to pause when blocked (or
          force-complete as an escape hatch).

        The ``verify_completion`` criteria mode needs a judge model, so the
        tool is bound to the manager's ``judge_model`` and the agent's
        working directory (for running ``verify_command`` in test mode).
        Idempotent.
        """
        from agentica.tools.goal_tool import GoalTool

        mgr = self.get_goal_manager()
        if self.tools is None:
            self.tools = []
        for t in self.tools:
            if isinstance(t, GoalTool):
                return
        work_dir = getattr(self, "work_dir", None) or getattr(self, "base_dir", None)
        self.tools.append(
            GoalTool(mgr.session_log, judge_model=mgr.judge_model, work_dir=work_dir)
        )

    def detach_goal_tool(self) -> None:
        """Remove ``GoalTool`` after an in-place ``run_goal`` so later chat
        turns do not keep ``verify_completion`` / ``update_goal``.
        """
        from agentica.tools.goal_tool import GoalTool

        if not self.tools:
            return
        remaining = [t for t in self.tools if not isinstance(t, GoalTool)]
        if len(remaining) == len(self.tools):
            return
        self.tools = remaining
        self._wire_tools_to_self()

    async def run_goal(
        self,
        objective: str,
        *,
        turn_budget: Optional[int] = None,
        token_budget: Optional[int] = None,
        wall_clock_budget_sec: Optional[float] = None,
        attach_goal_tool: bool = True,
        event_callback: Optional[Callable[..., None]] = None,
        stream_chunks: Optional[Callable[[RunResponse], Any]] = None,
        isolate: bool = True,
        verifier: Optional[Callable[..., Any]] = None,
        seed_messages: Optional[Sequence[Union[Message, Dict[str, Any]]]] = None,
        auto_judge: bool = False,
    ) -> "GoalRunResult":
        """Drive the standing-goal loop until completion / pause / budget.

        Budget model: ``token_budget`` is the primary cost gate and defaults
        to unlimited (``None``) for CLI, SDK and Web. Pass a positive int to
        cap spend, or ``-1`` for the same unlimited sentinel. ``turn_budget``
        defaults to ``None`` (no turn cap); pass an int only when you want an
        extra hard turn ceiling. ``wall_clock_budget_sec`` remains optional
        for SLA-style limits.

        Ergonomic entry point: callers do NOT touch ``SessionLog``,
        ``GoalManager``, or ``GoalTool`` directly. The loop:

            1. Sets the objective on the manager (resets turns_used etc).
            2. Binds ``TaskAnchor`` to the objective so retrieval / prompt
               anchoring use it for every turn.
            3. Optionally attaches ``GoalTool`` so the model can short
               circuit the judge.
            4. Runs ``self.run()`` repeatedly, feeding each turn's
               ``token_delta`` and wall-clock seconds into the manager.
            5. Stops when the manager says the goal is complete /
               paused / budget_limited.

        Args:
            objective: The standing goal text. Used as the first prompt.
            turn_budget: Optional max LLM turns. ``None`` = no turn limit.
            token_budget: Max cumulative input+output tokens. ``None`` /
                ``-1`` = unlimited. Pass a positive int to cap spend.
            wall_clock_budget_sec: Max agent wall-clock seconds. ``None`` /
                ``-1`` = unlimited. Recommended ``1800``–``3600``
                for long tasks.

                The three budgets are **independent hard caps — whichever
                hits first stops the loop** (AND/intersection semantics).
                Priority each turn: ``budget > tool short-circuit > judge``.
            attach_goal_tool: Register ``GoalTool`` (``verify_completion`` +
                ``update_goal``) on this agent so the model can drive
                completion itself. This is the DEFAULT completion path.
            auto_judge: When True, restore the legacy per-turn LLM judge as a
                completion fallback (costs a judge call every turn). Default
                False: completion comes only from ``verify_completion`` /
                verifier, and the loop otherwise runs until a budget cap.
            event_callback: ``goal.*`` event hook.
            stream_chunks: Optional per-chunk hook. When set, each turn
                uses ``run_stream`` and this is called with every
                ``RunResponse`` chunk (sync or async). The web ``/goal``
                path uses it so tool calls and tokens appear as they
                happen, matching ordinary chat. Unset keeps ``run()``
                (SDK / tests). The CLI does not use this — its REPL
                already streams via ``run_stream_sync``.
            isolate: When True (SDK default), the loop runs on ``clone()``
                so two concurrent ``run_goal()`` calls on one instance
                cannot share memory or steer buffers. The gateway holds a
                per-session lock and must pass False: Web steer / cancel /
                the next chat turn all address the cached agent, and a
                clone would leave them talking to an idle parent.
            verifier: Optional callable that decides per-turn whether the
                goal is satisfied, WITHOUT an LLM call. Signature:
                ``(VerifierContext) -> Optional[VerifierResult]`` (sync or
                async). Returning ``VerifierResult(done=True, ...)`` stops
                the loop immediately; ``done=False`` continues; ``None``
                falls back to the LLM judge. A bare ``bool`` is accepted
                as shorthand for ``VerifierResult(done=bool_value)``.
                Exceptions raised by the verifier are caught and treated
                as ``None`` — a buggy verifier must not crash the loop.
                Priority: budget > tool short-circuit > tool-stuck >
                verifier > judge.
            seed_messages: Prior conversation to seed into the (cloned) goal
                agent's working memory before the first turn, so the loop
                starts from real context. Used by the Web UI so ``/goal`` sees
                the chat history the user built up — the CLI drives ``/goal`` on
                the live agent and keeps history naturally. Accepts a sequence
                of ``Message`` objects or plain ``{role, content}`` dicts.

        Returns:
            ``agentica.goals.GoalRunResult`` with final status / reason /
            ``RunResponse`` / GoalState snapshot / turns_used.
        """
        from agentica.goals import GoalRunResult

        # SDK default: clone so concurrent run_goal() calls on one instance
        # cannot share memory or steer buffers. The gateway holds a
        # per-session lock and passes isolate=False — Web steer / cancel /
        # the next chat turn all address the cached agent.
        agent = self.clone() if isolate else self

        # Seed only the clone. Hydrating the live agent would duplicate
        # history it already holds.
        if isolate and seed_messages:
            seed_dicts = [
                m.model_dump(exclude_none=True) if isinstance(m, Message) else m
                for m in seed_messages
            ]
            agent.working_memory.hydrate_runs_from_history(seed_dicts)

        mgr = agent.get_goal_manager(
            event_callback=event_callback, verifier=verifier, auto_judge=auto_judge
        )
        state = mgr.set(
            objective,
            turn_budget=turn_budget,
            token_budget=token_budget,
            wall_clock_budget_sec=wall_clock_budget_sec,
        )

        # Pin the anchor up front so the first turn already uses it.
        # source="goal" → anchor is rendered into the system prompt every
        # turn for long-task drift defense; this is the whole point of
        # run_goal().
        agent.task_anchor = TaskAnchor(
            goal=state.objective,
            source_query=state.objective,
            source="goal",
        )
        agent._anchor_session_id = agent.session_id

        if attach_goal_tool:
            agent.enable_goal_tool()

        prompt = state.objective
        last_run_response: Optional[RunResponse] = None
        tokens_baseline = 0

        try:
            while True:
                step = await agent.run_goal_step(
                    prompt, tokens_baseline=tokens_baseline, stream_chunks=stream_chunks,
                )
                last_run_response = step.run_response
                tokens_baseline = step.tokens_baseline

                if not step.decision.should_continue:
                    final_state = mgr.load()
                    return GoalRunResult(
                        status=step.decision.status,
                        reason=step.decision.reason,
                        run_response=last_run_response,
                        goal=final_state,
                        turns_used=final_state.turns_used if final_state else 0,
                    )
                prompt = step.decision.continuation_prompt
        finally:
            if not isolate and attach_goal_tool:
                agent.detach_goal_tool()

    async def run_goal_step(
        self,
        prompt: str,
        *,
        tokens_baseline: int = 0,
        stream_chunks: Optional[Callable[[RunResponse], Any]] = None,
    ) -> "GoalStepResult":
        """Run ONE turn of the standing-goal loop and evaluate it.

        This is the shared, loop-agnostic unit of ``run_goal()``: it runs a
        single ``self.run(prompt)``, computes the turn's token delta and tool
        signals, feeds them to ``GoalManager.evaluate_after_turn()``, and
        returns the resulting :class:`~agentica.goals.GoalStepResult`.

        Both drivers reuse this:

        - ``run_goal()`` calls it in a ``while True`` until
          ``decision.should_continue`` is False (SDK / Gateway path).
        - The CLI ``/goal`` handler calls it once per turn from its REPL
          loop, so it keeps its own outer shell (user-input preemption,
          Ctrl+C pause, continuation queueing) instead of re-implementing the
          per-turn evaluation.

        The agent must already have a goal set (``get_goal_manager().set()``)
        — this method does NOT set the objective, bind the anchor, or attach
        the goal tool; the caller owns that setup once, up front.

        Args:
            prompt: The prompt for this turn (objective on turn 1, then each
                ``decision.continuation_prompt``).
            tokens_baseline: Accumulated input+output tokens across all PRIOR
                turns of this goal. Each turn's per-run token usage is added
                on and returned as ``result.tokens_baseline``; pass it back on
                the next call.
            stream_chunks: When set, this turn uses ``run_stream`` and the
                hook sees every chunk. Unset uses ``run()``.

        Returns:
            ``GoalStepResult`` with the turn's ``run_response``, the
            ``decision`` from the manager, and the updated ``tokens_baseline``.
        """
        from agentica.goals import GoalStepResult

        mgr = self.get_goal_manager()

        t0 = time.monotonic()
        if stream_chunks is None:
            response = await self.run(prompt)
        else:
            from agentica.run_config import RunConfig
            async for chunk in self.run_stream(
                prompt, config=RunConfig(stream_intermediate_steps=True),
            ):
                if chunk is None:
                    continue
                maybe = stream_chunks(chunk)
                if inspect.isawaitable(maybe):
                    await maybe
            response = self.run_response
        elapsed = time.monotonic() - t0

        final_text, token_delta, new_baseline, tool_pairs = mgr.extract_turn_signals(
            response, tokens_baseline
        )

        decision = await mgr.evaluate_after_turn(
            final_text,
            token_delta=token_delta,
            elapsed_sec=elapsed,
            tool_calls=tool_pairs or None,
        )

        return GoalStepResult(
            run_response=response,
            decision=decision,
            tokens_baseline=new_baseline,
        )
