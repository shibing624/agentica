# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Standing-goal loop SDK demo.

Shows how to drive Agentica's persistent ``/goal`` machinery from plain
Python — i.e. without going through the interactive CLI. You get the
same primitives the CLI uses:

    GoalManager     persistent state machine (set / pause / resume / clear)
    GoalTool        receive-only model tool: agent can mark complete or
                    paused, breaking the auto-continuation loop early
    SessionLog      append-only JSONL persistence; survives process restart
    event_callback  hook ``goal.set / continuing / completed / paused``
                    into your tracing / observability layer

The loop is just:

    1. Set a goal on a fresh GoalManager bound to ``agent._session_log``.
    2. Call ``agent.run(prompt)``.
    3. Pass the result into ``mgr.evaluate_after_turn(...)``.
       The manager runs the judge (``agent.auxiliary_model``) and tells
       you whether to stop or to use ``decision.continuation_prompt`` as
       the next prompt.
    4. Honor token / wall-clock budgets by feeding deltas from the
       agent's CostTracker.

Requires:  DEEPSEEK_API_KEY  (or swap in any other Agentica model).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, DeepSeekChat
from agentica.goals import GoalManager
from agentica.run_events import RunEventType
from agentica.tools.goal_tool import GoalTool


async def run_goal(
    agent: Agent,
    mgr: GoalManager,
    objective: str,
    *,
    max_turns: int = 5,
) -> None:
    """Generic driver: set objective, loop until judge says done or budget."""
    mgr.set(objective)
    print(f"\n[goal set] {objective}")
    print(f"           budget={mgr.load().turn_budget} turns")

    prompt = objective
    tokens_baseline = 0

    for turn in range(1, max_turns + 1):
        t0 = time.monotonic()
        resp = await agent.run(prompt)
        elapsed = time.monotonic() - t0
        content = (resp.content or "").strip()

        # Per-turn token delta from the agent's CostTracker.
        token_delta = 0
        if resp.cost_tracker is not None:
            total_now = (
                resp.cost_tracker.total_input_tokens
                + resp.cost_tracker.total_output_tokens
            )
            token_delta = max(0, total_now - tokens_baseline)
            tokens_baseline = total_now

        print(f"\n--- turn {turn} ({elapsed:.1f}s, {token_delta} tokens) ---")
        print(f"agent> {content[:200]}")

        decision = await mgr.evaluate_after_turn(
            content, token_delta=token_delta, elapsed_sec=elapsed,
        )
        print(f"judge> {decision.verdict}: {decision.reason[:160]}")

        if not decision.should_continue:
            print(f"[goal stopped] status={decision.status}")
            return
        prompt = decision.continuation_prompt

    print("[goal stopped] max_turns reached")


async def example_1_basic_loop() -> None:
    """A short objective that the model can finish in 1–2 turns."""
    print("=" * 60)
    print("Example 1: basic goal loop with external judge")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-basic",
        model=DeepSeekChat(),
        auxiliary_model=DeepSeekChat(),
        instructions="You are terse. One step per turn. State 'done' clearly when finished.",
    )
    mgr = GoalManager(
        agent._session_log,
        judge_model=agent.auxiliary_model,
        default_turn_budget=3,
    )
    await run_goal(agent, mgr, "Compute 17+9+16 and state the integer answer.")


async def example_2_with_goal_tool() -> None:
    """Attach the GoalTool so the agent can self-signal completion."""
    print("\n" + "=" * 60)
    print("Example 2: agent can short-circuit the judge via GoalTool")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-tool",
        model=DeepSeekChat(),
        auxiliary_model=DeepSeekChat(),
        instructions=(
            "You are terse. When you have finished the user's standing goal, "
            "call the update_goal tool with status='complete'. If you are "
            "blocked waiting for the user, call update_goal with status='paused'."
        ),
    )
    mgr = GoalManager(agent._session_log, judge_model=agent.auxiliary_model)
    # The tool is bound to the same SessionLog the manager reads from, so
    # any tool write is picked up by mgr.evaluate_after_turn() (it re-reads
    # disk first and skips the judge if the tool already marked complete).
    agent.tools = (agent.tools or []) + [GoalTool(agent._session_log)]
    await run_goal(
        agent, mgr,
        "Greet the user in exactly one sentence, then mark the goal complete.",
    )


async def example_3_token_budget() -> None:
    """Hard token cap — status becomes ``budget_limited`` (NOT ``paused``)."""
    print("\n" + "=" * 60)
    print("Example 3: token budget -> budget_limited")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-budget",
        model=DeepSeekChat(),
        auxiliary_model=DeepSeekChat(),
    )
    mgr = GoalManager(agent._session_log, judge_model=agent.auxiliary_model)
    # token_budget is small on purpose — one DeepSeek turn will blow it.
    mgr.set("Summarize TCP slow start in 2 sentences.", token_budget=30)

    resp = await agent.run(mgr.load().objective)
    print(f"agent> {(resp.content or '').strip()[:200]}")
    delta = (
        (resp.cost_tracker.total_input_tokens + resp.cost_tracker.total_output_tokens)
        if resp.cost_tracker else 0
    )
    decision = await mgr.evaluate_after_turn(resp.content or "", token_delta=delta)
    print(f"decision> status={decision.status}  reason={decision.reason}")
    print(f"          (use mgr.resume() to continue after extending the cap)")


async def example_4_event_callback() -> None:
    """Subscribe to ``goal.*`` events for tracing / observability."""
    print("\n" + "=" * 60)
    print("Example 4: event_callback hooks tracing / observability")
    print("=" * 60)

    events: list = []

    def on_goal(event_type: RunEventType, payload: dict) -> None:
        events.append((event_type.value, payload.get("status")))

    agent = Agent(
        session_id="goal-demo-events",
        model=DeepSeekChat(),
        auxiliary_model=DeepSeekChat(),
        instructions="Be terse. State 'done' clearly when finished.",
    )
    mgr = GoalManager(
        agent._session_log,
        judge_model=agent.auxiliary_model,
        event_callback=on_goal,
    )
    await run_goal(agent, mgr, "Say hi in one short sentence.")

    print("\ngoal lifecycle events:")
    for ev, status in events:
        print(f"  {ev:18s}  status={status}")


async def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Set DEEPSEEK_API_KEY (or adapt the model factory) to run this demo.")
        return
    await example_1_basic_loop()
    await example_2_with_goal_tool()
    await example_3_token_budget()
    await example_4_event_callback()


if __name__ == "__main__":
    asyncio.run(main())
