# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Standing-goal loop SDK demo (new ergonomic API).

Most users only need one call:

    result = await agent.run_goal("Compute 17+9+16", turn_budget=3)
    print(result.status, result.reason)
    print(result.response_content)

Budget semantics (turn / token / wall-clock):
    - The 3 budgets are **independent hard caps** — whichever hits first stops
      the loop ("AND / intersection" semantics: every cap must stay under the
      limit for the loop to keep going).
    - ``None`` means "do not enforce". ``turn_budget`` falls back to
      ``DEFAULT_TURN_BUDGET = 100`` as a runaway safety net and cannot be fully
      disabled — pass a large number (e.g. 10_000) instead.
    - Priority on each turn: budget > tool short-circuit > judge. budget caps
      are hard and override the model's own ``update_goal`` signal.

`Agent.run_goal()` internally:
    - lazily creates the SessionLog and GoalManager
    - binds TaskAnchor to the objective
    - attaches GoalTool so the model can self-mark complete/paused
    - loops, feeding token + wall-clock deltas back to the manager
    - stops on complete / paused / budget_limited and returns
      a flat ``GoalRunResult``

Power-users can still grab the manager via ``agent.get_goal_manager()``
and drive the loop by hand — that's Example 4 below.

Requires:  DEEPSEEK_API_KEY  (or swap in any other Agentica model).
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, DeepSeekChat
from agentica.run_events import RunEventType


async def example_1_one_liner() -> None:
    """The 90% case: one line drives the entire loop."""
    print("=" * 60)
    print("Example 1: agent.run_goal()  — the one-liner")
    print("=" * 60)

    # Best practice: strong main model + cheap aux for judge / housekeeping.
    # Aux is called every turn by the judge, so splitting saves 5–10x cost.
    agent = Agent(
        session_id="goal-demo-basic",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash"),
        instructions="You are terse. One step per turn. State 'done' when finished.",
    )

    # Trivial task → don't bother with token/wall-clock caps. turn_budget
    # alone is enough; omit it entirely to use the default 100.
    result = await agent.run_goal(
        "Compute 17+9+16 and state the integer answer.",
        turn_budget=3,
    )

    print(f"status        = {result.status}")
    print(f"reason        = {result.reason}")
    print(f"turns_used    = {result.turns_used}")
    print(f"answer        = {result.response_content.strip()[:120]}")


async def example_2_budgets() -> None:
    """Hard token / wall-clock caps — status becomes ``budget_limited``."""
    print("\n" + "=" * 60)
    print("Example 2: budgets  — token_budget / wall_clock_budget_sec")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-budget",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash"),
    )

    # Real-world recipe (commented out — would cost real $$):
    #
    #   result = await agent.run_goal(
    #       "Implement feature X and make pytest pass",
    #       token_budget=200_000,        # ~30 turns of typical coding work
    #       wall_clock_budget_sec=1800,  # 30 min SLA
    #   )                                 # turn_budget falls back to default 100
    #
    # Here we deliberately set token_budget tiny so the loop hits the cap
    # on the first turn — proves status becomes 'budget_limited'.
    result = await agent.run_goal(
        "Summarize TCP slow start in 2 sentences.",
        token_budget=30,
        wall_clock_budget_sec=120,
    )

    print(f"status   = {result.status}    # 'budget_limited' (not 'paused')")
    print(f"reason   = {result.reason}")
    print(f"tokens   = {result.goal.tokens_used} / {result.goal.token_budget}")


async def example_3_events() -> None:
    """Subscribe to ``goal.*`` events for tracing / observability."""
    print("\n" + "=" * 60)
    print("Example 3: event_callback hooks tracing layer")
    print("=" * 60)

    events: list = []

    def on_goal(event_type: RunEventType, payload: dict) -> None:
        events.append((event_type.value, payload.get("status")))

    agent = Agent(
        session_id="goal-demo-events",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash"),
        instructions="Be terse. State 'done' clearly when finished.",
    )

    await agent.run_goal(
        "Say hi in one short sentence.",
        event_callback=on_goal,
    )

    print("goal lifecycle events:")
    for ev, status in events:
        print(f"  {ev:18s}  status={status}")


async def example_4_manual_control() -> None:
    """Power-user path: keep ``run_goal()``'s ergonomics but drive turns
    yourself when you need per-turn side effects (custom logging, UI,
    streaming-aware progress bars, etc.).
    """
    print("\n" + "=" * 60)
    print("Example 4: manual loop via agent.get_goal_manager()")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-manual",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash"),
        instructions="You are terse. State 'done' clearly when finished.",
    )
    mgr = agent.get_goal_manager(default_turn_budget=3)
    agent.enable_goal_tool()

    mgr.set("Greet the user in exactly one sentence.")
    prompt = mgr.load().objective

    while True:
        resp = await agent.run(prompt)
        print(f"[turn {mgr.load().turns_used + 1}] agent> {(resp.content or '').strip()[:100]}")

        ct = resp.cost_tracker
        delta = (ct.total_input_tokens + ct.total_output_tokens) if ct else 0
        decision = await mgr.evaluate_after_turn(resp.content or "", token_delta=delta)
        print(f"          judge> {decision.verdict}: {decision.reason[:120]}")

        if not decision.should_continue:
            print(f"[stopped] status={decision.status}")
            break
        prompt = decision.continuation_prompt


async def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Set DEEPSEEK_API_KEY (or adapt the model factory) to run this demo.")
        return
    await example_1_one_liner()
    await example_2_budgets()
    await example_3_events()
    await example_4_manual_control()


if __name__ == "__main__":
    asyncio.run(main())
