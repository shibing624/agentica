# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Standing-goal loop SDK demo (new ergonomic API).

Most users only need one call:

    result = await agent.run_goal("Compute 17+9+16")
    print(result.status, result.reason)
    print(result.response_content)

Budget semantics (token / turn / wall-clock):
    - ``token_budget`` is the primary cost gate and defaults to unlimited
      (``None``) for the SDK, CLI ``/goal``, and Web. Pass a positive int to
      cap spend. Tokens track real work far better than turns — one turn can
      cost 100 or 50_000 tokens.
    - ``turn_budget`` and ``wall_clock_budget_sec`` default to ``None`` (off).
      Pass them only for an extra turn ceiling or an SLA deadline.
    - The caps are **independent** — whichever hits first stops the loop.
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


def _print_budget(result) -> None:
    """Show the budget line the CLI's ``/goal status`` would print."""
    goal = result.goal
    cap = f"{goal.token_budget:,}" if goal.token_budget is not None else "unlimited"
    print(f"  tokens = {goal.tokens_used:,} / {cap}")
    print(f"  turns  = {goal.turns_used} (cap: {goal.turn_budget or 'none'})")


async def example_1_one_liner() -> None:
    """The 90% case: one line drives the entire loop, token budget unlimited by default."""
    print("=" * 60)
    print("Example 1: agent.run_goal()  — the one-liner")
    print("=" * 60)

    # Best practice: strong main model + cheap auxiliary for judge / housekeeping.
    # Auxiliary is called every turn by the judge, so splitting saves 5–10x cost.
    agent = Agent(
        session_id="goal-demo-basic",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash", max_completion_tokens=4096),
        instructions="You are terse. One step per turn. State 'done' when finished.",
    )

    # No budget arguments at all: token_budget stays unlimited and there
    # is no turn cap. This is exactly what CLI users get from `/goal <text>`.
    result = await agent.run_goal("Compute 17+9+16 and state the integer answer.")

    print(f"status        = {result.status}")
    print(f"reason        = {result.reason}")
    print(f"answer        = {result.response_content.strip()[:120]}")
    _print_budget(result)


async def example_2_token_budget() -> None:
    """``token_budget`` is the cost gate: exceeding it yields ``budget_limited``."""
    print("\n" + "=" * 60)
    print("Example 2: token_budget  — the primary cost gate")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-budget",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash", max_completion_tokens=4096),
    )

    # Real-world recipe (commented out — would cost real $$):
    #
    #   result = await agent.run_goal(
    #       "Implement feature X and make pytest pass",
    #       token_budget=200_000,        # optional cap; omit for unlimited
    #       wall_clock_budget_sec=1800,  # optional 30 min SLA
    #   )
    #
    # Here token_budget is deliberately tiny so the very first turn blows the
    # cap — the loop stops with status 'budget_limited' rather than 'paused',
    # meaning "decide whether to raise the cap or accept the partial result".
    result = await agent.run_goal(
        "Summarize TCP slow start in 2 sentences.",
        token_budget=30,
    )

    print(f"status   = {result.status}    # 'budget_limited' (not 'paused')")
    print(f"reason   = {result.reason}")
    _print_budget(result)


async def example_2b_turn_budget_opt_in() -> None:
    """``turn_budget`` is opt-in on top of the always-on token budget."""
    print("\n" + "=" * 60)
    print("Example 2b: turn_budget  — optional extra ceiling")
    print("=" * 60)

    agent = Agent(
        session_id="goal-demo-turns",
        model=DeepSeekChat(id="deepseek-v4-pro"),
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash", max_completion_tokens=4096),
        instructions="Refine one small detail per turn. Do not rush to finish.",
    )

    # token_budget stays unlimited; turn_budget=2 adds a second, independent
    # ceiling. Whichever binds first wins — and the agent can still end earlier
    # by calling verify_completion.
    result = await agent.run_goal(
        "Keep refining a one-line haiku about TCP.",
        turn_budget=2,
    )

    print(f"status   = {result.status}")
    print(f"reason   = {result.reason}")
    _print_budget(result)


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
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash", max_completion_tokens=4096),
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
        auxiliary_model=DeepSeekChat(id="deepseek-v4-flash", max_completion_tokens=4096),
        instructions="You are terse. State 'done' clearly when finished.",
    )
    # default_token_budget applies to every goal this manager sets; a per-goal
    # token_budget= on mgr.set() overrides it.
    mgr = agent.get_goal_manager(default_token_budget=20_000, default_turn_budget=3)
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
            state = mgr.load()
            print(f"[stopped] status={decision.status}")
            print(f"          tokens={state.tokens_used:,}/{state.token_budget:,}")
            break
        prompt = decision.continuation_prompt


async def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Set DEEPSEEK_API_KEY (or adapt the model factory) to run this demo.")
        return
    await example_1_one_liner()
    await example_2_token_budget()
    await example_2b_turn_budget_opt_in()
    await example_3_events()
    await example_4_manual_control()


if __name__ == "__main__":
    asyncio.run(main())
