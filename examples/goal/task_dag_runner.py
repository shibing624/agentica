#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Multi-phase goal demo — run_goal() loops until the objective is
              done, observed turn-by-turn via event_callback.

    tasks_dag.md ──read──▶ agent.run_goal() ──loop──▶ Turn 1: web_search + web_search (parallel)
                                                       Turn 2: synthesize (serial, needs Turn 1)
                                                       judge: done? → stop with status=complete

run_goal() is the standing-goal loop: it runs the agent turn after turn,
feeding each turn to an LLM judge, until the judge says "done" (or a budget
hits). The DAG (parallel vs serial) emerges naturally inside each turn:
  same-turn tool calls  → asyncio.gather (parallel)
  cross-turn tool calls → next turn waits for previous results (serial)

Budgets are generous on purpose: web search returns a lot of tokens, so a
multi-phase research goal needs room. Too small a token_budget stops the loop
after turn 1 with status=budget_limited before the task is actually finished.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from agentica import DeepAgent, DeepSeekChat

HERE = Path(__file__).parent
TASK_FILE = HERE / "tasks_dag.md"
RESULT_FILE = HERE / "task_dag_results.md"


def on_goal_event(event, payload: dict) -> None:
    """One readable line per goal-loop transition.

    The ``event`` is a RunEventType; ``payload`` carries a human-readable
    ``message`` (already prefixed with a status glyph) plus live counters
    (turns_used / turn_budget / tokens_used / token_budget).
    """
    turns = f"turn {payload.get('turns_used')}/{payload.get('turn_budget')}"
    toks = payload.get("tokens_used")
    budget = payload.get("token_budget")
    cost = f", {toks:,}/{budget:,} tok" if toks and budget else ""
    msg = payload.get("message") or payload.get("reason") or ""
    print(f"  [{turns}{cost}] {msg}".rstrip(), flush=True)


async def main() -> None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        print("Set DEEPSEEK_API_KEY env var to run this demo.")
        return

    objective = TASK_FILE.read_text(encoding="utf-8")

    agent = DeepAgent(
        model=DeepSeekChat(id="deepseek-chat", api_key=key),
        # web_search returns compact snippets — enough for a comparison
        # paragraph and bounded in tokens. We deliberately DISABLE fetch_url /
        # file tools here: fetching full doc pages can dump hundreds of KB into
        # context and blow the token budget in a single turn, which is exactly
        # the failure this demo is meant to avoid.
        include_fetch_url=False,
        include_file_tools=False,
        include_execute=False,
        instructions=(
            "Use ONLY web_search (compact snippets). Do not fetch full pages or read files. "
            "Push all independent Phase 1 web_search calls in ONE response (same turn = parallel). "
            "Wait for Phase 1 results before starting Phase 2 (cross-turn = serial). "
            "When all phases are done, call update_goal(status='complete', "
            "final_answer=<the full Phase 1 summaries AND the Phase 2 Chinese "
            "comparison paragraph>). Put the COMPLETE deliverable in final_answer; "
            "do not write a separate 'task done' message afterwards."
        ),
    )

    print(f"Objective loaded ({len(objective)} chars). Running goal loop...\n")

    result = await agent.run_goal(
        objective,
        # Generous budgets: a 2-phase web-research goal needs room. These are
        # safety nets, not the primary stop signal — the judge / update_goal
        # tool decides completion. Shrink them only to demo budget_limited.
        turn_budget=8,
        token_budget=200_000,
        wall_clock_budget_sec=600,
        event_callback=on_goal_event,
    )

    print(f"\nGoal finished: status={result.status}  turns={result.turns_used}  reason={result.reason}")
    print(f"Saving final answer to {RESULT_FILE}")
    RESULT_FILE.write_text(
        f"# Task DAG Result\n\n"
        f"> status={result.status} | turns={result.turns_used} | reason={result.reason}\n\n"
        f"{result.response_content}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
