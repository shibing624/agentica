# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Swarm multi-agent parallel collaboration demo

Run with:
    OPENAI_API_KEY=... python examples/agent_patterns/08_swarm.py

Demonstrates Swarm — a peer-to-peer autonomous collaboration system. Compared
with the lighter ``Agent.as_tool()`` composition primitive and ``Workflow``
(deterministic, hand-wired pipeline), Swarm fans out subtasks across multiple
worker agents and merges their results:

1. **Parallel mode**: All agents run the same task concurrently, results are merged.
2. **Autonomous mode**: A coordinator decomposes tasks, assigns to workers, and a
   synthesizer combines the worker outputs into the final answer.
"""
import sys
import os
from textwrap import shorten

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from agentica import Agent, OpenAIChat
from agentica.orchestration.swarm import Swarm


MODEL_ID = "gpt-4o-mini"


def _build_model() -> OpenAIChat:
    """Build a fresh model for each demo agent.

    Using fresh Agent instances per demo keeps working memory and runtime state
    isolated, so the parallel example does not leak context into the autonomous
    example when this file is run top-to-bottom.
    """
    return OpenAIChat(id=MODEL_ID)


def _build_specialists():
    """Create a fresh set of specialist agents for one demo run."""
    return {
        "researcher": Agent(
            name="researcher",
            model=_build_model(),
            description="Research specialist focused on factual analysis and source discovery.",
            instructions="Provide factual, well-scoped analysis. Call out uncertainty explicitly.",
        ),
        "coder": Agent(
            name="coder",
            model=_build_model(),
            description="Engineering specialist focused on clean Python solutions.",
            instructions="Write practical, readable Python guidance with implementation details.",
        ),
        "reviewer": Agent(
            name="reviewer",
            model=_build_model(),
            description="Review specialist focused on correctness, risk, and missing tests.",
            instructions="Stress-test proposals, identify risks, and suggest focused improvements.",
        ),
        "coordinator": Agent(
            name="coordinator",
            model=_build_model(),
            description="Task coordinator that decomposes requests and routes work to the best worker.",
        ),
    }


def _print_header(title: str, description: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(description)


def _preview(text: str, width: int = 88) -> str:
    return shorten(text.replace("\n", " "), width=width, placeholder="...")


def _print_agent_results(agent_results) -> None:
    print(f"Agent results: {len(agent_results)}")
    for result in agent_results:
        status = "OK" if result.get("success") else "FAILED"
        name = result.get("agent", "?")
        subtask = result.get("subtask")
        token_count = result.get("total_tokens", 0)
        extra = f" [{token_count} tokens]" if token_count else ""
        if subtask:
            extra += f" subtask={_preview(subtask, width=56)}"
        print(f"  - {name}: [{status}]{extra}")
        print(f"    {_preview(str(result.get('content', '')), width=100)}")


# ---------------------------------------------------------------------------
# 1. Parallel mode demo
# ---------------------------------------------------------------------------

async def demo_parallel():
    """All agents run the same task, results are merged."""
    agents = _build_specialists()
    _print_header(
        "Demo 1: Swarm Parallel Mode",
        "Each worker answers the same question independently; Swarm then synthesizes the results.",
    )

    swarm = Swarm(
        agents=[agents["researcher"], agents["coder"], agents["reviewer"]],
        mode="parallel",
    )

    result = await swarm.run("What are the top 3 best practices for building AI agents?")

    print(f"Mode: {result.mode}")
    print(f"Time: {result.total_time:.3f}s")
    _print_agent_results(result.agent_results)
    print(f"\nSynthesized response:\n{result.content}\n")


# ---------------------------------------------------------------------------
# 2. Autonomous mode demo
# ---------------------------------------------------------------------------

async def demo_autonomous():
    """Coordinator decomposes task, assigns to workers, synthesizer merges."""
    agents = _build_specialists()
    _print_header(
        "Demo 2: Swarm Autonomous Mode",
        "The coordinator decomposes the task, the workers execute in parallel, and the reviewer synthesizes the final answer.",
    )

    swarm = Swarm(
        agents=[agents["researcher"], agents["coder"]],
        mode="autonomous",
        coordinator=agents["coordinator"],
        synthesizer=agents["reviewer"],
        max_concurrent=2,
    )

    result = await swarm.run(
        "Build a Python function that fetches weather data from an API, "
        "with proper error handling and tests."
    )

    print(f"Mode: {result.mode}")
    print(f"Time: {result.total_time:.3f}s")
    _print_agent_results(result.agent_results)
    print(f"\nSynthesized response:\n{result.content}\n")


# ---------------------------------------------------------------------------
# 3. Run all demos
# ---------------------------------------------------------------------------

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "This demo requires OPENAI_API_KEY in the environment. "
            "Example: OPENAI_API_KEY=... python examples/agent_patterns/08_swarm.py"
        )

    print("Agentica Swarm Demo\n")
    print(f"Model: {MODEL_ID}")
    print("Note: Swarm mode is configured on Swarm(..., mode=...), not on swarm.run().")

    await demo_parallel()
    await demo_autonomous()

    print("\n" + "=" * 72)
    print("Swarm vs Agent.as_tool() vs Workflow")
    print("=" * 72)
    print("""
    Swarm           — Multi-agent fan-out + synthesis (parallel / autonomous)
    Agent.as_tool() — Lightweight black-box delegation (single tool call)
    Workflow        — Deterministic pipeline with fixed execution order
    """)


if __name__ == "__main__":
    asyncio.run(main())
