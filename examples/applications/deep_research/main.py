# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Research Agent — DeepAgent with 40+ built-in tools demo.

Demonstrates DeepAgent capabilities powered by the Runner agentic loop:
- 40+ built-in tools (file ops, web search, execute code, subagent task, todos)
- Runner agentic loop: LLM ↔ tool-call auto-loop with multi-turn reasoning
- Two-layer compression (tool-result budget → Layer 1 evict → Layer 2
  native/LLM summarise; reactive compact on prompt_too_long)
- Death spiral detection (stops on consecutive all-error tool turns)
- Cost tracking + cost budget control (max_cost_usd)
- Repeated tool-call detection (inject "change strategy" message)
- Workspace memory (AGENT.md, MEMORY.md, daily memory, relevance recall)
- Agentic prompt (enhanced system prompt with self-verification)

Usage:
    python main.py
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import DeepAgent, RunConfig


async def main():
    """Run DeepAgent with sample queries to demonstrate full capabilities."""
    print("=" * 60)
    print("Deep Research Agent Demo")
    print("=" * 60)

    agent = DeepAgent(
        description=(
            "You are a world-class deep research assistant with full system access. "
            "You can search the web, read/write files, execute code, manage tasks, "
            "and conduct multi-step research. Always cite sources and produce reports."
        ),
        debug=True,
    )
    print(f"Model: {agent.model.id}")
    print(f"Tools: {len(agent.get_tools() or [])} loaded")

    # Sample queries demonstrating different capabilities
    queries = [
        # Query 1: Web search + analysis (triggers agentic loop: search → fetch → analyze)
        "Search the web for the latest advances in RAG (Retrieval-Augmented Generation) in 2025, "
        "summarize the top 3 techniques and their pros/cons in a structured table.",

        # Query 2: Code generation + file writing (triggers: write_file → execute → verify)
        "Write a Python script that implements a simple in-memory vector search using cosine similarity, "
        "save it to tmp/vector_search.py, then run it to verify it works.",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Query {i}: {query}")
        print(f"{'=' * 60}\n")

        # Use cost budget to prevent runaway spending (Runner checks per loop iteration)
        response = await agent.run(query)
        print(f"\nResponse:\n{response.content}")

        # Show cost summary (CostTracker is always active in Runner)
        if response.cost_tracker and response.cost_tracker.total_cost_usd > 0:
            print(f"\n--- Cost: ${response.cost_tracker.total_cost_usd:.4f} "
                  f"({response.cost_tracker.total_input_tokens} in + "
                  f"{response.cost_tracker.total_output_tokens} out) ---")


if __name__ == "__main__":
    asyncio.run(main())
