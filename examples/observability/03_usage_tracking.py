# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Usage Aggregation Demo - Cross-request token tracking

This example demonstrates the structured Usage model that aggregates
token usage across multiple LLM calls within a single agent run:
1. Per-request usage entries (RequestUsage)
2. Cross-request aggregation (Usage.requests, total_tokens, etc.)
3. Token detail breakdowns (cached_tokens, reasoning_tokens)
4. Multi-turn cumulative tracking
5. Subagent usage merge pattern
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Usage, RequestUsage, TokenDetails


async def single_request_usage():
    """Demo 1: Single request - inspect Usage on RunResponse."""
    print("=" * 60)
    print("Demo 1: Single Request Usage")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="UsageDemo",
        debug=True,
    )

    response = await agent.run("What is 2+2? Answer in one word.")

    print(f"Response: {response.content}")

    # Access structured usage
    if response.usage:
        u = response.usage
        print(f"\n--- Usage ---")
        print(f"  Requests:      {u.requests}")
        print(f"  Input tokens:  {u.input_tokens}")
        print(f"  Output tokens: {u.output_tokens}")
        print(f"  Total tokens:  {u.total_tokens}")

        # Per-request detail entries
        for entry in u.request_usage_entries:
            print(f"\n  [Request #{entry.request_index}]")
            print(f"    Input:  {entry.input_tokens}")
            print(f"    Output: {entry.output_tokens}")
            print(f"    Total:  {entry.total_tokens}")
            if entry.response_time:
                print(f"    Time:   {entry.response_time:.3f}s")

        # Token detail breakdowns
        if u.input_tokens_details.cached_tokens > 0:
            print(f"\n  Cached input tokens: {u.input_tokens_details.cached_tokens}")
        if u.output_tokens_details.reasoning_tokens > 0:
            print(f"  Reasoning tokens:    {u.output_tokens_details.reasoning_tokens}")

        # Serialization
        print(f"\n--- Serialized (dict) ---")
        d = u.model_dump()
        print(f"  {d}")
    else:
        print("  (No usage data available)")


async def multi_turn_usage():
    """Demo 2: Multi-turn conversation - cumulative usage tracking."""
    print("\n" + "=" * 60)
    print("Demo 2: Multi-turn Cumulative Usage")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        name="MultiTurnUsage",
        add_history_to_messages=True,
        debug=True,
    )

    questions = [
        "Name 3 programming languages.",
        "Which one is best for data science?",
        "Give a one-line code example.",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n--- Turn {i}: {q} ---")
        response = await agent.run(q)
        print(f"Response: {response.content[:120]}...")

        if response.usage:
            u = response.usage
            print(f"  Cumulative: {u.requests} requests, {u.total_tokens} total tokens")

            # Show the latest request entry
            if u.request_usage_entries:
                latest = u.request_usage_entries[-1]
                print(f"  Latest request: {latest.input_tokens} in / {latest.output_tokens} out")

    # Final cumulative summary
    if response.usage:
        u = response.usage
        print(f"\n--- Final Summary ---")
        print(f"  Total requests:      {u.requests}")
        print(f"  Total input tokens:  {u.input_tokens}")
        print(f"  Total output tokens: {u.output_tokens}")
        print(f"  Total tokens:        {u.total_tokens}")
        print(f"  Entries count:       {len(u.request_usage_entries)}")


def usage_model_demo():
    """Demo 3: Usage model operations (no LLM call needed)."""
    print("\n" + "=" * 60)
    print("Demo 3: Usage Model Operations")
    print("=" * 60)

    # Build usage manually (e.g., from logged data)
    usage = Usage()

    # Simulate 3 LLM calls
    usage.add(RequestUsage(
        input_tokens=500, output_tokens=100, total_tokens=600,
        response_time=0.8,
        input_tokens_details=TokenDetails(cached_tokens=200),
    ))
    usage.add(RequestUsage(
        input_tokens=800, output_tokens=150, total_tokens=950,
        response_time=1.2,
        output_tokens_details=TokenDetails(reasoning_tokens=50),
    ))
    usage.add(RequestUsage(
        input_tokens=600, output_tokens=200, total_tokens=800,
        response_time=0.9,
    ))

    print(f"  Requests:           {usage.requests}")
    print(f"  Total input:        {usage.input_tokens}")
    print(f"  Total output:       {usage.output_tokens}")
    print(f"  Total:              {usage.total_tokens}")
    print(f"  Cached input:       {usage.input_tokens_details.cached_tokens}")
    print(f"  Reasoning output:   {usage.output_tokens_details.reasoning_tokens}")

    # Merge subagent usage
    subagent_usage = Usage()
    subagent_usage.add(RequestUsage(
        input_tokens=300, output_tokens=80, total_tokens=380,
        response_time=0.5,
    ))
    print(f"\n  Subagent usage: {subagent_usage.requests} requests, {subagent_usage.total_tokens} tokens")

    usage.merge(subagent_usage)
    print(f"  After merge:    {usage.requests} requests, {usage.total_tokens} tokens")

    # Cost estimation
    PRICE_PER_M = {"input": 0.15, "output": 0.60}  # gpt-4o-mini
    cost = (usage.input_tokens / 1e6) * PRICE_PER_M["input"] + \
           (usage.output_tokens / 1e6) * PRICE_PER_M["output"]
    print(f"\n  Estimated cost (gpt-4o-mini): ${cost:.6f}")


async def main():
    await single_request_usage()
    await multi_turn_usage()
    usage_model_demo()


if __name__ == "__main__":
    asyncio.run(main())
