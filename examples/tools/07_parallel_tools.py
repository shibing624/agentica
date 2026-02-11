# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Parallel async tool execution demo

Demonstrates the core advantage of async-first tool architecture:

  Parallel:   wall-clock ≈ max(tool_times)  ← asyncio.TaskGroup in framework
  Sequential: wall-clock ≈ sum(tool_times)  ← one-by-one await

This example:
1. Defines 3 async I/O-bound tools (each simulates 1.5s network latency)
2. Runs them sequentially vs in parallel, printing wall-clock comparison
3. Shows Agent-level parallel tool calling with one-line metrics access
"""
import sys
import os
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# Three async I/O-bound tools — each simulates 1.5s network latency
# ============================================================================

async def search_weather(city: str) -> str:
    """Search real-time weather for a city.

    Args:
        city: City name, e.g. "北京", "上海"

    Returns:
        Weather information string
    """
    await asyncio.sleep(1.5)  # simulate API call
    weathers = {
        "北京": "晴, 28°C, 湿度 35%",
        "上海": "多云, 31°C, 湿度 72%",
        "深圳": "雷阵雨, 33°C, 湿度 85%",
    }
    return weathers.get(city, f"{city}: 晴, 25°C")


async def search_stock(symbol: str) -> str:
    """Search latest stock price.

    Args:
        symbol: Stock symbol, e.g. "AAPL", "GOOGL", "MSFT"

    Returns:
        Stock price information string
    """
    await asyncio.sleep(1.5)  # simulate API call
    prices = {
        "AAPL": "$198.50 (+1.2%)",
        "GOOGL": "$175.30 (-0.5%)",
        "MSFT": "$420.10 (+0.8%)",
    }
    return prices.get(symbol, f"{symbol}: $100.00")


async def search_news(topic: str) -> str:
    """Search latest news headlines for a topic.

    Args:
        topic: Topic keyword, e.g. "AI", "科技"

    Returns:
        News headlines string
    """
    await asyncio.sleep(1.5)  # simulate API call
    news = {
        "AI": "1. GPT-5 发布在即  2. 开源大模型突破千亿参数  3. AI Agent 成为企业标配",
        "科技": "1. 苹果发布新款 Vision Pro  2. 量子计算突破性进展  3. 6G 标准制定启动",
    }
    return news.get(topic, f"{topic}: 暂无最新新闻")


# ============================================================================
# Demo 1: Pure tool-level comparison (no LLM, deterministic)
# ============================================================================

async def demo_tool_parallelism():
    """Compare sequential vs parallel execution of 3 async tools."""
    print("=" * 60)
    print("Demo 1: Tool-Level Parallel Execution")
    print("=" * 60)
    print("3 async tools, each with 1.5s simulated I/O latency\n")

    # --- Sequential ---
    t0 = time.perf_counter()
    r1 = await search_weather("北京")
    r2 = await search_stock("AAPL")
    r3 = await search_news("AI")
    seq_time = time.perf_counter() - t0
    print(f"[Sequential] {seq_time:.2f}s")
    print(f"  weather: {r1}")
    print(f"  stock:   {r2}")
    print(f"  news:    {r3}")

    # --- Parallel ---
    t0 = time.perf_counter()
    r1, r2, r3 = await asyncio.gather(
        search_weather("北京"),
        search_stock("AAPL"),
        search_news("AI"),
    )
    par_time = time.perf_counter() - t0
    print(f"\n[Parallel]   {par_time:.2f}s")
    print(f"  weather: {r1}")
    print(f"  stock:   {r2}")
    print(f"  news:    {r3}")

    print(f"\n  Speedup: {seq_time / par_time:.1f}x "
          f"({seq_time:.2f}s → {par_time:.2f}s)")
    print(f"  Saved:   {seq_time - par_time:.2f}s\n")


# ============================================================================
# Demo 2: Agent-level parallel tool calling (requires LLM API)
# ============================================================================

async def demo_agent_parallel():
    """Show Agent parallel tool execution with tool_calls convenience API."""
    from agentica import Agent, OpenAIChat

    print("=" * 60)
    print("Demo 2: Agent Parallel Tool Calling")
    print("=" * 60)
    print("Agent → LLM returns 3 tool_calls → framework runs in parallel\n")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[search_weather, search_stock, search_news],
        instructions=[
            "你是一个信息聚合助手。",
            "当用户同时问多个不同来源的信息时，一次性调用所有需要的工具。",
        ],
    )

    query = "帮我同时查：北京天气、AAPL股价、AI最新新闻"

    response = await agent.run(query)
    print(response.content)

    # ---- Flat attribute access via response.tool_calls ----
    if response.tool_calls:
        print(f"\n  Tool call count: {response.tool_call_count}")
        for t in response.tool_calls:
            status = "ERROR" if t.is_error else "OK"
            preview = str(t.content)[:50] if t.content else ""
            print(f"  {t.tool_name}({t.tool_args}) → {preview}  [{t.elapsed:.2f}s] {status}")

        sum_t = sum(t.elapsed for t in response.tool_calls)
        max_t = max(t.elapsed for t in response.tool_calls)
        print(f"\n  sum(tool_times) = {sum_t:.2f}s  ← sequential would take this")
        print(f"  max(tool_times) = {max_t:.2f}s  ← parallel actual ≈ this")
        print(f"  Tool speedup:     {sum_t / max_t:.1f}x")
    print()


# ============================================================================
# Main
# ============================================================================

async def main():
    # Demo 1 always runs (no external dependencies)
    await demo_tool_parallelism()

    # Demo 2 requires OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        await demo_agent_parallel()
    else:
        print("=" * 60)
        print("Demo 2: Skipped (set OPENAI_API_KEY to run Agent demo)")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
