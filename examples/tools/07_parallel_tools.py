# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Parallel async tool execution demo

Demonstrates the core advantage of async-first architecture:
When the LLM returns multiple tool_calls in one response,
the framework executes them IN PARALLEL via asyncio.TaskGroup —
wall-clock time ≈ max(tool_times) instead of sum(tool_times).

This example:
1. Defines 3 async I/O-bound tools (each simulates 1~2s network latency)
2. Asks the Agent a question that requires ALL 3 tools simultaneously
3. Prints elapsed time — should be ~2s, NOT ~4.5s
"""
import sys
import os
import asyncio
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, logger

# Disable noisy logs for cleaner output
logger.propagate = False


# ============================================================================
# Three async I/O-bound tools — each simulates network latency
# ============================================================================

async def search_weather(city: str) -> str:
    """Search real-time weather for a city.

    Args:
        city: City name, e.g. "Beijing", "Shanghai"

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
        topic: Topic keyword, e.g. "AI", "finance"

    Returns:
        News headlines string
    """
    await asyncio.sleep(1.5)  # simulate API call
    news = {
        "AI": "1. GPT-5 发布在即  2. 开源大模型突破千亿参数  3. AI Agent 成为企业标配",
        "科技": "1. 苹果发布新款 Vision Pro  2. 量子计算突破性进展  3. 6G 标准制定启动",
    }
    return news.get(topic, f"{topic} 相关: 暂无最新新闻")


# ============================================================================
# Main - Demo parallel tool execution
# ============================================================================

async def main():
    print("=" * 60)
    print("Parallel Async Tool Execution Demo")
    print("=" * 60)

    # ---- Part 1: Direct parallel tool calls (guaranteed parallelism) ----
    print("\n--- Demo 1: Direct asyncio.gather (3 tools in parallel) ---")

    tool_count = 3
    delay_per_tool = 1.5
    sequential_time = tool_count * delay_per_tool

    t0 = time.perf_counter()
    results = await asyncio.gather(
        search_weather("北京"),
        search_stock("AAPL"),
        search_news("AI"),
    )
    parallel_elapsed = time.perf_counter() - t0

    print(f"Results:")
    for r in results:
        print(f"  • {r}")
    print(f"\nTiming:")
    print(f"  Sequential estimate: {sequential_time:.1f}s")
    print(f"  Parallel actual: {parallel_elapsed:.2f}s")
    print(f"  Speedup: {sequential_time/parallel_elapsed:.1f}x")

    # ---- Part 2: Agent with parallel tool calls ----
    print("\n--- Demo 2: Agent tool calling (depends on LLM behavior) ---")

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[search_weather, search_stock, search_news],
        instructions=[
            "你是一个信息聚合助手。",
            "当用户问多个问题时，请一次性调用所有需要的工具。",
        ],
    )

    query = "帮我同时查一下：1) 北京今天天气 2) 苹果(AAPL)股价 3) AI最新新闻"

    t0 = time.perf_counter()
    response = await agent.run(query)
    agent_elapsed = time.perf_counter() - t0

    print(f"Query: {query}")
    print(f"Response: {response.content[:200]}...")
    print(f"Agent time: {agent_elapsed:.2f}s")
    print(f"Note: Agent time includes LLM processing + tool calls")
    if response.tools:
        print(f"Tools executed: {len(response.tools)}")

    print(f"\n{'=' * 60}")
    print("Summary: Direct parallel calls achieve near-linear speedup")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
