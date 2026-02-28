# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Session history demo — tool call memory across turns

Demonstrates:
1. No history — agent forgets tool calls
2. With history — tool call context preserved across turns
3. Session summary — compressed history with tool context
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, WorkingMemory


def get_weather(city: str) -> str:
    """Get current weather for a city."""
    weather_data = {
        "北京": "晴天，气温28°C，湿度45%",
        "上海": "多云，气温26°C，湿度70%",
        "深圳": "小雨，气温30°C，湿度85%",
        "杭州": "阴天，气温24°C，湿度60%",
    }
    return weather_data.get(city, f"{city}：暂无天气数据")


def get_population(city: str) -> str:
    """Get population info for a city."""
    pop_data = {
        "北京": "北京常住人口约2189万人",
        "上海": "上海常住人口约2487万人",
        "深圳": "深圳常住人口约1768万人",
        "杭州": "杭州常住人口约1237万人",
    }
    return pop_data.get(city, f"{city}：暂无人口数据")


async def main():
    # ============================================================
    # Part 1: No history — agent forgets tool calls
    # ============================================================
    print("=" * 60)
    print("Part 1: No history — agent forgets tool calls")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(),
        tools=[get_weather, get_population],
        add_history_to_messages=False,
    )

    r = await agent.run("帮我查一下北京的天气")
    print(r)
    r = await agent.run("我刚才让你查了哪个城市的什么信息？用了什么工具？")
    print(r)
    print("(Agent has no history, so it doesn't remember the tool call)")

    # ============================================================
    # Part 2: With history — tool call context preserved
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 2: With history — tool call context preserved")
    print("=" * 60)

    agent2 = Agent(
        model=OpenAIChat(),
        tools=[get_weather, get_population],
        add_history_to_messages=True,
        history_window=5,
        debug=True,
    )

    print("\n--- Turn 1: query weather ---")
    print(await agent2.run("帮我查一下北京的天气"))

    print("\n--- Turn 2: query population ---")
    print(await agent2.run("再查一下上海的人口"))

    print("\n--- Turn 3: ask about previous tool calls ---")
    print(await agent2.run("我前面让你查了哪些城市的什么信息？分别用了什么工具？请列出来"))

    # ============================================================
    # Part 3: Session summary with tool context
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 3: Session summary with tool context")
    print("=" * 60)

    memory = WorkingMemory.with_summary()
    agent3 = Agent(
        model=OpenAIChat(),
        tools=[get_weather, get_population],
        working_memory=memory,
        add_history_to_messages=True,
    )

    print(await agent3.run("查一下深圳的天气和杭州的人口"))
    print(await agent3.run("对比一下刚才查到的深圳天气和杭州人口数据，哪个城市更适合居住？"))

    summary = await memory.update_summary()
    if summary:
        print(f"\n--- Session Summary ---")
        print(f"Summary: {summary.summary}")
        if summary.topics:
            print(f"Topics: {', '.join(summary.topics)}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
