# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Session history demo — conversation history and session summaries

Demonstrates:
1. No history — each turn is independent
2. With history — add_history_to_messages enables multi-turn context
3. Session summary — AgentMemory.with_summary() for compressed history
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, AgentMemory


async def main():
    # ============================================================
    # Part 1: Basic Agent (no history)
    # ============================================================
    print("=" * 60)
    print("Part 1: Basic Agent (no history)")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(),
        add_history_to_messages=False,
    )

    r = await agent.run("李四住在北京，一家三口住大别墅")
    print(r)
    r = await agent.run("我前面问了啥")
    print(r)
    print("(Agent has no history, so it doesn't know what you asked before)")

    # ============================================================
    # Part 2: Agent with History
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 2: Agent with History (add_history_to_messages=True)")
    print("=" * 60)

    agent2 = Agent(
        model=OpenAIChat(),
        add_history_to_messages=True,
        num_history_responses=5,
    )

    print("\n--- Multi-turn conversation ---")
    print(await agent2.run("My name is Alice and I'm a software engineer."))
    print(await agent2.run("I like hiking on weekends."))
    print(await agent2.run("What's my name and what do I like?"))

    # ============================================================
    # Part 3: Agent with Session Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Part 3: Agent with Session Summary")
    print("=" * 60)

    memory = AgentMemory.with_summary()
    agent3 = Agent(
        model=OpenAIChat(),
        memory=memory,
        add_history_to_messages=True,
    )

    print("\n--- Conversation ---")
    print(await agent3.run("我叫张三，是一名软件工程师，住在北京"))
    print(await agent3.run("我最喜欢的电影是《花样年华》"))
    print(await agent3.run("我喜欢打篮球和游泳"))

    # Generate session summary
    summary = await memory.update_summary()
    if summary:
        print(f"\n--- Session Summary ---")
        print(f"Summary: {summary.summary}")
        if summary.topics:
            print(f"Topics: {', '.join(summary.topics)}")

    # The summary is now available in memory for future runs
    print(await agent3.run("你能总结下我们聊了什么吗？"))

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
