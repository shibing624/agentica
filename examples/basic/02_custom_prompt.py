# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Custom prompt demo - Demonstrates how to customize system prompt and user messages

This example shows:
1. Custom system prompt (instructions)
2. Custom user messages
3. Message list input format
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat


async def main():
    # Example 1: Custom system prompt using instructions
    print("=" * 60)
    print("Example 1: Custom System Prompt")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions="你是一个专业的中国历史专家，请用简洁的语言回答问题。",
        debug=True,
    )

    await agent.print_response_stream("介绍一下唐朝")

    # Example 2: Custom user messages with message list input
    print("\n" + "=" * 60)
    print("Example 2: Message List Input")
    print("=" * 60)

    agent2 = Agent(
        model=OpenAIChat(id="gpt-4o-mini", stop="</answer>"),
        debug=True,
    )

    response = await agent2.run(
        messages=[
            {"role": "user", "content": "What is the color of a banana? Provide your answer in the xml tag <answer>."},
            {"role": "assistant", "content": "<answer>"},
        ],
    )
    print(response)

    # Example 3: Using description for agent persona
    print("\n" + "=" * 60)
    print("Example 3: Agent with Description")
    print("=" * 60)

    agent3 = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        description="You are a helpful coding assistant specialized in Python.",
        instructions=[
            "Always provide code examples when explaining concepts.",
            "Use clear variable names and add comments.",
        ],
    )

    await agent3.print_response_stream("如何用Python实现快速排序?")


if __name__ == "__main__":
    asyncio.run(main())
