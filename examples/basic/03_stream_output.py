# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Stream output demo - Demonstrates streaming response

This example shows different ways to use streaming output:
1. Using print_response with stream=True
2. Using run with stream=True and iterating over chunks
3. Async streaming
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat

# Example 1: Simple streaming with print_response
print("=" * 60)
print("Example 1: Streaming with print_response")
print("=" * 60)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
)

agent.print_response("写一首关于春天的短诗", stream=True)

# Example 2: Manual streaming iteration
print("\n" + "=" * 60)
print("Example 2: Manual Streaming Iteration")
print("=" * 60)

print("Response: ", end="")
for chunk in agent.run("用三句话介绍人工智能", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# Example 3: Async streaming with aprint_response
print("\n" + "=" * 60)
print("Example 3: Async Streaming")
print("=" * 60)


async def async_stream_demo():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    
    # Use aprint_response for async streaming output
    await agent.aprint_response("什么是机器学习?", stream=True)


asyncio.run(async_stream_demo())
