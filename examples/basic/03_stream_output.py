# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Stream output demo - Demonstrates streaming response

This example shows different ways to use streaming output:
1. Using print_response with stream=True (sync)
2. Using run with stream=True and iterating over chunks (sync)
3. Async streaming with aprint_response
4. Async streaming with arun_stream (recommended for async streaming)
5. Async non-streaming with arun
"""
import sys
import os
import asyncio

# Insert at beginning to ensure local version is used
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, DeepSeek

# Example 1: Simple streaming with print_response
print("=" * 60)
print("Example 1: Streaming with print_response")
print("=" * 60)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
)

agent.print_response("hi", stream=True)

# Example 2: Manual streaming iteration (sync)
print("\n" + "=" * 60)
print("Example 2: Manual Streaming Iteration (sync)")
print("=" * 60)

print("Response: ", end="")
for chunk in agent.run("hi", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# Example 3: Async streaming with aprint_response
print("\n" + "=" * 60)
print("Example 3: Async Streaming with aprint_response")
print("=" * 60)


async def async_aprint_demo():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    
    # Use aprint_response for async streaming output
    await agent.aprint_response("hi", stream=True)


asyncio.run(async_aprint_demo())

# Example 4: Async streaming with arun_stream (recommended)
print("\n" + "=" * 60)
print("Example 4: Async Streaming with arun_stream (recommended)")
print("=" * 60)


async def async_arun_stream_demo():
    agent = Agent(
        model=DeepSeek(id='deepseek-reasoner'),
    )
    
    # Use arun_stream for async streaming - this is the recommended way
    # arun_stream is an async generator, can be used directly with async for
    reasoning_printed = False
    content_started = False
    async for chunk in agent.arun_stream("9.18比9.01大吗"):
        # Stream reasoning content (thinking process)
        if chunk.reasoning_content:
            if not reasoning_printed:
                print("💭 Thinking:")
                reasoning_printed = True
            print(chunk.reasoning_content, end="", flush=True)
        # Stream final content
        if chunk.content:
            if not content_started:
                print("\n\n--- Final Response ---")
                content_started = True
            print(chunk.content, end="", flush=True)
    print()

asyncio.run(async_arun_stream_demo())

# Example 5: Async non-streaming with arun
print("\n" + "=" * 60)
print("Example 5: Async Non-Streaming with arun")
print("=" * 60)


async def async_arun_non_stream_demo():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    
    # Use arun without streaming - returns a single RunResponse
    response = await agent.arun("hi")
    print(f"Response: {response.content}")


asyncio.run(async_arun_non_stream_demo())
