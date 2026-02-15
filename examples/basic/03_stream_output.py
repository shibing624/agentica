# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Stream output demo - Demonstrates streaming response

This example shows different ways to use streaming output:
1. Using print_response_stream_sync (sync)
2. Using run_stream_sync and iterating over chunks (sync)
3. Async streaming with print_response
4. Async streaming with run_stream (recommended for async streaming)
5. Async non-streaming with run
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, DeepSeekChat

# Example 1: Simple streaming with print_response
print("=" * 60)
print("Example 1: Streaming with print_response")
print("=" * 60)

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
)

agent.print_response_stream_sync("hi")

# Example 2: Manual streaming iteration (sync)
print("\n" + "=" * 60)
print("Example 2: Manual Streaming Iteration (sync)")
print("=" * 60)

print("Response: ", end="")
for chunk in agent.run_stream_sync("hi"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()

# Example 3: Async streaming with print_response
print("\n" + "=" * 60)
print("Example 3: Async Streaming with print_response")
print("=" * 60)


async def async_print_demo():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    
    # Use print_response_stream for async streaming output
    await agent.print_response_stream("hi")


asyncio.run(async_print_demo())

# Example 4: Async streaming with run_stream (recommended)
print("\n" + "=" * 60)
print("Example 4: Async Streaming with run_stream (recommended)")
print("=" * 60)


async def async_stream_demo():
    agent = Agent(
        model=DeepSeekChat(id='deepseek-reasoner'),
    )
    
    # Use run_stream for async streaming - this is the recommended way
    # run_stream is an async generator, can be used directly with async for
    reasoning_printed = False
    content_started = False
    async for chunk in agent.run_stream("9.18比9.01大吗"):
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

asyncio.run(async_stream_demo())

# Example 5: Async non-streaming with run
print("\n" + "=" * 60)
print("Example 5: Async Non-Streaming with run")
print("=" * 60)


async def async_run_demo():
    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    
    # Use run without streaming - returns a single RunResponse
    response = await agent.run("hi")
    print(f"Response: {response.content}")


asyncio.run(async_run_demo())
