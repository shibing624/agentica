# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: LiteLLM model demo - Demonstrates using LiteLLM for unified model access

LiteLLM provides a unified interface to call 100+ LLM APIs using the OpenAI format.
Supported providers: OpenAI, Anthropic, Azure, Huggingface, Cohere, Together, Replicate,
Ollama, Bedrock, Vertex AI, and many more.

Install: pip install litellm
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.model.litellm import LiteLLMChat


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LiteLLM Model Provider Demo")
    print("=" * 60 + "\n")
    
    # Example 1: Basic
    print("=" * 60)
    print("Example 1: Basic LiteLLM with OpenAI")
    print("=" * 60)
    
    agent = Agent(
        name="Assistant",
        model=LiteLLMChat(id="openai/gpt-4o-mini"),
    )
    
    response = await agent.run("一句话介绍北京")
    print(f"Response:\n{response}")

    # Example 2: Tool calling
    print("\n" + "=" * 60)
    print("Example 2: LiteLLM with Tool Calling")
    print("=" * 60)
    
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        weather_data = {
            "beijing": "Sunny, 25C",
            "shanghai": "Cloudy, 22C",
            "tokyo": "Rainy, 18C",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")
    
    agent2 = Agent(
        name="Weather Assistant",
        model=LiteLLMChat(id="openai/gpt-4o-mini"),
        tools=[get_weather],
        instructions="You are a weather assistant. Use the get_weather tool to answer weather questions.",
    )
    
    response = await agent2.run("What's the weather like in Beijing?")
    print(f"Response:\n{response.content}")

    # Example 3: Streaming
    print("\n" + "=" * 60)
    print("Example 3: LiteLLM with Streaming")
    print("=" * 60)
    
    agent3 = Agent(
        name="Streaming Assistant",
        model=LiteLLMChat(id="openai/gpt-4o-mini"),
    )
    
    print("Streaming response:")
    async for chunk in agent3.run_stream("Write a short poem about AI."):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    # Example 4: Concurrent async
    print("\n" + "=" * 60)
    print("Example 4: LiteLLM with Async Concurrency")
    print("=" * 60)
    
    queries = ["What is Python?", "What is JavaScript?", "What is Rust?"]
    
    async def run_query(query: str) -> str:
        response = await agent.run(query)
        return f"Q: {query}\nA: {response.content[:100]}..."
    
    results = await asyncio.gather(*[run_query(q) for q in queries])
    
    for result in results:
        print(result)
        print()


if __name__ == "__main__":
    asyncio.run(main())
