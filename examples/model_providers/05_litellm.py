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
from agentica.model.litellm import LiteLLM


def basic_example():
    """Basic example: Use LiteLLM with OpenAI provider."""
    print("=" * 60)
    print("Example 1: Basic LiteLLM with OpenAI")
    print("=" * 60)
    
    agent = Agent(
        name="Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
    )
    
    response = agent.run_sync("一句话介绍北京")
    print(f"Response:\n{response}")


def tool_calling_example():
    """Example: LiteLLM with tool calling."""
    print("\n" + "=" * 60)
    print("Example 2: LiteLLM with Tool Calling")
    print("=" * 60)
    
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        weather_data = {
            "beijing": "Sunny, 25°C",
            "shanghai": "Cloudy, 22°C",
            "tokyo": "Rainy, 18°C",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")
    
    agent = Agent(
        name="Weather Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
        tools=[get_weather],
        instructions="You are a weather assistant. Use the get_weather tool to answer weather questions.",
    )
    
    response = agent.run_sync("What's the weather like in Beijing?")
    print(f"Response:\n{response.content}")


def streaming_example():
    """Example: LiteLLM with streaming output."""
    print("\n" + "=" * 60)
    print("Example 3: LiteLLM with Streaming")
    print("=" * 60)
    
    agent = Agent(
        name="Streaming Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
    )
    
    print("Streaming response:")
    for chunk in agent.run_sync("Write a short poem about AI.", stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


async def async_example():
    """Example: LiteLLM with async execution."""
    print("\n" + "=" * 60)
    print("Example 4: LiteLLM with Async")
    print("=" * 60)
    
    agent = Agent(
        name="Async Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
    )
    
    queries = ["What is Python?", "What is JavaScript?", "What is Rust?"]
    
    async def run_query(query: str) -> str:
        response = await agent.run(query)
        return f"Q: {query}\nA: {response.content[:100]}..."
    
    results = await asyncio.gather(*[run_query(q) for q in queries])
    
    for result in results:
        print(result)
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LiteLLM Model Provider Demo")
    print("=" * 60 + "\n")
    
    try:
        basic_example()
    except Exception as e:
        print(f"Basic example failed: {e}\n")
    
    try:
        tool_calling_example()
    except Exception as e:
        print(f"Tool calling example failed: {e}\n")
    
    try:
        streaming_example()
    except Exception as e:
        print(f"Streaming example failed: {e}\n")
    
    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"Async example failed: {e}\n")


if __name__ == "__main__":
    main()
