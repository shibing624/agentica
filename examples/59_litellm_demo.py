# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: LiteLLM Model Provider Demo

LiteLLM provides a unified interface to call 100+ LLM APIs using the OpenAI format.
Supported providers: OpenAI, Anthropic, Azure, Huggingface, Cohere, Together, Replicate,
Ollama, Bedrock, Vertex AI, and many more.

Install: pip install litellm

Usage:
    python 59_litellm_demo.py
"""
import sys
import os
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agentica import Agent
from agentica.model.litellm import LiteLLM


def basic_example():
    """Basic example: Use LiteLLM with OpenAI provider."""
    print("=" * 60)
    print("Example 1: Basic LiteLLM with OpenAI")
    print("=" * 60)
    
    # Create agent with LiteLLM model
    # The model id format is: provider/model_name
    agent = Agent(
        name="Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
    )
    
    # Run the agent
    response = agent.run("一句话介绍北京")
    print(f"Response:\n{response}")
    print()


def zhipuai_thinking_example():
    """Example: Use LiteLLM with ZhipuAI thinking model (GLM-4.6)."""
    print("=" * 60)
    print("Example 2: LiteLLM with ZhipuAI Thinking Model (GLM-4.6)")
    print("=" * 60)
    
    import os
    # ZhipuAI requires ZAI_API_KEY environment variable
    # Set it from ZHIPUAI_API_KEY if available
    if os.getenv("ZHIPUAI_API_KEY") and not os.getenv("ZAI_API_KEY"):
        os.environ["ZAI_API_KEY"] = os.getenv("ZHIPUAI_API_KEY")
    
    # Use ZhipuAI GLM-4.6 thinking model via LiteLLM
    # GLM-4.6 is a reasoning model that returns thinking process in reasoning_content
    agent = Agent(
        name="ZhipuAI Thinking Assistant",
        model=LiteLLM(
            id="zai/glm-4.6v-flash",
            max_tokens=2048,
        ),
        instructions="You are a helpful assistant.",
    )
    
    response = agent.run("What is 15 * 23?")
    
    # GLM-4.6 returns thinking process in reasoning_content
    # The content may be empty as the model focuses on reasoning
    if response.content:
        print(f"Response:\n{response}")
    print()


def deepseek_thinking_example():
    """Example: Use LiteLLM with DeepSeek reasoning model."""
    print("=" * 60)
    print("Example 3: LiteLLM with DeepSeek Reasoner (Thinking Mode)")
    print("=" * 60)
    
    # DeepSeek Reasoner supports thinking mode with two methods:
    # Method 1: thinking={"type": "enabled"}
    # Method 2: reasoning_effort="low"/"medium"/"high"
    
    # Method 1: Using thinking parameter
    agent = Agent(
        name="DeepSeek Reasoner",
        model=LiteLLM(
            id="deepseek/deepseek-reasoner",
            thinking={"type": "enabled"},  # Enable thinking mode
            max_tokens=2048,
        ),
        instructions="You are a helpful assistant. Think step by step.",
    )
    
    response = agent.run("What is 25 * 17?")
    
    if response.content:
        print(f"Final Answer:\n{response.content}")
    
    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        thinking = response.reasoning_content
        if len(thinking) > 500:
            thinking = thinking[:500] + "..."
        print(f"\n[Reasoning Process]:\n{thinking}")
    print()


def deepseek_reasoning_effort_example():
    """Example: Use LiteLLM with DeepSeek using reasoning_effort parameter."""
    print("=" * 60)
    print("Example 4: LiteLLM with DeepSeek (reasoning_effort)")
    print("=" * 60)
    
    # Use reasoning_effort parameter: "low", "medium", "high"
    # All values map to thinking enabled internally
    agent = Agent(
        name="DeepSeek Assistant",
        model=LiteLLM(
            id="deepseek/deepseek-reasoner",
            reasoning_effort="high",  # high reasoning effort
            max_tokens=2048,
        ),
        instructions="You are a helpful assistant.",
    )
    
    response = agent.run("Explain the Pythagorean theorem briefly.")
    
    if response.content:
        print(f"Response:\n{response.content}")
    
    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        thinking = response.reasoning_content
        if len(thinking) > 500:
            thinking = thinking[:500] + "..."
        print(f"\n[Reasoning Process]:\n{thinking}")
    print()


def anthropic_example():
    """Example: Use LiteLLM with Anthropic Claude."""
    print("=" * 60)
    print("Example 5: LiteLLM with Anthropic Claude")
    print("=" * 60)
    
    # Use Anthropic Claude via LiteLLM
    # Requires: export ANTHROPIC_API_KEY=your_key
    agent = Agent(
        name="Claude Assistant",
        model=LiteLLM(
            id="anthropic/claude-3-haiku-20240307",
            temperature=0.7,
            max_tokens=1024,
        ),
        instructions="You are a helpful assistant. Be concise.",
    )
    
    response = agent.run("What is the capital of France?")
    print(f"Response:\n{response}")
    print()


def ollama_example():
    """Example: Use LiteLLM with local Ollama models."""
    print("=" * 60)
    print("Example 6: LiteLLM with Ollama (Local)")
    print("=" * 60)
    
    # Use local Ollama model via LiteLLM
    # Requires: ollama running locally with llama2 model
    agent = Agent(
        name="Local Assistant",
        model=LiteLLM(
            id="ollama/llama2",
            api_base="http://localhost:11434",  # Ollama default port
        ),
        instructions="You are a helpful assistant.",
    )
    
    response = agent.run("What is 2 + 2?")
    print(f"Response:\n{response}")
    print()


def tool_calling_example():
    """Example: LiteLLM with tool calling."""
    print("=" * 60)
    print("Example 7: LiteLLM with Tool Calling")
    print("=" * 60)
    
    def get_weather(city: str) -> str:
        """Get the current weather for a city.
        
        Args:
            city: The city name to get weather for.
            
        Returns:
            Weather information string.
        """
        # Mock weather data
        weather_data = {
            "beijing": "Sunny, 25°C",
            "shanghai": "Cloudy, 22°C",
            "tokyo": "Rainy, 18°C",
            "new york": "Partly cloudy, 20°C",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")
    
    agent = Agent(
        name="Weather Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
        tools=[get_weather],
        instructions="You are a weather assistant. Use the get_weather tool to answer weather questions.",
    )
    
    response = agent.run("What's the weather like in Beijing?")
    print(f"Response:\n{response.content}")
    print()


def streaming_example():
    """Example: LiteLLM with streaming output."""
    print("=" * 60)
    print("Example 8: LiteLLM with Streaming")
    print("=" * 60)
    
    agent = Agent(
        name="Streaming Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
        instructions="You are a helpful assistant.",
    )
    
    print("Streaming response:")
    for chunk in agent.run("Write a short poem about AI.", stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


async def async_example():
    """Example: LiteLLM with async execution."""
    print("=" * 60)
    print("Example 9: LiteLLM with Async")
    print("=" * 60)
    
    agent = Agent(
        name="Async Assistant",
        model=LiteLLM(id="openai/gpt-4o-mini"),
        instructions="You are a helpful assistant. Be brief.",
    )
    
    # Run multiple queries concurrently
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]
    
    async def run_query(query: str) -> str:
        response = await agent.arun(query)
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
    
    # Basic example (requires OPENAI_API_KEY)
    try:
        basic_example()
    except Exception as e:
        print(f"Basic example failed: {e}\n")
    
    # ZhipuAI thinking model example (requires ZHIPUAI_API_KEY)
    try:
        zhipuai_thinking_example()
    except Exception as e:
        print(f"ZhipuAI thinking example failed: {e}\n")
    
    # DeepSeek thinking model example (requires DEEPSEEK_API_KEY)
    try:
        deepseek_thinking_example()
    except Exception as e:
        print(f"DeepSeek thinking example failed: {e}\n")
    
    # DeepSeek reasoning_effort example (requires DEEPSEEK_API_KEY)
    try:
        deepseek_reasoning_effort_example()
    except Exception as e:
        print(f"DeepSeek reasoning_effort example failed: {e}\n")
    
    # Tool calling example
    try:
        tool_calling_example()
    except Exception as e:
        print(f"Tool calling example failed: {e}\n")
    
    # Streaming example
    try:
        streaming_example()
    except Exception as e:
        print(f"Streaming example failed: {e}\n")
    
    # Async example
    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"Async example failed: {e}\n")
    
    # Uncomment to run provider-specific examples:
    # anthropic_example()  # Requires ANTHROPIC_API_KEY
    # ollama_example()     # Requires local Ollama


if __name__ == "__main__":
    main()
