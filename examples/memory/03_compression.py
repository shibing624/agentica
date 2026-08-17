# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Token compression demo - Demonstrates token counting and compression

This example shows:
1. Token counting for messages, tools, and multi-modal content
2. Two-layer context compression: free eviction, then LLM summarisation
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Message
from agentica.agent.config import ToolConfig
from agentica.utils.tokens import (
    count_tokens,
    count_text_tokens,
    count_message_tokens,
    count_tool_tokens,
)
from agentica.compression import CompressionManager, evict_context


def demo_token_counting():
    """Demonstrate token counting functionality."""
    print("=" * 60)
    print("Token Counting Demo")
    print("=" * 60)

    # 1. Count text tokens
    text = "Hello, this is a test message for token counting."
    tokens = count_text_tokens(text, model_id="gpt-4o")
    print(f"\n1. Text token counting:")
    print(f"   Text: '{text}'")
    print(f"   Tokens: {tokens}")

    # 2. Count message tokens
    message = Message(role="user", content="What is the weather like today in Beijing?")
    msg_tokens = count_message_tokens(message, model_id="gpt-4o")
    print(f"\n2. Message token counting:")
    print(f"   Message: {message.content}")
    print(f"   Tokens: {msg_tokens}")

    # 3. Count multiple messages
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Tell me a joke."),
        Message(role="assistant", content="Why don't scientists trust atoms? Because they make up everything!"),
    ]
    total_tokens = count_tokens(messages, model_id="gpt-4o")
    print(f"\n3. Multiple messages token counting:")
    print(f"   Number of messages: {len(messages)}")
    print(f"   Total tokens: {total_tokens}")

    # 4. Count tool tokens
    def get_weather(city: str, unit: str = "celsius") -> str:
        """Get the current weather for a city."""
        return f"Weather in {city}: 25°C, sunny"

    from agentica.tools.base import Function
    weather_func = Function.from_callable(get_weather)
    tool_tokens = count_tool_tokens([weather_func], model_id="gpt-4o")
    print(f"\n4. Tool token counting:")
    print(f"   Tool: get_weather")
    print(f"   Tokens: {tool_tokens}")


def demo_layer1_eviction():
    """Layer 1: reclaim context without an LLM call."""
    print("\n" + "=" * 60)
    print("Layer 1: tool-result eviction (free)")
    print("=" * 60)

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Search for information about AI."),
        Message(
            role="tool",
            tool_call_id="call_1",
            content="""According to recent market analysis, Artificial Intelligence 
            has made significant announcements in the technology sector. The field has seen 
            tremendous growth with major companies like OpenAI, Google, and Anthropic leading the way.
            OpenAI released GPT-4 in March 2023, which represents a significant advancement.""",
            tool_name="search_web",
        ),
        Message(
            role="tool",
            tool_call_id="call_2",
            content="""The history of artificial intelligence dates back to the 1950s when Alan Turing
            proposed the Turing Test. The term 'Artificial Intelligence' was coined by John McCarthy
            in 1956 at the Dartmouth Conference. Early AI research focused on symbolic reasoning.""",
            tool_name="search_history",
        ),
        Message(role="assistant", content="Let me look at one more source."),
        Message(role="tool", tool_call_id="call_3", tool_name="search_papers",
                content="The newest round of results, which the model has not read yet."),
    ]

    print("\n1. Before eviction:")
    for msg in messages:
        if msg.role == "tool":
            print(f"   {msg.tool_name}: {len(str(msg.content))} chars")

    # Roomy window: nothing is evicted, because a result the window had room
    # for costs a tool re-run to get back.
    roomy = evict_context(messages, context_tokens=1_000, context_window=100_000)
    print(f"\n2. Roomy window (1k/100k): evicted {roomy.tool_results}")

    # Tight window: oldest first, down to the target. The trailing round is
    # never touched — the model has not seen it yet.
    tight = evict_context(messages, context_tokens=900, context_window=1_000)
    print(f"3. Tight window (900/1k):   evicted {tight.tool_results}")

    print("\n4. After eviction:")
    for msg in messages:
        if msg.role == "tool":
            print(f"   {msg.tool_name}: {str(msg.content)[:70]}")


def demo_agent_with_compression():
    """Both layers are on by default; only Layer 2 is configurable."""
    print("\n" + "=" * 60)
    print("Agent with Compression Demo")
    print("=" * 60)

    agent1 = Agent(model=OpenAIChat(id="gpt-4o"), name="DefaultAgent")
    print("\n1. Default agent — Layer 2 is wired automatically:")
    print(f"   Manager: {type(agent1.tool_config.compression_manager).__name__}")
    print(f"   enable_evict={agent1.tool_config.enable_evict} "
          f"enable_auto_compact={agent1.tool_config.enable_auto_compact}")

    custom_compression = CompressionManager(
        model=OpenAIChat(id="gpt-4o-mini"),
        compress_token_limit=5000,
    )
    Agent(
        model=OpenAIChat(id="gpt-4o"),
        tool_config=ToolConfig(compression_manager=custom_compression),
        name="CustomCompressedAgent",
    )
    print("\n2. Agent with custom CompressionManager:")
    print(f"   Token limit: {custom_compression.compress_token_limit}")


if __name__ == "__main__":
    print("Agentica Token Counting & Compression Demo")
    print("=" * 60)

    demo_token_counting()
    demo_layer1_eviction()

    if os.getenv("OPENAI_API_KEY"):
        demo_agent_with_compression()
    else:
        print("\n[INFO] Set OPENAI_API_KEY to run the agent demo")
