# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Token compression demo - Demonstrates token counting and compression

This example shows:
1. Token counting for messages, tools, and multi-modal content
2. One compaction trigger: prune old tool results to head + tail, else a new window
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
from agentica.compression import CompressionManager, prune_tool_results


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


def demo_layer1_prune():
    """Layer 1: at the compaction line, prune old results to head + tail."""
    print("\n" + "=" * 60)
    print("Layer 1: tool-result prune (no LLM)")
    print("=" * 60)

    long_page = " ".join(f"AI-history-fact-{i}" for i in range(1500))
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Search for information about AI."),
        Message(role="assistant", content="", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "search_web", "arguments": "{}"},
        }]),
        Message(role="tool", tool_call_id="call_1", tool_name="search_web", content=long_page),
        Message(role="assistant", content="Let me look at one more source.", tool_calls=[{
            "id": "call_2", "type": "function",
            "function": {"name": "search_papers", "arguments": "{}"},
        }]),
        Message(role="tool", tool_call_id="call_2", tool_name="search_papers",
                content="The newest round of results, which the model has not read yet."),
    ]

    print(f"\n1. Before: search_web result is {len(messages[3].content):,} chars")

    # Not enough to reach the target: nothing changes, the runner cuts a new
    # window instead — one cache break either way, never two.
    missed = prune_tool_results(messages, context_tokens=50_000, target_tokens=1_000)
    print(f"2. Prune cannot reach the target: pruned {missed.pruned} (window will be cut)")

    reached = prune_tool_results(messages, context_tokens=50_000, target_tokens=48_000)
    print(f"3. Prune reaches the target:      pruned {reached.pruned}, freed ~{reached.freed_tokens:,} tokens")
    print(f"   search_web is now {len(messages[3].content):,} chars (head + tail kept)")
    print(f"   search_papers (unseen) untouched: {messages[5].content[:40]}...")


def demo_agent_with_compression():
    """Both layers are on by default; Layer 2 no longer calls an LLM."""
    print("\n" + "=" * 60)
    print("Agent with Compression Demo")
    print("=" * 60)

    agent1 = Agent(model=OpenAIChat(id="gpt-4o"), name="DefaultAgent")
    print("\n1. Default agent — Layer 2 is wired automatically:")
    print(f"   Manager: {type(agent1.tool_config.compression_manager).__name__}")
    print(f"   enable_evict={agent1.tool_config.enable_evict} "
          f"enable_auto_compact={agent1.tool_config.enable_auto_compact}")

    custom_compression = CompressionManager(compact_token_limit=5000)
    Agent(
        model=OpenAIChat(id="gpt-4o"),
        tool_config=ToolConfig(compression_manager=custom_compression),
        name="CustomCompressedAgent",
    )
    print("\n2. Agent with custom CompressionManager:")
    print(f"   compact_token_limit: {custom_compression.compact_token_limit}")


if __name__ == "__main__":
    print("Agentica Token Counting & Compression Demo")
    print("=" * 60)

    demo_token_counting()
    demo_layer1_prune()

    if os.getenv("OPENAI_API_KEY"):
        demo_agent_with_compression()
    else:
        print("\n[INFO] Set OPENAI_API_KEY to run the agent demo")
