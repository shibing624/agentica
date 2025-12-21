# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Token compression demo - Demonstrates token counting and compression

This example shows:
1. Token counting for messages, tools, and multi-modal content
2. Tool result compression to save context space
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, Message
from agentica.utils.tokens import (
    count_tokens,
    count_text_tokens,
    count_message_tokens,
    count_tool_tokens,
)
from agentica.compression import CompressionManager


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


def demo_compression_manager():
    """Demonstrate compression manager functionality."""
    print("\n" + "=" * 60)
    print("Compression Manager Demo")
    print("=" * 60)

    compression_manager = CompressionManager(
        model=OpenAIChat(id="gpt-4o-mini"),
        compress_tool_results=True,
        compress_tool_results_limit=2,
    )

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
    ]

    print(f"\n1. Before compression:")
    print(f"   Number of messages: {len(messages)}")
    for msg in messages:
        if msg.role == "tool":
            print(f"   Tool result ({msg.tool_name}): {len(str(msg.content))} chars")

    should_compress = compression_manager.should_compress(messages)
    print(f"\n2. Should compress: {should_compress}")

    if should_compress:
        compression_manager.compress(messages)
        
        print(f"\n3. After compression:")
        for msg in messages:
            if msg.role == "tool" and msg.compressed_content:
                print(f"   Tool result ({msg.tool_name}):")
                print(f"      Original: {len(str(msg.content))} chars")
                print(f"      Compressed: {len(msg.compressed_content)} chars")

        stats = compression_manager.get_stats()
        print(f"\n4. Compression stats:")
        print(f"   Tool results compressed: {stats.get('tool_results_compressed', 0)}")
        print(f"   Compression ratio: {stats.get('compression_ratio', 1.0):.2%}")


def demo_agent_with_compression():
    """Demonstrate Agent with compression enabled."""
    print("\n" + "=" * 60)
    print("Agent with Compression Demo")
    print("=" * 60)

    # Method 1: Simple - enable compression via flag
    agent1 = Agent(
        model=OpenAIChat(id="gpt-4o"),
        compress_tool_results=True,
        name="CompressedAgent",
    )
    print("\n1. Agent with compress_tool_results=True:")
    print(f"   Compression manager: {agent1.compression_manager is not None}")

    # Method 2: Custom compression manager
    custom_compression = CompressionManager(
        model=OpenAIChat(id="gpt-4o-mini"),
        compress_token_limit=5000,
        compress_tool_results_limit=3,
    )

    agent2 = Agent(
        model=OpenAIChat(id="gpt-4o"),
        compression_manager=custom_compression,
        name="CustomCompressedAgent",
    )
    print("\n2. Agent with custom CompressionManager:")
    print(f"   Token limit: {custom_compression.compress_token_limit}")
    print(f"   Tool results limit: {custom_compression.compress_tool_results_limit}")


if __name__ == "__main__":
    print("Agentica Token Counting & Compression Demo")
    print("=" * 60)

    demo_token_counting()

    if os.getenv("OPENAI_API_KEY"):
        demo_compression_manager()
        demo_agent_with_compression()
    else:
        print("\n[INFO] Set OPENAI_API_KEY to run compression demos")
