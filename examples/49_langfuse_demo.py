# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Langfuse integration demo, demonstrates LLM observability with Langfuse

Langfuse provides:
- Automatic tracing of all LLM calls
- Session grouping for multi-turn conversations
- Tool call tracking within agent runs

Setup:
    1. pip install langfuse
    2. Set environment variables:
        LANGFUSE_SECRET_KEY="sk-lf-xxx"
        LANGFUSE_PUBLIC_KEY="pk-lf-xxx"
        LANGFUSE_HOST="https://cloud.langfuse.com"
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica import Agent
from agentica.model.openai import OpenAIChat
from agentica.tools.baidu_search_tool import BaiduSearchTool

if __name__ == '__main__':
    # Example 1: Basic agent with Langfuse tracing
    # All LLM calls within agent.run() are grouped in a single trace
    print("=" * 50)
    print("Example 1: Basic Agent with Langfuse Tracing")
    print("=" * 50)

    agent = Agent(
        name="Langfuse Demo Agent",
        user_id="user-123",  # Passed to Langfuse for user tracking
        session_id="session-abc",  # Groups multi-turn conversations
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_tags=["demo", "basic"],  # Optional tags for filtering
        ),
        system_prompt="You are a helpful assistant.",
        debug_mode=True,
    )

    response = agent.run("Tell me a short joke about programming.")
    print(f"Response: {response.content}")
    print()

    # Example 2: Multi-turn conversation with same session_id
    # All runs with the same session_id are grouped together in Langfuse
    print("=" * 50)
    print("Example 2: Multi-turn Conversation (Same Session)")
    print("=" * 50)

    # First turn
    response1 = agent.run("What is Python?")
    print(f"Turn 1: {response1.content}")

    # Second turn - same session, will be grouped with first turn
    response2 = agent.run("What are its main features?")
    print(f"Turn 2: {response2.content}")

    # Third turn
    response3 = agent.run("Give me a simple example.")
    print(f"Turn 3: {response3.content}")
    print()

    # Example 3: Agent with tools - multiple LLM calls in one run
    # When agent uses tools, all LLM calls are grouped in one trace
    print("=" * 50)
    print("Example 3: Agent with Tools (Multi-step Trace)")
    print("=" * 50)

    agent_with_tools = Agent(
        name="Research Agent",
        user_id="user-456",
        session_id="research-session-001",
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_tags=["research", "tools"],
        ),
        tools=[BaiduSearchTool()],
        system_prompt="You are a research assistant. Use search tools to find information.",
        show_tool_calls=True,
        debug_mode=True,
    )

    # This single run may invoke LLM multiple times (initial + after tool results)
    # All calls are grouped under one trace in Langfuse
    response = agent_with_tools.run("What are the latest developments in AI agents in 2025? 中文回答")
    print(f"Research Response: {response.content}")
    print()

    # Example 4: Different sessions for different users
    print("=" * 50)
    print("Example 4: Different Users/Sessions")
    print("=" * 50)

    # User A's session
    agent_user_a = Agent(
        name="Assistant",
        user_id="alice",
        session_id="alice-session-001",
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    response_a = agent_user_a.run("Hello, I'm Alice!")
    print(f"Alice's response: {response_a.content}")

    # User B's session - tracked separately in Langfuse
    agent_user_b = Agent(
        name="Assistant",
        user_id="bob",
        session_id="bob-session-001",
        model=OpenAIChat(id="gpt-4o-mini"),
    )
    response_b = agent_user_b.run("Hi, I'm Bob!")
    print(f"Bob's response: {response_b.content}")
    print()

    print("=" * 50)
    print("Check your Langfuse dashboard to see:")
    print("- All traces grouped by session_id")
    print("- Multi-step tool calls within single traces")
    print("- User tracking across sessions")
    print("=" * 50)
