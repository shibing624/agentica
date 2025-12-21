# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Langfuse integration demo - Demonstrates LLM observability

Langfuse provides:
- Automatic tracing of all LLM calls
- Session grouping for multi-turn conversations
- Tool call tracking within agent runs

Setup:
    1. pip install langfuse
    2. Set environment variables:
        LANGFUSE_SECRET_KEY="xxx"
        LANGFUSE_PUBLIC_KEY="xxx"
        LANGFUSE_BASE_URL="xxx"
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.model.openai import OpenAIChat
from agentica.tools.baidu_search_tool import BaiduSearchTool

if __name__ == '__main__':
    # Example 1: Basic agent with Langfuse tracing
    print("=" * 50)
    print("Example 1: Basic Agent with Langfuse Tracing")
    print("=" * 50)

    agent = Agent(
        name="Langfuse Demo Agent",
        user_id="user-123",
        session_id="session-abc",
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_tags=["demo", "basic"],
        ),
        system_prompt="You are a helpful assistant.",
        debug_mode=True,
    )

    response = agent.run("讲个笑话")
    print(f"Response: {response.content}")
    print()

    # Example 2: Multi-turn conversation
    print("=" * 50)
    print("Example 2: Multi-turn Conversation (Same Session)")
    print("=" * 50)

    response1 = agent.run("What is Python?")
    print(f"Turn 1: {response1.content}")

    response2 = agent.run("What are its main features?")
    print(f"Turn 2: {response2.content}")
    print()

    # Example 3: Agent with tools
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

    response = agent_with_tools.run("What are the latest developments in AI agents in 2025? 中文回答")
    print(f"Research Response: {response.content}")
    print()

    print("=" * 50)
    print("Check your Langfuse dashboard to see:")
    print("- All traces grouped by session_id")
    print("- Multi-step tool calls within single traces")
    print("- User tracking across sessions")
    print("=" * 50)
