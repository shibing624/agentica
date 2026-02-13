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
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent
from agentica.model.openai import OpenAIChat
from agentica.tools.baidu_search_tool import BaiduSearchTool
from agentica.agent.config import PromptConfig


async def main():
    # Example 1: Basic agent with Langfuse tracing
    print("=" * 50)
    print("Example 1: Basic Agent with Langfuse Tracing")
    print("=" * 50)

    agent = Agent(
        name="Langfuse Demo Agent",
        # user_id and session_id removed in V2 API
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_tags=["demo", "basic"],
        ),
        prompt_config=PromptConfig(system_prompt="You are a helpful assistant."),
        debug_mode=True,
    )

    response = await agent.run("讲个笑话")
    print(f"Response: {response.content}")
    print()

    # Example 2: Multi-turn conversation
    print("=" * 50)
    print("Example 2: Multi-turn Conversation (Same Session)")
    print("=" * 50)

    response1 = await agent.run("What is Python?")
    print(f"Turn 1: {response1.content}")

    response2 = await agent.run("What are its main features?")
    print(f"Turn 2: {response2.content}")
    print()

    # Example 3: Agent with tools
    print("=" * 50)
    print("Example 3: Agent with Tools (Multi-step Trace)")
    print("=" * 50)

    agent_with_tools = Agent(
        name="Research Agent",
        # user_id and session_id removed in V2 API
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_tags=["research", "tools"],
        ),
        tools=[BaiduSearchTool()],
        prompt_config=PromptConfig(system_prompt="You are a research assistant. Use search tools to find information."),
        debug_mode=True,
    )

    response = await agent_with_tools.run("What are the latest developments in AI agents in 2025? 中文回答")
    print(f"Research Response: {response.content}")
    print()

    print("=" * 50)
    print("Check your Langfuse dashboard to see:")
    print("- All traces grouped by session_id")
    print("- Multi-step tool calls within single traces")
    print("- User tracking across sessions")
    print("=" * 50)


if __name__ == '__main__':
    asyncio.run(main())
