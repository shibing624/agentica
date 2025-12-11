# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Langfuse integration demo, demonstrates LLM observability with Langfuse

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

if __name__ == '__main__':
    # Example 1: Basic agent with Langfuse tracing
    # Langfuse automatically traces all OpenAI calls via OpenTelemetry
    print("=" * 50)
    print("Example 1: Basic Agent with Langfuse Tracing")
    print("=" * 50)

    agent = Agent(
        name="Langfuse Demo Agent",
        user_id="user-123",  # Passed to Langfuse metadata
        session_id="session-abc",  # Passed to Langfuse metadata
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_tags=["demo", "test"],  # Optional tags for filtering in Langfuse
        ),
        system_prompt="You are a helpful assistant.",
        debug=True,
    )

    response = agent.run("Tell me a short joke about programming.")
    print(f"Response: {response.content}")
    print()

    # Example 2: Agent with custom trace name
    print("=" * 50)
    print("Example 2: Agent with Custom Trace Name")
    print("=" * 50)

    agent2 = Agent(
        name="Custom Trace Agent",
        user_id="user-456",
        model=OpenAIChat(
            id="gpt-4o-mini",
            langfuse_trace_name="custom-joke-generator",  # Custom name in Langfuse
            langfuse_tags=["custom", "joke"],
        ),
        system_prompt="You are a comedian.",
        debug=True,
    )

    response2 = agent2.run("Tell me a joke about AI.")
    print(f"Response: {response2.content}")
    print()
