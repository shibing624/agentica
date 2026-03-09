# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web Search Agent Demo

Uses Agent with BuiltinWebSearchTool and BuiltinFetchUrlTool for deep research tasks.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFetchUrlTool


def main():
    """Demo: Agent with web search tools for deep research tasks."""
    print("=" * 60)
    print("Web Search Agent Demo")
    print("=" * 60)

    agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Researcher",
        description="A deep research agent that finds precise answers.",
        tools=[BuiltinWebSearchTool(), BuiltinFetchUrlTool()],
        debug=True,
    )

    response = agent.run_sync(
        "Who won the 2024 Nobel Prize in Physics and what was their contribution?"
    )
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    main()
