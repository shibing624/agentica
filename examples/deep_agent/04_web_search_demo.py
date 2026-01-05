# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent Web Search Demo

This example demonstrates DeepAgent's web capabilities:
- Web search for information gathering
- URL fetching for detailed content

For comprehensive deep research with reflection and context management,
see: examples/applications/deep_research/main.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


def web_search_demo():
    """Demo: Web search for information."""
    print("=" * 60)
    print("Demo 1: Web Search")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="WebSearcher",
        description="An assistant with web search capabilities",
        instructions=[
            "You are a helpful assistant.",
            "Search the web for accurate, up-to-date information.",
            "Always cite your sources.",
        ],
        include_web_search=True,
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Search for the latest news about large language models. "
        "Provide a brief summary with sources."
    )
    print(f"\nResponse:\n{response.content}")


def url_fetch_demo():
    """Demo: Fetch and analyze web content."""
    print("\n" + "=" * 60)
    print("Demo 2: URL Content Fetching")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="ContentFetcher",
        description="An assistant that can fetch and analyze web content",
        include_fetch_url=True,
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Fetch the content from https://en.wikipedia.org/wiki/Artificial_intelligence "
        "and provide a brief summary of the main topics covered."
    )
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    web_search_demo()
    # url_fetch_demo()  # Uncomment to run
