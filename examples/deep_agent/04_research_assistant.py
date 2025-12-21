# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: DeepAgent Research Assistant Demo

This example demonstrates DeepAgent as a research assistant:
- Web search for information gathering
- URL fetching for detailed content
- Synthesizing information from multiple sources
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import DeepAgent, OpenAIChat


def web_search_demo():
    """Demo: Web search for research."""
    print("=" * 60)
    print("Demo 1: Web Search Research")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="Researcher",
        description="A research assistant with web search capabilities",
        instructions=[
            "You are a research assistant.",
            "Search the web for accurate, up-to-date information.",
            "Always cite your sources.",
        ],
        include_web_search=True,
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Research the latest developments in large language models in this last year. "
        "Provide a summary with key points and sources."
    )
    print(f"\nResponse:\n{response.content}")


def url_fetch_demo():
    """Demo: Fetch and analyze web content."""
    print("\n" + "=" * 60)
    print("Demo 2: URL Content Analysis")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="ContentAnalyzer",
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


def comprehensive_research_demo():
    """Demo: Comprehensive research combining multiple tools."""
    print("\n" + "=" * 60)
    print("Demo 3: Comprehensive Research")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="ComprehensiveResearcher",
        description="A comprehensive research assistant",
        instructions=[
            "You are a senior research analyst.",
            "Conduct thorough research using multiple sources.",
            "Synthesize information into clear, actionable insights.",
            "Always provide references for your findings.",
        ],
        include_web_search=True,
        include_fetch_url=True,
        include_execute=True,
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Research the current state of AI agent frameworks. "
        "Compare at least 3 popular frameworks, their features, and use cases. "
        "Create a comparison table and provide recommendations."
    )
    print(f"\nResponse:\n{response.content}")


def topic_deep_dive_demo():
    """Demo: Deep dive into a specific topic."""
    print("\n" + "=" * 60)
    print("Demo 4: Topic Deep Dive")
    print("=" * 60)

    agent = DeepAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="DeepDiveResearcher",
        description="A research assistant for deep topic analysis",
        instructions=[
            "You are an expert researcher.",
            "Provide comprehensive, well-structured analysis.",
            "Include historical context, current state, and future trends.",
        ],
        include_web_search=True,
        show_tool_calls=True,
        # debug_mode=True,
    )

    response = agent.run(
        "Provide a deep dive analysis on 'Retrieval Augmented Generation (RAG)'. "
        "Cover: 1) What it is, 2) How it works, 3) Key applications, "
        "4) Challenges and limitations, 5) Future directions."
    )
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    web_search_demo()
    # url_fetch_demo()  # Uncomment to run
    # comprehensive_research_demo()  # Uncomment to run
    # topic_deep_dive_demo()  # Uncomment to run
