# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WebSearchAgent Demo

WebSearchAgent is a simple Agent subclass that auto-configures web_search
and fetch_url tools with an optimized search strategy prompt (ReAct pattern).
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agentica import OpenAIChat
from agentica.web_search_agent import WebSearchAgent


def main():
    """Demo: WebSearchAgent for deep research tasks."""
    print("=" * 60)
    print("WebSearchAgent Demo")
    print("=" * 60)

    agent = WebSearchAgent(
        model=OpenAIChat(id="gpt-4o"),
        name="Researcher",
        description="A deep research agent that finds precise answers.",
        debug=True,
    )

    response = agent.run_sync(
        "Who won the 2024 Nobel Prize in Physics and what was their contribution?"
    )
    print(f"\nResponse:\n{response.content}")


if __name__ == "__main__":
    main()
