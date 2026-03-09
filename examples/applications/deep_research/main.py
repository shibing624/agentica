# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Research Application using Agent with built-in tools

This application demonstrates comprehensive deep research capabilities:
1. Search the web for information with iterative refinement
2. Cross-validate findings from multiple sources
3. Generate comprehensive reports with citations

Usage:
    python main.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import Agent, DeepSeekChat
from agentica.tools.buildin_tools import get_builtin_tools
from agentica.agent.config import ToolConfig, PromptConfig


def create_deep_research_agent():
    """Create a deep research agent with full capabilities."""
    return Agent(
        model=DeepSeekChat(),
        name="DeepResearcher",
        description="A comprehensive deep research assistant",
        tools=get_builtin_tools(),
        tool_config=ToolConfig(compress_tool_results=True),
        prompt_config=PromptConfig(markdown=True, enable_agentic_prompt=True),
        debug=True,
    )


def research_topic(agent, topic: str) -> str:
    """Conduct deep research on a topic.
    
    Args:
        agent: The deep research agent
        topic: The topic to research
        
    Returns:
        The research report
    """
    print(f"\n{'=' * 60}")
    print(f"Deep Researching: {topic}")
    print(f"{'=' * 60}\n")

    response = agent.run_sync(topic)
    return response.content


def main():
    """Main entry point with example research topics."""
    print("=" * 60)
    print("Deep Research Application")
    print("=" * 60)

    agent = create_deep_research_agent()
    print(f"Agent created: {agent}")
    print()

    examples = [
        "RAG技术的原理和最佳实践",
    ]

    topic = examples[0]
    report = research_topic(agent, topic)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
