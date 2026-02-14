# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Deep Research Application using DeepAgent

This application demonstrates comprehensive deep research capabilities:
1. Search the web for information with iterative refinement
2. Cross-validate findings from multiple sources
3. Generate comprehensive reports with citations
4. Handle context overflow intelligently
5. Step reflection for quality improvement
6. Repetition detection to avoid redundant searches

Usage:
    python main.py
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import DeepAgent, DeepSeek


def create_deep_research_agent():
    """Create a deep research agent with full capabilities."""
    return DeepAgent(
        model=DeepSeek(),
        name="DeepResearcher",
        description="A comprehensive deep research assistant with reflection and context management",
        # Enable deep research mode
        enable_deep_research=True,
        enable_step_reflection=True,
        reflection_frequency=3,
        # Context management
        context_soft_limit=80000,
        context_hard_limit=120000,
        enable_context_overflow_handling=True,
        compress_tool_results=True,
        # Repetition detection
        enable_repetition_detection=True,
        max_same_tool_calls=3,
        # Tools
        include_web_search=True,
        include_fetch_url=True,
        include_execute=True,
        include_todos=True,
        include_file_tools=True,
        # Debug
        markdown=True,
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

    # Create the deep research agent
    agent = create_deep_research_agent()
    print(f"Agent created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")
    print()

    # Example research topics - uncomment one to run
    examples = [
        # "AI Agent框架的发展趋势和主流框架对比分析",
        # "2024年大语言模型的最新进展",
        "RAG技术的原理和最佳实践",
        # "Multi-Agent系统的设计模式",
    ]

    # Run the first example
    topic = examples[0]
    report = research_topic(agent, topic)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
