# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Web Search Application using WebSearchAgent

Demonstrates WebSearchAgent capabilities:
1. Query decomposition for multi-hop questions
2. Structured evidence collection and tracking
3. Multi-round search with automatic termination
4. Answer verification with cross-validation
5. Precise answer extraction

Usage:
    python main.py
"""
import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from agentica import DeepSeekChat
from agentica.web_search_agent import WebSearchAgent
from agentica.agent.config import ToolConfig


def create_web_search_agent():
    """Create a WebSearchAgent with search enhancement capabilities."""
    return WebSearchAgent(
        model=DeepSeekChat(),
        name="WebSearcher",
        description="An enhanced web search agent with evidence tracking and answer verification",
        # Search orchestration config
        max_search_rounds=10,
        max_queries_per_round=5,
        min_evidence_count=2,
        confidence_threshold=0.7,
        enable_query_decomposition=True,
        enable_answer_verification=True,
        enable_evidence_tracking=True,
        # DeepAgent config
        enable_step_reflection=True,
        reflection_frequency=3,
        enable_context_overflow_handling=True,
        tool_config=ToolConfig(compress_tool_results=True),
        enable_repetition_detection=True,
        max_same_tool_calls=3,
        debug=True,
    )


async def deep_search_example(agent, question: str) -> str:
    """Run a deep search using the structured search pipeline.

    Args:
        agent: The WebSearchAgent instance
        question: The question to answer

    Returns:
        The synthesized answer
    """
    print(f"\n{'=' * 60}")
    print(f"Deep Search: {question}")
    print(f"{'=' * 60}\n")

    answer = await agent.deep_search(question)

    # Print search status
    status = agent.get_search_status()
    print(f"\nSearch Status: {status}")
    print(f"\nEvidence Summary:\n{agent.get_evidence_summary()}")
    print(f"\nAnswer: {answer}")

    return answer


def agent_run_example(agent, question: str) -> str:
    """Run a standard agent query (uses LLM-driven tool calling).

    Args:
        agent: The WebSearchAgent instance
        question: The question to answer

    Returns:
        The response content
    """
    print(f"\n{'=' * 60}")
    print(f"Agent Run: {question}")
    print(f"{'=' * 60}\n")

    response = agent.run_sync(question)
    print(f"\nResponse: {response.content}")
    return response.content


async def main():
    """Main entry point with example search tasks."""
    print("=" * 60)
    print("Web Search Agent Application")
    print("=" * 60)

    agent = create_web_search_agent()
    print(f"Agent created: {agent}")
    print(f"Search status: {agent.get_search_status()}")
    print()

    # Example 1: Deep search (structured multi-round search)
    question = "RAG技术的原理和最佳实践"
    await deep_search_example(agent, question)

    # Reset for next search
    agent.reset_search_state()


if __name__ == "__main__":
    asyncio.run(main())
