# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WebSearchAgent - Agent with web search + fetch_url tools, ReAct pattern.

Simple wrapper around Agent that auto-configures web search and URL fetching tools
with an optimized system prompt for deep research tasks.
"""

from typing import Any, List

from agentica.agent import Agent
from agentica.tools.buildin_tools import BuiltinWebSearchTool, BuiltinFetchUrlTool

WEB_SEARCH_SYSTEM_PROMPT = """\
You are a deep research agent specialized in finding precise, factual answers through systematic web search.

## Search Strategy
- Break complex questions into specific, searchable sub-queries
- Start with the most distinctive or unique criteria to narrow results quickly
- Use multiple query formulations (different keywords, synonyms, related terms) to maximize recall
- If initial searches fail, try alternative angles: different time periods, related entities, or broader/narrower scope

## Document Reading
- After finding promising search results, ALWAYS use fetch_url to read the full document
- Search snippets are often incomplete or misleading - verify by reading the source
- Extract specific facts, names, dates, and numbers from full documents
- Pay attention to document dates and credibility

## Persistence
- Do NOT give up after 1-2 failed searches - try at least 3-5 different query formulations
- If direct searches fail, try searching for related entities or events that might mention the answer
- Use information from partial results to refine subsequent searches
- Think step by step about what information you still need

## Cross-verification
- Verify candidate answers from at least 2 independent sources when possible
- If sources conflict, search for more specific queries to resolve the disagreement
- Consider whether the answer makes sense given the context and constraints

## Answer Format
- Provide a precise, concise answer (exact name, number, date, etc.)
- Do NOT include explanations, hedging, or uncertainty markers in the final answer
- If you truly cannot find the answer after exhaustive search, say so clearly
"""


class WebSearchAgent(Agent):
    """Agent with web search + fetch_url tools, using ReAct pattern.

    Automatically adds BuiltinWebSearchTool and BuiltinFetchUrlTool,
    and injects a search-optimized system prompt.

    Example:
        agent = WebSearchAgent(model=OpenAIChat(id="gpt-4o"))
        response = agent.run_sync("Who won the 2024 Nobel Prize in Physics?")
        print(response.content)
    """

    def __init__(
        self,
        *,
        include_fetch_url: bool = True,
        max_content_length: int = 16000,
        **kwargs: Any,
    ):
        """Initialize WebSearchAgent.

        Args:
            include_fetch_url: Whether to include the fetch_url tool (default True).
            max_content_length: Max content length for fetch_url results.
            **kwargs: All other arguments passed to Agent.__init__().
        """
        # Build web tools
        web_tools: List[Any] = [BuiltinWebSearchTool()]
        if include_fetch_url:
            web_tools.append(BuiltinFetchUrlTool(max_content_length=max_content_length))

        # Merge with user-provided tools
        user_tools = kwargs.pop("tools", None) or []
        kwargs["tools"] = web_tools + list(user_tools)

        # Inject search strategy into instructions
        user_instructions = kwargs.pop("instructions", None)
        instructions = [WEB_SEARCH_SYSTEM_PROMPT]
        if user_instructions:
            if isinstance(user_instructions, str):
                instructions.append(user_instructions)
            elif isinstance(user_instructions, list):
                instructions.extend(user_instructions)
        kwargs["instructions"] = instructions

        super().__init__(**kwargs)
