# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WebSearchAgent - Enhanced search agent with structured evidence collection and verification.

Extends DeepAgent with:
1. SearchOrchestrator: Programmatic search flow control
2. QueryDecomposer: Query decomposition and multi-angle rewriting
3. EvidenceStore: Structured evidence collection and tracking
4. AnswerVerifier: Cross-validation and confidence assessment

Key Features:
- Multi-round search with automatic termination when evidence is sufficient
- Query decomposition for multi-hop questions
- Structured evidence extraction from search and fetch results
- Answer verification with cross-validation
- Reverse search verification (optional)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agentica.deep_agent import DeepAgent
from agentica.model.message import Message
from agentica.search.orchestrator import SearchOrchestrator
from agentica.search.evidence_store import SearchResult, EvidenceStore
from agentica.search.query_decomposer import QueryDecomposer
from agentica.search.answer_verifier import AnswerVerifier
from agentica.utils.log import logger


WEB_SEARCH_AGENT_SYSTEM_PROMPT = """You are a deep research agent specialized in finding precise answers through systematic web search.

## Search Strategy

1. **Decompose**: Break complex questions into specific, searchable sub-queries
2. **Search**: Execute searches with diverse query formulations
3. **Extract**: Identify and extract relevant facts from search results
4. **Verify**: Cross-validate findings from multiple sources
5. **Synthesize**: Combine evidence into a precise answer

## Key Principles

- Use multiple query formulations to maximize recall
- Always verify facts from at least 2 independent sources
- Be precise: prefer exact names, numbers, dates over vague descriptions
- When evidence conflicts, search for more specific queries to resolve
- Stop searching when evidence is sufficient and consistent

## Answer Format

- For factual questions: provide the exact answer (name, number, date, etc.)
- For complex questions: provide a concise, well-structured answer
- Always cite the key sources that support your answer
"""


@dataclass(init=False)
class WebSearchAgent(DeepAgent):
    """WebSearchAgent - Enhanced Agent for deep web search tasks.

    Extends DeepAgent with structured search orchestration, query decomposition,
    evidence tracking, and answer verification capabilities.

    Example:
        ```python
        from agentica import OpenAIChat
        from agentica.web_search_agent import WebSearchAgent

        agent = WebSearchAgent(
            model=OpenAIChat(id="gpt-4o"),
            max_search_rounds=10,
            confidence_threshold=0.7,
        )

        # Use deep_search for structured multi-round search
        result = await agent.deep_search("Who won the 2024 Turing Award?")

        # Or use standard run() which includes search tools
        response = agent.run("Research the latest AI agent frameworks")
        ```
    """

    # Search orchestration config
    max_search_rounds: int = 15
    max_queries_per_round: int = 5
    min_evidence_count: int = 2
    confidence_threshold: float = 0.8
    enable_query_decomposition: bool = True
    enable_answer_verification: bool = True
    enable_evidence_tracking: bool = True
    enable_reverse_verification: bool = False
    max_verification_retries: int = 2

    # Internal components
    _orchestrator: SearchOrchestrator = field(default=None, init=False, repr=False)

    def __init__(
        self,
        *,
        # WebSearchAgent specific parameters
        max_search_rounds: int = 15,
        max_queries_per_round: int = 5,
        min_evidence_count: int = 2,
        confidence_threshold: float = 0.8,
        enable_query_decomposition: bool = True,
        enable_answer_verification: bool = True,
        enable_evidence_tracking: bool = True,
        enable_reverse_verification: bool = False,
        max_verification_retries: int = 2,
        # DeepAgent / Agent parameters via kwargs
        **kwargs,
    ):
        """Initialize WebSearchAgent.

        Args:
            max_search_rounds: Maximum number of search rounds
            max_queries_per_round: Maximum queries per round
            min_evidence_count: Minimum evidence items required
            confidence_threshold: Confidence threshold for answer acceptance
            enable_query_decomposition: Enable query decomposition
            enable_answer_verification: Enable answer verification
            enable_evidence_tracking: Enable evidence tracking in hooks
            enable_reverse_verification: Enable reverse search verification
            max_verification_retries: Max retries for answer verification
            **kwargs: All DeepAgent/Agent parameters
        """
        # Store config before super().__init__
        self.max_search_rounds = max_search_rounds
        self.max_queries_per_round = max_queries_per_round
        self.min_evidence_count = min_evidence_count
        self.confidence_threshold = confidence_threshold
        self.enable_query_decomposition = enable_query_decomposition
        self.enable_answer_verification = enable_answer_verification
        self.enable_evidence_tracking = enable_evidence_tracking
        self.enable_reverse_verification = enable_reverse_verification
        self.max_verification_retries = max_verification_retries

        # Force enable web search and fetch URL
        kwargs['include_web_search'] = True
        kwargs['include_fetch_url'] = True

        # Inject system prompt for search strategy
        existing_instructions = kwargs.get('instructions', None)
        if existing_instructions:
            if isinstance(existing_instructions, list):
                existing_instructions = existing_instructions + [WEB_SEARCH_AGENT_SYSTEM_PROMPT]
            else:
                existing_instructions = [str(existing_instructions), WEB_SEARCH_AGENT_SYSTEM_PROMPT]
        else:
            existing_instructions = [WEB_SEARCH_AGENT_SYSTEM_PROMPT]
        kwargs['instructions'] = existing_instructions

        super().__init__(**kwargs)

        # Initialize orchestrator after model is available
        self._orchestrator = SearchOrchestrator(
            model=self.model,
            max_rounds=max_search_rounds,
            max_queries_per_round=max_queries_per_round,
            min_evidence_count=min_evidence_count,
            confidence_threshold=confidence_threshold,
        )

        # Enhance hooks for evidence tracking
        if enable_evidence_tracking:
            self._enhance_hooks_for_evidence()

    def _enhance_hooks_for_evidence(self) -> None:
        """Enhance DeepAgent's post_tool_hook to automatically extract evidence.

        Wraps the existing post_tool_hook to intercept web_search and fetch_url
        results and feed them into the EvidenceStore.
        """
        if self.model is None:
            return

        original_post_hook = getattr(self.model, '_post_tool_hook', None)
        agent = self  # capture for closure
        # Track the current question for evidence extraction
        agent._current_question = ""

        def enhanced_post_hook(function_call_results: list) -> None:
            # Execute original hook first
            if original_post_hook:
                original_post_hook(function_call_results)

            # Extract evidence from search/fetch tool results
            if not agent._current_question:
                return

            for msg in function_call_results:
                if not hasattr(msg, 'role') or msg.role != "tool":
                    continue
                tool_name = getattr(msg, 'tool_name', None)
                if tool_name not in ("web_search", "fetch_url"):
                    continue

                content = getattr(msg, 'content', None)
                if not content or not isinstance(content, str):
                    continue

                # Parse search results and create SearchResult objects
                search_results = agent._parse_tool_output_to_search_results(
                    tool_name, content
                )
                # Fire-and-forget evidence extraction
                for sr in search_results[:5]:  # Limit to top 5 per tool call
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(
                                agent._orchestrator.evidence_store.extract_and_store(
                                    sr, agent._current_question
                                )
                            )
                    except RuntimeError:
                        pass

        self.model._post_tool_hook = enhanced_post_hook

    @staticmethod
    def _parse_tool_output_to_search_results(tool_name: str, content: str) -> List[SearchResult]:
        """Parse tool output into SearchResult objects.

        Args:
            tool_name: Name of the tool (web_search or fetch_url)
            content: Raw tool output content

        Returns:
            List of SearchResult objects
        """
        results = []

        if tool_name == "web_search":
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    # Handle both single and batch query results
                    items = data.get("results", [])
                    if not items and isinstance(data, dict):
                        # Try to find results in nested structure
                        for key, val in data.items():
                            if isinstance(val, list):
                                items = val
                                break
                    for item in items:
                        if isinstance(item, dict):
                            results.append(SearchResult(
                                url=item.get("url", item.get("link", "")),
                                title=item.get("title", ""),
                                content=item.get("content", item.get("abstract", item.get("snippet", ""))),
                                query=item.get("query", ""),
                            ))
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            results.append(SearchResult(
                                url=item.get("url", item.get("link", "")),
                                title=item.get("title", ""),
                                content=item.get("content", item.get("abstract", item.get("snippet", ""))),
                                query=item.get("query", ""),
                            ))
            except (json.JSONDecodeError, TypeError):
                pass

        elif tool_name == "fetch_url":
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    results.append(SearchResult(
                        url=data.get("url", ""),
                        title=data.get("title", ""),
                        content=data.get("content", ""),
                    ))
            except (json.JSONDecodeError, TypeError):
                # Raw text content
                results.append(SearchResult(content=content[:4000]))

        return results

    async def deep_search(self, question: str) -> str:
        """Execute a structured multi-round deep search.

        This is the primary method for complex search tasks.
        Orchestrates: decompose -> search -> extract -> verify -> synthesize.

        Args:
            question: The question to answer

        Returns:
            The final synthesized answer
        """
        start_time = time.time()
        self._current_question = question
        self._orchestrator.reset()

        logger.info(f"Starting deep search for: {question[:100]}...")

        # Step 1: Decompose question into sub-queries
        if self.enable_query_decomposition:
            sub_queries = await self._orchestrator.decompose_query(question)
            logger.debug(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
        else:
            self._orchestrator._query_queue.append(question)
            self._orchestrator._original_question = question

        # Step 2: Multi-round search loop
        for round_idx in range(self.max_search_rounds):
            queries = self._orchestrator.get_next_queries()
            if not queries:
                logger.debug(f"No more queries at round {round_idx + 1}")
                break

            logger.debug(f"Round {round_idx + 1}: searching {len(queries)} queries")

            # Parallel search execution
            search_results = await self._parallel_search(queries)

            # Process results and extract evidence
            new_evidence = await self._orchestrator.process_results(search_results, question)
            logger.debug(f"Round {round_idx + 1}: {new_evidence} new evidence items")

            # Check sufficiency
            if self._orchestrator.is_sufficient():
                logger.debug(f"Evidence sufficient at round {round_idx + 1}")
                break

            # Generate follow-up queries for next round
            followups = await self._orchestrator.generate_followup_queries(question)
            if not followups:
                logger.debug("No follow-up queries generated, ending search")
                break

        # Step 3: Synthesize answer
        answer = await self._orchestrator.synthesize_answer(question)

        # Step 4: Verify answer
        if self.enable_answer_verification and answer:
            for retry in range(self.max_verification_retries):
                verification = await self._orchestrator.verify_answer(question, answer)
                logger.debug(
                    f"Verification attempt {retry + 1}: confident={verification.is_confident}, "
                    f"score={verification.confidence_score:.2f}"
                )

                if verification.is_confident:
                    break

                # If not confident, search with suggested queries
                if verification.suggested_queries:
                    self._orchestrator._query_queue.extend(verification.suggested_queries)
                    queries = self._orchestrator.get_next_queries()
                    if queries:
                        results = await self._parallel_search(queries)
                        await self._orchestrator.process_results(results, question)
                        answer = await self._orchestrator.synthesize_answer(question)

            # Optional: reverse verification
            if self.enable_reverse_verification and answer:
                reverse_result = await self._orchestrator.answer_verifier.reverse_verify(
                    question, answer, self._search_fn
                )
                if not reverse_result.is_confident:
                    logger.warning(f"Reverse verification failed: {reverse_result.reasoning}")

        elapsed = time.time() - start_time
        status = self._orchestrator.get_status()
        logger.info(
            f"Deep search completed in {elapsed:.1f}s. "
            f"Rounds: {status['current_round']}, Evidence: {status['evidence_count']}, "
            f"Confidence: {status['confidence']}"
        )

        return answer

    async def _parallel_search(self, queries: List[str]) -> List[SearchResult]:
        """Execute multiple search queries in parallel.

        Uses the built-in web_search tool to perform searches.

        Args:
            queries: List of search queries

        Returns:
            List of SearchResult objects from all queries
        """
        all_results = []

        # Use built-in web search tool
        from agentica.tools.buildin_tools import BuiltinWebSearchTool
        search_tool = None
        for tool in self.tools or []:
            if isinstance(tool, BuiltinWebSearchTool):
                search_tool = tool
                break

        if search_tool is None:
            logger.warning("No web search tool found")
            return all_results

        # Parallel search all queries
        async def search_one(query: str) -> List[SearchResult]:
            try:
                result_str = await search_tool.web_search(query, max_results=5)
                results = self._parse_tool_output_to_search_results("web_search", result_str)
                # Set query on results
                for r in results:
                    if not r.query:
                        r.query = query
                return results
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")
                return []

        tasks = [search_one(q) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results_list:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Search task failed: {result}")

        return all_results

    async def _search_fn(self, query: str) -> str:
        """Search function wrapper for reverse verification.

        Args:
            query: Search query

        Returns:
            Raw search result string
        """
        from agentica.tools.buildin_tools import BuiltinWebSearchTool
        for tool in self.tools or []:
            if isinstance(tool, BuiltinWebSearchTool):
                return await tool.web_search(query, max_results=3)
        return ""

    def get_evidence_summary(self) -> str:
        """Get a summary of all collected evidence.

        Returns:
            Formatted evidence summary
        """
        return self._orchestrator.evidence_store.get_summary()

    def get_search_status(self) -> dict:
        """Get current search orchestrator status.

        Returns:
            Dict with search progress information
        """
        return self._orchestrator.get_status()

    def reset_search_state(self) -> None:
        """Reset all search state for a new search session."""
        self._orchestrator.reset()
        self._current_question = ""

    def __repr__(self) -> str:
        """Return string representation of WebSearchAgent."""
        builtin_tools = self.get_builtin_tool_names()
        return (
            f"WebSearchAgent(name={self.name}, "
            f"max_rounds={self.max_search_rounds}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"builtin_tools={len(builtin_tools)})"
        )


if __name__ == '__main__':
    agent = WebSearchAgent(
        name="TestWebSearchAgent",
        description="A test web search agent",
        max_search_rounds=5,
        confidence_threshold=0.7,
        debug=True,
    )

    print(f"Created: {agent}")
    print(f"Builtin tools: {agent.get_builtin_tool_names()}")
    print(f"Search status: {agent.get_search_status()}")
