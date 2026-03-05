# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: QueryDecomposer - Decompose complex queries into sub-queries and generate rewrites.

Responsibilities:
1. Decompose complex multi-hop questions into independent sub-queries
2. Generate multi-angle rewrites for each sub-query to improve recall
3. Generate follow-up queries based on evidence gaps
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Any

from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompts from MD files
QUERY_DECOMPOSE_PROMPT = _load_prompt("query_decompose.md")
FOLLOWUP_QUERY_PROMPT = _load_prompt("search_reflection.md")


@dataclass
class QueryDecomposer:
    """Query decomposition and rewriting for improved search recall.

    Decomposes complex multi-hop questions into independent sub-queries,
    generates multi-angle rewrites, and creates follow-up queries based
    on evidence gaps.
    """

    model: Any = None  # LLM model instance

    async def decompose(self, question: str) -> List[str]:
        """Decompose a complex question into sub-queries.

        Strategy:
        1. Identify key entities and constraints in the question
        2. Generate independent search queries for each entity/constraint
        3. Generate multi-angle rewrites of the original question

        Args:
            question: The original complex question

        Returns:
            List of sub-queries for parallel search

        Example:
            Input: "Who won the 2024 Turing Award and where do they work?"
            Output: [
                "2024 Turing Award winner",
                "2024 ACM Turing Award recipient",
                "2024 Turing Award winner university affiliation",
            ]
        """
        if self.model is None:
            logger.warning("No model configured for QueryDecomposer, returning original question")
            return [question]

        prompt = QUERY_DECOMPOSE_PROMPT.format(question=question) if QUERY_DECOMPOSE_PROMPT else (
            f"You are a search query expert. Decompose the following question into 2-5 independent search queries.\n"
            f"Each query should target a different aspect or use different keywords.\n"
            f"Include both English and Chinese queries if the question involves multilingual content.\n\n"
            f"Question: {question}\n\n"
            f"Output a JSON array of search query strings. Only output the JSON array, nothing else."
        )

        try:
            response = await self.model.response([
                Message(role="system", content="You are a search query decomposition expert. Output only valid JSON."),
                Message(role="user", content=prompt),
            ])
            queries = self._parse_queries(response.content)
            if not queries:
                logger.warning("QueryDecomposer returned empty result, using original question")
                return [question]
            logger.debug(f"Decomposed question into {len(queries)} sub-queries")
            return queries
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [question]

    async def generate_followup(self, question: str, evidence_summary: str) -> List[str]:
        """Generate follow-up queries based on evidence gaps.

        Analyzes the evidence collected so far and identifies information gaps,
        then generates targeted queries to fill those gaps.

        Args:
            question: The original question
            evidence_summary: Summary of evidence collected so far

        Returns:
            List of follow-up queries, empty if evidence is sufficient
        """
        if self.model is None:
            return []

        prompt = FOLLOWUP_QUERY_PROMPT.format(
            question=question, evidence_summary=evidence_summary
        ) if FOLLOWUP_QUERY_PROMPT else (
            f"Original question: {question}\n\n"
            f"Evidence collected so far:\n{evidence_summary}\n\n"
            f"Analyze what key information is still missing to answer the question.\n"
            f"Generate 1-3 targeted search queries to fill the gaps.\n"
            f"If the evidence is sufficient to answer the question, return an empty JSON array [].\n"
            f"Output as JSON array of strings only."
        )

        try:
            response = await self.model.response([
                Message(role="system", content="You are a search query expert. Output only valid JSON."),
                Message(role="user", content=prompt),
            ])
            queries = self._parse_queries(response.content)
            logger.debug(f"Generated {len(queries)} follow-up queries")
            return queries
        except Exception as e:
            logger.error(f"Follow-up query generation failed: {e}")
            return []

    @staticmethod
    def _parse_queries(content: Optional[str]) -> List[str]:
        """Parse LLM response to extract query list.

        Handles JSON arrays, markdown code blocks, and line-separated formats.

        Args:
            content: Raw LLM response content

        Returns:
            List of query strings
        """
        if not content:
            return []

        text = content.strip()

        # Strip markdown code block wrappers
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(q).strip() for q in result if q and str(q).strip()]
        except json.JSONDecodeError:
            pass

        # Fallback: line-separated or numbered list
        queries = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering like "1.", "1)", "- "
            import re
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            line = line.strip('"\'')
            if line:
                queries.append(line)

        return queries
