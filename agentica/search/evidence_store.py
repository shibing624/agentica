# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: EvidenceStore - Structured evidence collection, deduplication, and tracking.

Responsibilities:
1. Extract structured evidence from search results using LLM
2. Deduplicate and detect conflicting evidence
3. Track entities and their relationships
4. Assess information sufficiency
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set

from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt template
EVIDENCE_EXTRACT_PROMPT = _load_prompt("evidence_extract.md")


@dataclass
class Evidence:
    """A single piece of evidence extracted from search results."""

    content: str  # Evidence content (the relevant fact)
    source_url: str  # Source URL
    source_title: str  # Source page title
    query: str  # The search query that produced this evidence
    entities: List[str] = field(default_factory=list)  # Extracted entities
    relevance_score: float = 0.0  # Relevance to the original question (0-1)
    timestamp: str = ""  # Information timestamp if available


@dataclass
class SearchResult:
    """A single search result item."""

    url: str = ""
    title: str = ""
    content: str = ""
    query: str = ""


@dataclass
class EvidenceStore:
    """Evidence storage and management.

    Manages structured evidence extracted from search results.
    Provides deduplication, entity tracking, and sufficiency assessment.
    """

    model: Any = None  # LLM model instance

    _evidence_list: List[Evidence] = field(default_factory=list)
    _entities: Dict[str, List[str]] = field(default_factory=dict)  # entity -> [related facts]
    _seen_contents: Set[str] = field(default_factory=set)  # content fingerprints for dedup

    async def extract_and_store(
        self, search_result: SearchResult, question: str
    ) -> Optional[Evidence]:
        """Extract evidence from a search result and store it.

        Uses LLM to extract relevant facts, key entities, and assess relevance.

        Args:
            search_result: The search result to process
            question: The original question for relevance assessment

        Returns:
            Extracted Evidence if relevant, None otherwise
        """
        if not search_result.content or not search_result.content.strip():
            return None

        # Truncate content to avoid excessive token usage
        content = search_result.content[:4000]

        prompt = EVIDENCE_EXTRACT_PROMPT.format(
            question=question,
            content=content,
            source=search_result.url,
            title=search_result.title,
        ) if EVIDENCE_EXTRACT_PROMPT else (
            f"Question: {question}\n\n"
            f"Source: {search_result.title} ({search_result.url})\n"
            f"Content:\n{content}\n\n"
            f"Extract the key facts from this content that are relevant to answering the question.\n"
            f"Output JSON with fields:\n"
            f'  "relevant_facts": "concise summary of relevant facts (1-3 sentences)",\n'
            f'  "entities": ["list", "of", "key", "entities"],\n'
            f'  "relevance_score": 0.0-1.0,\n'
            f'  "timestamp": "date/time if mentioned, empty string otherwise"\n'
            f"If the content is completely irrelevant, set relevance_score to 0.\n"
            f"Output only valid JSON."
        )

        if self.model is None:
            # Fallback: store raw content without LLM extraction
            evidence = Evidence(
                content=content[:500],
                source_url=search_result.url,
                source_title=search_result.title,
                query=search_result.query,
                relevance_score=0.5,
            )
            return self._store_if_new(evidence)

        try:
            response = await self.model.response([
                Message(role="system", content="You are an evidence extraction expert. Output only valid JSON."),
                Message(role="user", content=prompt),
            ])
            evidence = self._parse_evidence(response.content, search_result)
            if evidence and evidence.relevance_score > 0.2:
                return self._store_if_new(evidence)
            return None
        except Exception as e:
            logger.error(f"Evidence extraction failed: {e}")
            return None

    def add_evidence_direct(
        self,
        content: str,
        source_url: str = "",
        source_title: str = "",
        query: str = "",
        relevance_score: float = 0.5,
    ) -> Optional[Evidence]:
        """Add evidence directly without LLM extraction.

        Useful for adding evidence from tool call results processed by the agent.

        Args:
            content: Evidence content
            source_url: Source URL
            source_title: Source title
            query: The query that produced this evidence
            relevance_score: Relevance score

        Returns:
            The stored Evidence if new, None if duplicate
        """
        evidence = Evidence(
            content=content[:2000],
            source_url=source_url,
            source_title=source_title,
            query=query,
            relevance_score=relevance_score,
        )
        return self._store_if_new(evidence)

    def _store_if_new(self, evidence: Evidence) -> Optional[Evidence]:
        """Store evidence if it's not a duplicate.

        Uses content fingerprint for deduplication.

        Args:
            evidence: Evidence to store

        Returns:
            The evidence if stored, None if duplicate
        """
        # Simple dedup: check content similarity
        fingerprint = evidence.content[:200].lower().strip()
        if fingerprint in self._seen_contents:
            logger.debug(f"Duplicate evidence skipped from {evidence.source_url}")
            return None

        self._seen_contents.add(fingerprint)
        self._evidence_list.append(evidence)

        # Update entity tracking
        for entity in evidence.entities:
            self._entities.setdefault(entity, []).append(evidence.content[:200])

        logger.debug(
            f"Stored evidence #{len(self._evidence_list)} from {evidence.source_title}, "
            f"relevance={evidence.relevance_score:.2f}"
        )
        return evidence

    def get_evidence_count(self) -> int:
        """Get the number of stored evidence items."""
        return len(self._evidence_list)

    def get_confidence(self) -> float:
        """Assess overall confidence based on collected evidence.

        Factors:
        1. Average relevance score
        2. Source diversity (different URLs)
        3. Evidence count

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not self._evidence_list:
            return 0.0

        avg_relevance = sum(e.relevance_score for e in self._evidence_list) / len(self._evidence_list)

        # Source diversity factor: at least 2 different sources
        unique_sources = len(set(e.source_url for e in self._evidence_list if e.source_url))
        source_factor = min(unique_sources / 2.0, 1.0)

        # Evidence count factor: more evidence = higher confidence (capped at 5)
        count_factor = min(len(self._evidence_list) / 3.0, 1.0)

        confidence = avg_relevance * 0.5 + source_factor * 0.3 + count_factor * 0.2
        return min(confidence, 1.0)

    def get_summary(self) -> str:
        """Get a text summary of all collected evidence.

        Used for generating follow-up queries and final answer synthesis.

        Returns:
            Formatted evidence summary string
        """
        if not self._evidence_list:
            return "No evidence collected yet."

        lines = []
        for i, e in enumerate(self._evidence_list, 1):
            score_str = f"[relevance={e.relevance_score:.1f}]"
            source_str = f"[{e.source_title}]" if e.source_title else f"[{e.source_url}]"
            lines.append(f"{i}. {score_str} {source_str} {e.content[:300]}")

        # Add entity summary if entities exist
        if self._entities:
            lines.append("\nKey entities found:")
            for entity, facts in list(self._entities.items())[:10]:
                lines.append(f"  - {entity}: {len(facts)} mention(s)")

        return "\n".join(lines)

    def get_all_evidence(self) -> List[Evidence]:
        """Get all evidence sorted by relevance score (highest first)."""
        return sorted(self._evidence_list, key=lambda e: e.relevance_score, reverse=True)

    def get_evidence_for_answer(self, max_items: int = 10) -> str:
        """Get formatted evidence text for answer synthesis.

        Args:
            max_items: Maximum number of evidence items to include

        Returns:
            Formatted evidence text
        """
        evidence = self.get_all_evidence()[:max_items]
        if not evidence:
            return "No evidence available."

        lines = []
        for i, e in enumerate(evidence, 1):
            source = e.source_title or e.source_url or "unknown"
            lines.append(f"[Evidence {i}] Source: {source}\n{e.content}")
        return "\n\n".join(lines)

    def get_entities(self) -> Dict[str, List[str]]:
        """Get tracked entities and their associated facts."""
        return dict(self._entities)

    def clear(self) -> None:
        """Clear all stored evidence and reset state."""
        self._evidence_list.clear()
        self._entities.clear()
        self._seen_contents.clear()

    @staticmethod
    def _parse_evidence(content: Optional[str], search_result: SearchResult) -> Optional[Evidence]:
        """Parse LLM response to extract an Evidence object.

        Args:
            content: Raw LLM response
            search_result: The original search result

        Returns:
            Evidence object or None
        """
        if not content:
            return None

        text = content.strip()

        # Strip markdown code block
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                return None

            relevant_facts = data.get("relevant_facts", "")
            if not relevant_facts:
                return None

            return Evidence(
                content=str(relevant_facts),
                source_url=search_result.url,
                source_title=search_result.title,
                query=search_result.query,
                entities=data.get("entities", []),
                relevance_score=float(data.get("relevance_score", 0.5)),
                timestamp=str(data.get("timestamp", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse evidence JSON: {e}")
            # Fallback: use raw text as evidence
            return Evidence(
                content=text[:500],
                source_url=search_result.url,
                source_title=search_result.title,
                query=search_result.query,
                relevance_score=0.3,
            )
