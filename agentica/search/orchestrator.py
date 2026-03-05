# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: SearchOrchestrator - Orchestrate the complete search workflow.

Manages the full search lifecycle:
1. Query decomposition via QueryDecomposer
2. Search round management with query queue
3. Evidence collection via EvidenceStore
4. Information sufficiency assessment
5. Answer synthesis and verification via AnswerVerifier
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Set

from agentica.model.message import Message
from agentica.search.query_decomposer import QueryDecomposer
from agentica.search.evidence_store import EvidenceStore, SearchResult
from agentica.search.answer_verifier import AnswerVerifier, VerificationResult
from agentica.utils.log import logger


@dataclass
class SearchOrchestrator:
    """Search orchestrator - manages search state and flow decisions.

    Coordinates QueryDecomposer, EvidenceStore, and AnswerVerifier to
    execute a structured multi-round search workflow.
    """

    model: Any = None  # LLM model instance

    # Sub-components
    query_decomposer: QueryDecomposer = field(default=None)
    evidence_store: EvidenceStore = field(default=None)
    answer_verifier: AnswerVerifier = field(default=None)

    # Configuration
    max_rounds: int = 15
    max_queries_per_round: int = 5
    min_evidence_count: int = 2
    confidence_threshold: float = 0.8

    # Internal state
    _current_round: int = field(default=0, init=False, repr=False)
    _query_queue: List[str] = field(default_factory=list, init=False, repr=False)
    _searched_queries: Set[str] = field(default_factory=set, init=False, repr=False)
    _original_question: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        """Initialize sub-components if not provided."""
        if self.query_decomposer is None:
            self.query_decomposer = QueryDecomposer(model=self.model)
        if self.evidence_store is None:
            self.evidence_store = EvidenceStore(model=self.model)
        if self.answer_verifier is None:
            self.answer_verifier = AnswerVerifier(model=self.model)

    async def decompose_query(self, question: str) -> List[str]:
        """Decompose the original question into sub-queries and enqueue them.

        Args:
            question: The original question

        Returns:
            List of sub-queries
        """
        self._original_question = question
        sub_queries = await self.query_decomposer.decompose(question)
        self._query_queue.extend(sub_queries)
        logger.debug(f"Decomposed into {len(sub_queries)} sub-queries, queue size: {len(self._query_queue)}")
        return sub_queries

    def get_next_queries(self) -> List[str]:
        """Get the next batch of queries to search, with deduplication.

        Returns:
            List of unique queries for this round (up to max_queries_per_round)
        """
        batch = []
        while self._query_queue and len(batch) < self.max_queries_per_round:
            q = self._query_queue.pop(0)
            # Normalize for dedup
            q_normalized = q.strip().lower()
            if q_normalized not in self._searched_queries:
                batch.append(q)
                self._searched_queries.add(q_normalized)
        self._current_round += 1
        logger.debug(f"Round {self._current_round}: {len(batch)} queries to search")
        return batch

    async def process_results(self, results: List[SearchResult], question: str = "") -> int:
        """Process search results and extract evidence.

        Args:
            results: List of search results to process
            question: The question for relevance assessment (uses stored question if empty)

        Returns:
            Number of new evidence items stored
        """
        q = question or self._original_question
        new_count = 0
        for result in results:
            evidence = await self.evidence_store.extract_and_store(result, q)
            if evidence is not None:
                new_count += 1
        logger.debug(
            f"Processed {len(results)} results, extracted {new_count} new evidence items. "
            f"Total evidence: {self.evidence_store.get_evidence_count()}"
        )
        return new_count

    def is_sufficient(self) -> bool:
        """Check if collected evidence is sufficient to answer the question.

        Returns:
            True if evidence count and confidence meet thresholds
        """
        count_ok = self.evidence_store.get_evidence_count() >= self.min_evidence_count
        confidence_ok = self.evidence_store.get_confidence() >= self.confidence_threshold
        result = count_ok and confidence_ok
        if result:
            logger.debug(
                f"Evidence sufficient: count={self.evidence_store.get_evidence_count()}, "
                f"confidence={self.evidence_store.get_confidence():.2f}"
            )
        return result

    async def generate_followup_queries(self, question: str = "") -> List[str]:
        """Generate follow-up queries based on evidence gaps.

        Args:
            question: The original question (uses stored question if empty)

        Returns:
            List of follow-up queries
        """
        q = question or self._original_question
        evidence_summary = self.evidence_store.get_summary()
        new_queries = await self.query_decomposer.generate_followup(q, evidence_summary)
        self._query_queue.extend(new_queries)
        return new_queries

    async def synthesize_answer(self, question: str = "") -> str:
        """Synthesize a final answer from collected evidence.

        Uses LLM to combine evidence into a precise, concise answer.

        Args:
            question: The original question (uses stored question if empty)

        Returns:
            The synthesized answer string
        """
        q = question or self._original_question
        evidence_text = self.evidence_store.get_evidence_for_answer(max_items=15)

        if self.model is None:
            return f"Based on {self.evidence_store.get_evidence_count()} evidence items:\n{evidence_text}"

        prompt = (
            f"Question: {q}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Based on the evidence above, provide a precise and concise answer to the question.\n"
            f"Requirements:\n"
            f"1. Answer must be based ONLY on the evidence provided\n"
            f"2. Be precise and specific (prefer exact names, numbers, dates)\n"
            f"3. If the evidence is contradictory, note the contradiction\n"
            f"4. If the evidence is insufficient, say so clearly\n"
            f"5. Keep the answer concise - a short phrase or 1-2 sentences for factual questions"
        )

        try:
            response = await self.model.response([
                Message(role="system", content="You are a precise answer synthesis expert."),
                Message(role="user", content=prompt),
            ])
            answer = response.content.strip() if response.content else ""
            logger.debug(f"Synthesized answer: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Could not synthesize answer: {e}"

    async def verify_answer(self, question: str, answer: str) -> VerificationResult:
        """Verify a candidate answer against evidence.

        Args:
            question: The original question
            answer: The candidate answer

        Returns:
            VerificationResult with confidence assessment
        """
        return await self.answer_verifier.verify(
            question or self._original_question,
            answer,
            self.evidence_store,
        )

    def get_status(self) -> dict:
        """Get current orchestrator status.

        Returns:
            Dict with round, evidence count, confidence, queue size
        """
        return {
            "current_round": self._current_round,
            "evidence_count": self.evidence_store.get_evidence_count(),
            "confidence": round(self.evidence_store.get_confidence(), 3),
            "queue_size": len(self._query_queue),
            "searched_count": len(self._searched_queries),
        }

    def reset(self) -> None:
        """Reset orchestrator state for a new search session."""
        self._current_round = 0
        self._query_queue.clear()
        self._searched_queries.clear()
        self._original_question = ""
        self.evidence_store.clear()
        logger.debug("SearchOrchestrator state reset")
