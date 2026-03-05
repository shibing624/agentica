# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: AnswerVerifier - Verify candidate answers against collected evidence.

Responsibilities:
1. Check candidate answer consistency with all evidence
2. Identify conflicting evidence
3. Reverse-verify by searching with answer as keyword
4. Assess overall confidence
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable, Awaitable

from agentica.model.message import Message
from agentica.search.evidence_store import EvidenceStore
from agentica.utils.log import logger
from agentica.prompts.base.utils import load_prompt as _load_prompt

# Load prompt template
ANSWER_VERIFY_PROMPT = _load_prompt("answer_verify.md")


@dataclass
class VerificationResult:
    """Result of answer verification."""

    is_confident: bool  # Whether the answer is confidently verified
    confidence_score: float  # Overall confidence score (0-1)
    reasoning: str  # Verification reasoning
    conflicting_evidence: List[str] = field(default_factory=list)  # Conflicting evidence items
    suggested_queries: List[str] = field(default_factory=list)  # Queries for further verification


@dataclass
class AnswerVerifier:
    """Answer verification with cross-validation and reverse search.

    Verifies candidate answers against collected evidence,
    identifies contradictions, and optionally performs reverse searches.
    """

    model: Any = None  # LLM model instance

    async def verify(
        self,
        question: str,
        candidate_answer: str,
        evidence_store: EvidenceStore,
    ) -> VerificationResult:
        """Verify a candidate answer against collected evidence.

        Steps:
        1. Check consistency between answer and each evidence item
        2. Identify contradictions
        3. Assess overall confidence

        Args:
            question: The original question
            candidate_answer: The candidate answer to verify
            evidence_store: The evidence store with collected evidence

        Returns:
            VerificationResult with confidence assessment
        """
        if self.model is None:
            # Without a model, return basic confidence from evidence store
            confidence = evidence_store.get_confidence()
            return VerificationResult(
                is_confident=confidence >= 0.6,
                confidence_score=confidence,
                reasoning="No model available for verification, using evidence confidence.",
            )

        evidence_list = evidence_store.get_all_evidence()
        if not evidence_list:
            return VerificationResult(
                is_confident=False,
                confidence_score=0.0,
                reasoning="No evidence available to verify the answer.",
            )

        evidence_text = "\n".join(
            f"[{i+1}. {e.source_title or e.source_url}] {e.content}" 
            for i, e in enumerate(evidence_list[:15])
        )

        prompt = ANSWER_VERIFY_PROMPT.format(
            question=question,
            candidate_answer=candidate_answer,
            evidence=evidence_text,
        ) if ANSWER_VERIFY_PROMPT else (
            f"Question: {question}\n\n"
            f"Candidate Answer: {candidate_answer}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Verify whether the candidate answer is correct based on the evidence.\n"
            f"Check for:\n"
            f"1. Consistency: Does the answer align with the evidence?\n"
            f"2. Contradictions: Is there any evidence that contradicts the answer?\n"
            f"3. Completeness: Is the evidence sufficient to confirm the answer?\n\n"
            f"Output JSON with fields:\n"
            f'  "is_confident": true/false,\n'
            f'  "confidence_score": 0.0-1.0,\n'
            f'  "reasoning": "explanation of verification",\n'
            f'  "conflicting_evidence": ["list of conflicting items"],\n'
            f'  "suggested_queries": ["queries for further verification if not confident"]\n'
            f"Output only valid JSON."
        )

        try:
            response = await self.model.response([
                Message(role="system", content="You are a fact-checking expert. Output only valid JSON."),
                Message(role="user", content=prompt),
            ])
            return self._parse_verification(response.content)
        except Exception as e:
            logger.error(f"Answer verification failed: {e}")
            return VerificationResult(
                is_confident=False,
                confidence_score=0.3,
                reasoning=f"Verification failed: {e}",
            )

    async def reverse_verify(
        self,
        question: str,
        answer: str,
        search_fn: Callable[[str], Awaitable[str]],
    ) -> VerificationResult:
        """Reverse verification: search with the answer as keyword to confirm.

        Constructs a reverse search query using the answer and checks if
        search results support the answer.

        Args:
            question: The original question
            answer: The candidate answer
            search_fn: Async function to perform web search

        Returns:
            VerificationResult from reverse verification
        """
        # Build reverse query: combine answer with key terms from question
        question_keywords = question[:80]
        reverse_query = f"{answer} {question_keywords}"

        try:
            results = await search_fn(reverse_query)

            if self.model is None:
                # Basic check: if search returned results, answer is somewhat confirmed
                has_results = bool(results and len(results) > 50)
                return VerificationResult(
                    is_confident=has_results,
                    confidence_score=0.6 if has_results else 0.3,
                    reasoning="Reverse search returned results" if has_results else "Reverse search returned no results",
                )

            # Use LLM to assess if reverse search results support the answer
            prompt = (
                f"Question: {question}\n"
                f"Answer being verified: {answer}\n\n"
                f"Reverse search results for '{reverse_query}':\n{results[:3000]}\n\n"
                f"Do the search results support that '{answer}' is the correct answer to the question?\n"
                f"Output JSON: {{'is_confirmed': true/false, 'reasoning': 'explanation'}}"
            )

            response = await self.model.response([
                Message(role="user", content=prompt),
            ])

            text = response.content.strip() if response.content else ""
            # Strip markdown
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            try:
                data = json.loads(text)
                confirmed = data.get("is_confirmed", False)
                reasoning = data.get("reasoning", "")
                return VerificationResult(
                    is_confident=confirmed,
                    confidence_score=0.8 if confirmed else 0.3,
                    reasoning=f"Reverse verification: {reasoning}",
                )
            except json.JSONDecodeError:
                return VerificationResult(
                    is_confident=False,
                    confidence_score=0.4,
                    reasoning="Could not parse reverse verification result",
                )
        except Exception as e:
            logger.error(f"Reverse verification failed: {e}")
            return VerificationResult(
                is_confident=False,
                confidence_score=0.3,
                reasoning=f"Reverse verification failed: {e}",
            )

    @staticmethod
    def _parse_verification(content: Optional[str]) -> VerificationResult:
        """Parse LLM response to extract VerificationResult.

        Args:
            content: Raw LLM response

        Returns:
            VerificationResult
        """
        if not content:
            return VerificationResult(
                is_confident=False,
                confidence_score=0.0,
                reasoning="Empty verification response",
            )

        text = content.strip()

        # Strip markdown code block
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")

            return VerificationResult(
                is_confident=bool(data.get("is_confident", False)),
                confidence_score=float(data.get("confidence_score", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                conflicting_evidence=data.get("conflicting_evidence", []),
                suggested_queries=data.get("suggested_queries", []),
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Failed to parse verification JSON: {e}")
            # Fallback: try to infer from text
            is_confident = any(w in text.lower() for w in ["confident", "correct", "verified", "consistent"])
            return VerificationResult(
                is_confident=is_confident,
                confidence_score=0.6 if is_confident else 0.3,
                reasoning=text[:500],
            )
