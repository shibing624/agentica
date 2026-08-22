# -*- coding: utf-8 -*-
"""Lightweight ReconcileMem evaluation engine.

This module intentionally provides deterministic baselines and metrics. LLM-based
resolvers can be plugged in later, but the default path is cheap and suitable for
log-scale replay.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

from evaluation.reconcile_mem.schemas import EvidenceItem, Prediction, ReconcileCase


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+")


METHOD_SOURCES: Dict[str, Sequence[str]] = {
    "canonical_only": ("memory",),
    "summary_only": ("memory", "memory_candidate"),
    "raw_rag": ("conversation",),
    "reconcile_mem": ("memory", "memory_candidate", "conversation"),
}


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text) if token.strip()]


def relevance_score(query: str, content: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_lower = content.lower()
    hits = sum(1.0 for token in query_tokens if token in content_lower)
    return hits / len(query_tokens)


def iter_items(case: ReconcileCase) -> Iterable[EvidenceItem]:
    yield from case.confirmed_memories
    yield from case.memory_candidates
    yield from case.conversations


def select_evidence(case: ReconcileCase, method: str, limit: int = 5) -> List[EvidenceItem]:
    if method not in METHOD_SOURCES:
        raise ValueError(f"unknown method: {method}")

    allowed_sources = set(METHOD_SOURCES[method])
    scored: List[EvidenceItem] = []
    for item in iter_items(case):
        if item.source not in allowed_sources:
            continue
        item.score = relevance_score(case.query, item.content)
        scored.append(item)

    scored.sort(key=lambda item: (-item.score, source_priority(item.source)))
    return scored[:limit]


def source_priority(source: str) -> int:
    if source == "memory":
        return 0
    if source == "conversation":
        return 1
    if source == "memory_candidate":
        return 2
    return 3


def predict_case(case: ReconcileCase, method: str, evidence_limit: int = 5) -> Prediction:
    selected = select_evidence(case, method=method, limit=evidence_limit)
    predicted_answer = selected[0].content if selected else ""
    predicted_decision = infer_decision(selected)
    return Prediction(
        case_id=case.case_id,
        method=method,
        predicted_answer=predicted_answer,
        predicted_decision=predicted_decision,
        selected_evidence=selected,
    )


def infer_decision(selected: List[EvidenceItem]) -> str:
    if not selected:
        return "unresolved"
    sources = {item.source for item in selected}
    if "conversation" in sources and ("memory" in sources or "memory_candidate" in sources):
        return "reconcile_with_evidence"
    if "conversation" in sources:
        return "answer_from_raw_evidence"
    if "memory_candidate" in sources and "memory" in sources:
        return "keep_both"
    if "memory_candidate" in sources:
        return "candidate_only"
    return "confirmed_memory"


def evaluate_predictions(cases: List[ReconcileCase], predictions: List[Prediction]) -> Dict[str, float]:
    case_by_id = {case.case_id: case for case in cases}
    total = len(predictions)
    if total == 0:
        return {
            "count": 0.0,
            "answer_contains_gold": 0.0,
            "evidence_recall": 0.0,
            "decision_accuracy": 0.0,
            "unsupported_answer_rate": 0.0,
        }

    answer_hits = 0
    evidence_hits = 0
    decision_hits = 0
    unsupported = 0

    for pred in predictions:
        case = case_by_id[pred.case_id]
        if case.gold_answer and case.gold_answer.lower() in pred.predicted_answer.lower():
            answer_hits += 1
        if has_gold_evidence(case, pred):
            evidence_hits += 1
        if case.gold_decision and pred.predicted_decision == case.gold_decision:
            decision_hits += 1
        if pred.predicted_answer and not pred.selected_evidence:
            unsupported += 1

    labeled_decisions = sum(1 for pred in predictions if case_by_id[pred.case_id].gold_decision)
    return {
        "count": float(total),
        "answer_contains_gold": answer_hits / total,
        "evidence_recall": evidence_hits / total,
        "decision_accuracy": decision_hits / labeled_decisions if labeled_decisions else 0.0,
        "unsupported_answer_rate": unsupported / total,
    }


def has_gold_evidence(case: ReconcileCase, prediction: Prediction) -> bool:
    if not case.gold_evidence_refs:
        return False
    selected_ids = set()
    for item in prediction.selected_evidence:
        if item.evidence_id:
            selected_ids.add(item.evidence_id)
        if item.file_path:
            selected_ids.add(item.file_path)
    return any(ref in selected_ids for ref in case.gold_evidence_refs)
