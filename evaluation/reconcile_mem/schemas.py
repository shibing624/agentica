# -*- coding: utf-8 -*-
"""Shared schemas for ReconcileMem evaluation JSONL files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvidenceItem:
    content: str
    source: str
    evidence_id: str = ""
    file_path: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceItem":
        return cls(
            content=str(data.get("content", "")),
            source=str(data.get("source", "")),
            evidence_id=str(data.get("evidence_id", data.get("file_path", ""))),
            file_path=str(data.get("file_path", "")),
            score=float(data.get("score", 0.0) or 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "evidence_id": self.evidence_id,
            "file_path": self.file_path,
            "score": self.score,
        }


@dataclass
class ReconcileCase:
    case_id: str
    query: str
    gold_answer: str = ""
    gold_decision: str = ""
    gold_evidence_refs: List[str] = field(default_factory=list)
    confirmed_memories: List[EvidenceItem] = field(default_factory=list)
    memory_candidates: List[EvidenceItem] = field(default_factory=list)
    conversations: List[EvidenceItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconcileCase":
        return cls(
            case_id=str(data.get("id", data.get("case_id", ""))),
            query=str(data.get("query", "")),
            gold_answer=str(data.get("gold_answer", "")),
            gold_decision=str(data.get("gold_decision", "")),
            gold_evidence_refs=[str(item) for item in data.get("gold_evidence_refs", [])],
            confirmed_memories=[EvidenceItem.from_dict(item) for item in data.get("confirmed_memories", [])],
            memory_candidates=[EvidenceItem.from_dict(item) for item in data.get("memory_candidates", [])],
            conversations=[EvidenceItem.from_dict(item) for item in data.get("conversations", [])],
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.case_id,
            "query": self.query,
            "gold_answer": self.gold_answer,
            "gold_decision": self.gold_decision,
            "gold_evidence_refs": self.gold_evidence_refs,
            "confirmed_memories": [item.to_dict() for item in self.confirmed_memories],
            "memory_candidates": [item.to_dict() for item in self.memory_candidates],
            "conversations": [item.to_dict() for item in self.conversations],
            "metadata": self.metadata,
        }


@dataclass
class Prediction:
    case_id: str
    method: str
    predicted_answer: str
    predicted_decision: str
    selected_evidence: List[EvidenceItem]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.case_id,
            "method": self.method,
            "predicted_answer": self.predicted_answer,
            "predicted_decision": self.predicted_decision,
            "selected_evidence": [item.to_dict() for item in self.selected_evidence],
        }
