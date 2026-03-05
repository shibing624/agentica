# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Search enhancement module for WebSearchAgent.

Components:
- QueryDecomposer: Query decomposition and rewriting
- EvidenceStore: Structured evidence collection and tracking
- AnswerVerifier: Answer verification with cross-validation
- SearchOrchestrator: Search flow orchestration
"""

from agentica.search.query_decomposer import QueryDecomposer
from agentica.search.evidence_store import Evidence, EvidenceStore
from agentica.search.answer_verifier import VerificationResult, AnswerVerifier
from agentica.search.orchestrator import SearchOrchestrator

__all__ = [
    "QueryDecomposer",
    "Evidence",
    "EvidenceStore",
    "VerificationResult",
    "AnswerVerifier",
    "SearchOrchestrator",
]
