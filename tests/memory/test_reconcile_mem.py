# -*- coding: utf-8 -*-
"""Smoke tests for ReconcileMem replay evaluation."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.reconcile_mem.reconcile_engine import evaluate_predictions, predict_case
from evaluation.reconcile_mem.schemas import ReconcileCase


def test_reconcile_mem_smoke_metrics():
    data_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation" / "reconcile_mem" / "data" / "smoke_cases.jsonl"
    )
    cases = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(ReconcileCase.from_dict(json.loads(line)))

    predictions = [
        predict_case(case, method="reconcile_mem", evidence_limit=5) for case in cases
    ]
    metrics = evaluate_predictions(cases, predictions)

    assert metrics["count"] == 2.0
    assert metrics["evidence_recall"] >= 0.5
    assert metrics["answer_contains_gold"] >= 0.5
