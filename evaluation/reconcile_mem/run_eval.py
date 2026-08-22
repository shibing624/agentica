# -*- coding: utf-8 -*-
"""Run ReconcileMem replay evaluation over normalized JSONL cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from evaluation.reconcile_mem.reconcile_engine import METHOD_SOURCES, evaluate_predictions, predict_case
from evaluation.reconcile_mem.schemas import Prediction, ReconcileCase


def load_cases(path: Path) -> List[ReconcileCase]:
    cases: List[ReconcileCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cases.append(ReconcileCase.from_dict(json.loads(line)))
    return cases


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReconcileMem evaluation")
    parser.add_argument("--input", required=True, help="Normalized JSONL cases")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and predictions")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_SOURCES.keys()),
        choices=list(METHOD_SOURCES.keys()),
    )
    parser.add_argument("--evidence-limit", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    cases = load_cases(input_path)

    metrics: Dict[str, Dict[str, float]] = {}
    for method in args.methods:
        predictions: List[Prediction] = [
            predict_case(case, method=method, evidence_limit=args.evidence_limit)
            for case in cases
        ]
        metrics[method] = evaluate_predictions(cases, predictions)
        write_jsonl(
            output_dir / f"predictions-{method}.jsonl",
            [prediction.to_dict() for prediction in predictions],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
