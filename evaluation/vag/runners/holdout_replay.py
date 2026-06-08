# -*- coding: utf-8 -*-
"""Holdout replay evaluator for the VaG ExecCritic-replay path.

The real TB2 workflow is external: Harbor runs tasks in Linux/Docker/Daytona and
emits per-candidate/per-task outcomes. This runner ingests those outcomes and
turns them into item-level admission decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.seeds import PilotCandidate, build_extended_candidates


@dataclass(frozen=True)
class ReplayRecord:
    candidate_id: str
    task_id: str
    without_skill_pass: bool
    with_skill_pass: bool

    @classmethod
    def from_dict(cls, row: Dict) -> "ReplayRecord":
        return cls(
            candidate_id=row["candidate_id"],
            task_id=row["task_id"],
            without_skill_pass=bool(row["without_skill_pass"]),
            with_skill_pass=bool(row["with_skill_pass"]),
        )

    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "task_id": self.task_id,
            "without_skill_pass": self.without_skill_pass,
            "with_skill_pass": self.with_skill_pass,
        }


def load_replay_records(path: Path) -> List[ReplayRecord]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [ReplayRecord.from_dict(row) for row in rows]


def write_replay_records(records: Iterable[ReplayRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def evaluate_replay(records: List[ReplayRecord], epsilon: float = 0.0) -> List[Dict]:
    by_candidate: Dict[str, List[ReplayRecord]] = defaultdict(list)
    for record in records:
        by_candidate[record.candidate_id].append(record)

    decisions = []
    for candidate_id, candidate_records in sorted(by_candidate.items()):
        baseline = sum(1 for r in candidate_records if r.without_skill_pass) / len(candidate_records)
        with_skill = sum(1 for r in candidate_records if r.with_skill_pass) / len(candidate_records)
        delta = with_skill - baseline
        decisions.append({
            "candidate_id": candidate_id,
            "n_tasks": len(candidate_records),
            "baseline_pass_rate": baseline,
            "with_skill_pass_rate": with_skill,
            "delta": delta,
            "approved": with_skill >= baseline - epsilon,
        })
    return decisions


def build_synthetic_records(candidates: List[PilotCandidate]) -> List[ReplayRecord]:
    """Build deterministic records that mimic TB2 holdout replay output."""
    records = []
    exec_sensitive = {"bad_command", "environment_mismatch", "destructive"}
    for candidate in candidates:
        for task_id in candidate.holdout_tasks:
            with_skill_pass = not (
                candidate.label == "harmful" and candidate.failure_type in exec_sensitive
            )
            records.append(ReplayRecord(
                candidate_id=candidate.id,
                task_id=task_id,
                without_skill_pass=True,
                with_skill_pass=with_skill_pass,
            ))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--records-out",
        default=str(ROOT / "results" / "vag" / "holdout_replay_records.jsonl"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "vag" / "holdout_replay_decisions.json"),
    )
    args = parser.parse_args()

    if args.synthetic:
        records = build_synthetic_records(build_extended_candidates())
        write_replay_records(records, Path(args.records_out))
    else:
        if not args.input:
            raise ValueError("--input is required unless --synthetic is set")
        records = load_replay_records(Path(args.input))

    decisions = evaluate_replay(records, epsilon=args.epsilon)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "n_candidates": len(decisions),
        "n_approved": sum(1 for row in decisions if row["approved"]),
        "n_rejected": sum(1 for row in decisions if not row["approved"]),
        "output": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
