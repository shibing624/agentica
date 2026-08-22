# -*- coding: utf-8 -*-
"""Best-effort converter from public conflict-memory JSONL to ReconcileMem cases.

MemConflict-style datasets may expose different field names across releases.
This converter keeps the mapping configurable through common fallback keys and
produces normalized JSONL for `run_eval.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from evaluation.reconcile_mem.schemas import EvidenceItem, ReconcileCase


QUERY_KEYS = ["query", "question", "input"]
ANSWER_KEYS = ["answer", "gold_answer", "target"]
EVIDENCE_KEYS = ["supporting_memories", "evidence", "memories", "history"]
CONFLICT_TYPE_KEYS = ["conflict_type", "type", "category"]


def first_value(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return default


def normalize_evidence(value: Any) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    if isinstance(value, str):
        return [EvidenceItem(content=value, source="conversation", evidence_id="raw")]
    if not isinstance(value, list):
        return items
    for index, item in enumerate(value):
        if isinstance(item, str):
            items.append(EvidenceItem(content=item, source="conversation", evidence_id=f"evidence::{index}"))
        elif isinstance(item, dict):
            content = str(item.get("content", item.get("text", item.get("memory", ""))))
            source = str(item.get("source", "conversation"))
            evidence_id = str(item.get("id", item.get("evidence_id", f"evidence::{index}")))
            items.append(EvidenceItem(content=content, source=source, evidence_id=evidence_id))
    return items


def convert_row(row: Dict[str, Any], index: int) -> ReconcileCase:
    evidence_value = None
    for key in EVIDENCE_KEYS:
        if key in row:
            evidence_value = row[key]
            break
    evidence = normalize_evidence(evidence_value)
    return ReconcileCase(
        case_id=str(row.get("id", row.get("case_id", index))),
        query=first_value(row, QUERY_KEYS),
        gold_answer=first_value(row, ANSWER_KEYS),
        gold_evidence_refs=[item.evidence_id for item in evidence if item.evidence_id],
        conversations=evidence,
        metadata={
            "source_dataset": "memconflict_like",
            "conflict_type": first_value(row, CONFLICT_TYPE_KEYS),
            "raw_keys": sorted(row.keys()),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert conflict-memory JSONL to ReconcileMem schema")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            case = convert_row(json.loads(line), index)
            target.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    print(f"converted {count} rows to {output_path}")


if __name__ == "__main__":
    main()
