# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Summarize Regression Injection ablation results into a Markdown table.

Reads the JSON produced by ``runners/regression_injection.py`` and emits a
critic-ablation table suitable for the paper / appendix:

| Config | Harmful admission ↓ | Benign retention ↑ | Cost / candidate ↓ | Latency ms | Per-failure breakdown |
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[3]


def _harmful_breakdown(rejected: List[Dict], failure_types: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for row in rejected:
        if row["label"] == "harmful":
            counts[row["failure_type"]] += 1
    return {ft: counts.get(ft, 0) for ft in failure_types}


def summarize(results: List[Dict]) -> str:
    failure_types = sorted({
        r["failure_type"]
        for cfg in results
        for r in cfg["accepted"] + cfg["rejected"]
        if r["label"] == "harmful"
    })

    header = (
        "| Config | Harmful admit ↓ | Benign retention ↑ | Cost / candidate ↓ | "
        "Latency ms | " + " | ".join(failure_types) + " |"
    )
    sep = "|" + "|".join("---" for _ in range(5 + len(failure_types))) + "|"
    rows = [header, sep]
    for cfg in results:
        breakdown = _harmful_breakdown(cfg["rejected"], failure_types)
        cells = [
            cfg["config"],
            f"{cfg['harmful_admission_rate']:.3f}",
            f"{cfg['benign_retention_rate']:.3f}",
            f"{cfg.get('gate_cost_per_candidate_usd', 0.0):.4f}",
            f"{cfg.get('gate_latency_ms', 0.0):.1f}",
        ]
        cells += [f"{breakdown[ft]}" for ft in failure_types]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_json",
        nargs="?",
        default=str(ROOT / "evaluation" / "vag" / "data" / "pilot_ablation_results.json"),
    )
    parser.add_argument("--out", default=None, help="Write Markdown to file (default: stdout)")
    args = parser.parse_args()

    results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    table = summarize(results)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
    else:
        sys.stdout.write(table + "\n")


if __name__ == "__main__":
    main()
