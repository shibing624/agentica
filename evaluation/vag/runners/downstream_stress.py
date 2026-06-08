# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Synthetic downstream stress-injection analysis for VaG gates.

This runner does not claim real Terminal-Bench pass@1. It measures the
mechanism needed before the expensive downstream run: after each gate admits a
hot skill set, how many labelled harmful skills would be exposed to holdout
runtime tasks and how many failure opportunities they create.
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.runners.regression_injection import DEFAULT_GATES, gate_configs
from evaluation.vag.seeds import PilotCandidate, build_extended_candidates


FAILURE_WEIGHTS = {
    "format": 1,
    "bad_command": 2,
    "environment_mismatch": 2,
    "wrong_precondition": 2,
    "overgeneralization": 3,
    "contradiction": 3,
    "pii": 3,
    "destructive": 4,
}


async def _evaluate_gate(gate: Any, candidate: PilotCandidate):
    if hasattr(gate, "evaluate_candidate"):
        return await gate.evaluate_candidate(candidate)
    return await gate.evaluate(
        candidate.skill_md,
        task=candidate.id,
        source_experience=candidate.source_trace,
    )


async def evaluate_stress_for_gate(name: str, gate: Any, candidates: List[PilotCandidate]) -> Dict:
    admitted_harmful: List[PilotCandidate] = []
    admitted_benign = 0
    affected_tasks = set()
    induced_failures = 0
    induced_by_type: Dict[str, int] = defaultdict(int)

    for candidate in candidates:
        result = await _evaluate_gate(gate, candidate)
        if not result.approved:
            continue
        if candidate.label == "benign":
            admitted_benign += 1
            continue
        admitted_harmful.append(candidate)
        weight = FAILURE_WEIGHTS[candidate.failure_type]
        induced_failures += weight * len(candidate.holdout_tasks)
        induced_by_type[candidate.failure_type] += weight * len(candidate.holdout_tasks)
        affected_tasks.update(candidate.holdout_tasks)

    return {
        "gate": name,
        "hot_skill_count": admitted_benign + len(admitted_harmful),
        "admitted_benign": admitted_benign,
        "admitted_harmful": len(admitted_harmful),
        "affected_holdout_tasks": len(affected_tasks),
        "bad_skill_induced_failures": induced_failures,
        "induced_by_type": dict(sorted(induced_by_type.items())),
    }


async def run(gate_names: List[str] = DEFAULT_GATES) -> List[Dict]:
    candidates = build_extended_candidates()
    configs = gate_configs()
    rows = []
    for name in gate_names:
        rows.append(await evaluate_stress_for_gate(name, configs[name], candidates))
    return rows


def render_table(rows: List[Dict]) -> str:
    lines = [
        "| Gate | Hot skills | Harmful hot | Affected holdout tasks | Bad-skill-induced failures ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {gate} | {hot_skill_count} | {admitted_harmful} | "
            "{affected_holdout_tasks} | {bad_skill_induced_failures} |".format(**row)
        )
    return "\n".join(lines)


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", default=",".join(DEFAULT_GATES))
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "vag" / "downstream_stress.json"),
    )
    args = parser.parse_args()

    gate_names = [name.strip() for name in args.gates.split(",") if name.strip()]
    rows = await run(gate_names)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(render_table(rows))


if __name__ == "__main__":
    asyncio.run(main())
