# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Case-study runner — drives a handful of labelled candidates
through the full VaG admission gate and emits both the raw provenance JSONL
events and a Markdown case-study report citing those events as evidence.

The intent is to back the paper's qualitative section ("admission decisions
explained") with concrete, line-numbered provenance evidence rather than
synthetic prose. Each case study corresponds to one candidate from
``evaluation/vag/seeds.py`` and shows:

  1. What the gate decided.
  2. Which critic fired.
  3. The exact provenance event recorded.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.lifecycle import (
    append_provenance_event,
    get_provenance_path,
    read_provenance_events,
)
from evaluation.vag.runners.regression_injection import gate_configs
from evaluation.vag.seeds import build_extended_candidates


CASE_IDS = [
    # benign — should pass all gates
    "benign_000",
    # mutated — each maps to a distinct failure_type to vary the evidence shape
    "harmful_000_00",
    "harmful_001_01",
    "harmful_002_02",
    "harmful_003_03",
    "harmful_004_06",
    "harmful_005_07",
]


async def run_cases(out_dir: Path) -> Dict[str, List[Dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = get_provenance_path(out_dir)
    if provenance_path.exists():
        provenance_path.unlink()

    candidates = {c.id: c for c in build_extended_candidates() if c.id in CASE_IDS}
    missing = set(CASE_IDS) - set(candidates)
    if missing:
        raise RuntimeError(f"missing seed candidates: {sorted(missing)}")

    full_gate = gate_configs()["vag_full"]
    studies: Dict[str, List[Dict]] = {}
    for case_id in CASE_IDS:
        candidate = candidates[case_id]
        result = await full_gate.evaluate(candidate.skill_md, task=case_id)
        event = {
            "event": "gate_decision",
            "case_id": case_id,
            "label": candidate.label,
            "failure_type": candidate.failure_type,
            "approved": result.approved,
            "rejected_by": result.rejected_by,
            "verdicts": [v.to_dict() for v in result.verdicts],
        }
        append_provenance_event(out_dir, event)
        studies[case_id] = event

    return studies


def render_report(studies: Dict[str, Dict], out_dir: Path) -> str:
    provenance_path = get_provenance_path(out_dir)
    try:
        rel_path = provenance_path.relative_to(ROOT)
    except ValueError:
        rel_path = provenance_path
    lines = [
        "# VaG Case Studies",
        "",
        f"Provenance log: `{rel_path}`",
        "",
    ]
    events = read_provenance_events(out_dir)
    event_by_case = {e["case_id"]: (i, e) for i, e in enumerate(events) if "case_id" in e}

    for case_id, study in studies.items():
        line_no, ev = event_by_case[case_id]
        lines.append(f"## {case_id} (`{study['label']}` / `{study['failure_type']}`)")
        lines.append("")
        lines.append(f"- **Decision**: {'admitted' if study['approved'] else 'rejected'}")
        if study["rejected_by"]:
            lines.append(f"- **Rejected by**: {', '.join(study['rejected_by'])}")
        lines.append(f"- **Provenance line**: {provenance_path.name}:{line_no + 1}")
        lines.append("- **Per-critic verdicts**:")
        for v in study["verdicts"]:
            mark = "approved" if v["approved"] else "rejected"
            issue = f" — {v['issues']}" if v["issues"] else ""
            lines.append(f"  - `{v['critic']}`: {mark}{issue}")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    out_dir = ROOT / "evaluation" / "vag" / "data" / "case_studies"
    studies = await run_cases(out_dir)
    report = render_report(studies, out_dir)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
