# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Build paper-ready VaG experiment tables from JSON outputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.analysis.summarize_admission import summarize
from evaluation.vag.runners.downstream_stress import render_table as render_stress_table


DEFAULT_ADMISSION_PATH = ROOT / "results" / "vag" / "hot_pollution_rate.json"
DEFAULT_STRESS_PATH = ROOT / "results" / "vag" / "downstream_stress.json"
DEFAULT_CROSS_MODEL_PATH = ROOT / "evaluation" / "vag" / "data" / "cross_model_sensitivity.json"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_cross_model_table(rows: List[Dict]) -> str:
    lines = [
        "| Judge model | Miss rate | Harmful admitted ↓ | Benign retained ↑ |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {miss_rate:.2f} | {harmful_admission_rate:.3f} | "
            "{benign_retention_rate:.3f} |".format(**row)
        )
    return "\n".join(lines)


def build_tables(
    admission_path: Path = DEFAULT_ADMISSION_PATH,
    stress_path: Path = DEFAULT_STRESS_PATH,
    cross_model_path: Path = DEFAULT_CROSS_MODEL_PATH,
) -> Dict[str, str]:
    return {
        "admission": summarize(_load_json(admission_path)),
        "downstream_stress": render_stress_table(_load_json(stress_path)),
        "cross_model": render_cross_model_table(_load_json(cross_model_path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admission", default=str(DEFAULT_ADMISSION_PATH))
    parser.add_argument("--stress", default=str(DEFAULT_STRESS_PATH))
    parser.add_argument("--cross-model", default=str(DEFAULT_CROSS_MODEL_PATH))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tables = build_tables(
        admission_path=Path(args.admission),
        stress_path=Path(args.stress),
        cross_model_path=Path(args.cross_model),
    )
    blob = json.dumps(tables, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob, encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    main()
