# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Adapter to drive a small Terminal-Bench 2 split through the VaG
admission pipeline.

This adapter intentionally does **not** vendor Terminal-Bench 2 itself
(that benchmark has its own Docker harness and licence). Instead it
defines a minimal task schema and a runner that:

  1. Reads a task split file (JSONL) describing each task's prompt,
     working dir, and expected behaviour markers.
  2. Generates a candidate SKILL.md per task via a user-provided
     ``skill_generator`` callable (or a deterministic stub for tests).
  3. Pushes each generated skill through ``SkillAdmissionGate`` and writes
     per-task admission outcomes plus aggregate stats.

To attach a real Terminal-Bench 2 split, point ``--split-path`` at a JSONL
file produced by ``tools/export_tb2_split.py`` (TODO for the paper run).
The bundled ``data/tb2_synthetic_split.jsonl`` is a tiny synthetic split
that exercises the adapter end-to-end without external downloads.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.runners.regression_injection import gate_configs


@dataclass(frozen=True)
class TBTask:
    task_id: str
    prompt: str
    working_dir: str
    expected_command_markers: List[str]


SkillGenerator = Callable[[TBTask], Awaitable[str]]


def _default_stub_generator(task: TBTask) -> str:
    return (
        "---\n"
        f"name: tb2-{task.task_id}\n"
        f"description: Generated stub skill for {task.task_id}\n"
        f"when-to-use: tasks similar to '{task.prompt[:60]}...'\n"
        "---\n\n"
        f"Work in `{task.working_dir}` and prefer commands that touch "
        f"{', '.join(task.expected_command_markers) or 'no specific files'}.\n"
    )


async def _stub_generator_async(task: TBTask) -> str:
    return _default_stub_generator(task)


def load_split(path: Path) -> List[TBTask]:
    tasks: List[TBTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(TBTask(
                task_id=row["task_id"],
                prompt=row["prompt"],
                working_dir=row.get("working_dir", "."),
                expected_command_markers=row.get("expected_command_markers", []),
            ))
    return tasks


async def run_split(
    tasks: List[TBTask],
    skill_generator: Optional[SkillGenerator] = None,
    gate_name: str = "vag_full",
) -> List[Dict]:
    generator = skill_generator or _stub_generator_async
    gate = gate_configs()[gate_name]

    outcomes: List[Dict] = []
    for task in tasks:
        skill_md = await generator(task)
        result = await gate.evaluate(skill_md, task=task.task_id)
        outcomes.append({
            "task_id": task.task_id,
            "approved": result.approved,
            "rejected_by": result.rejected_by,
            "skill_md_len": len(skill_md),
        })
    return outcomes


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-path",
        default=str(ROOT / "evaluation" / "vag" / "data" / "tb2_synthetic_split.jsonl"),
    )
    parser.add_argument("--gate", default="vag_full", choices=list(gate_configs().keys()))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tasks = load_split(Path(args.split_path))
    outcomes = await run_split(tasks, gate_name=args.gate)
    summary = {
        "split": args.split_path,
        "gate": args.gate,
        "n_tasks": len(tasks),
        "n_admitted": sum(1 for o in outcomes if o["approved"]),
        "outcomes": outcomes,
    }
    blob = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(blob, encoding="utf-8")
    print(blob)


if __name__ == "__main__":
    asyncio.run(main())
