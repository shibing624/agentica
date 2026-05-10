# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: InterCode Bash transfer adapter.

Mirrors the Terminal-Bench 2 adapter API but uses the InterCode-Bash task
schema. Used to test whether VaG admission gates trained on Terminal-Bench
traces transfer (zero-shot) to a different bash-task distribution.

Plug a real InterCode dump in via ``--split-path`` pointing at a JSONL
exported from the InterCode harness. A small synthetic split is bundled
under ``data/intercode_bash_synthetic.jsonl`` for adapter-level tests.
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
class InterCodeTask:
    task_id: str
    instruction: str
    gold_command: str


SkillGenerator = Callable[[InterCodeTask], Awaitable[str]]


async def _stub_generator(task: InterCodeTask) -> str:
    return (
        "---\n"
        f"name: ic-bash-{task.task_id}\n"
        f"description: Bash skill for '{task.instruction[:60]}'\n"
        "when-to-use: bash command synthesis tasks\n"
        "---\n\n"
        f"Plan: produce a command similar to `{task.gold_command}`.\n"
    )


def load_split(path: Path) -> List[InterCodeTask]:
    tasks: List[InterCodeTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(InterCodeTask(
                task_id=row["task_id"],
                instruction=row["instruction"],
                gold_command=row.get("gold_command", ""),
            ))
    return tasks


async def run_split(
    tasks: List[InterCodeTask],
    skill_generator: Optional[SkillGenerator] = None,
    gate_name: str = "vag_full",
) -> List[Dict]:
    generator = skill_generator or _stub_generator
    gate = gate_configs()[gate_name]

    outcomes: List[Dict] = []
    for task in tasks:
        skill_md = await generator(task)
        result = await gate.evaluate(skill_md, task=task.task_id)
        outcomes.append({
            "task_id": task.task_id,
            "approved": result.approved,
            "rejected_by": result.rejected_by,
        })
    return outcomes


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-path",
        default=str(ROOT / "evaluation" / "vag" / "data" / "intercode_bash_synthetic.jsonl"),
    )
    parser.add_argument("--gate", default="vag_full", choices=list(gate_configs().keys()))
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
