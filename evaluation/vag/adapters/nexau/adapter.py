# -*- coding: utf-8 -*-
"""NexAU adapter — minimal schema + admission runner."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.runners.regression_injection import gate_configs


@dataclass(frozen=True)
class NexAUTask:
    task_id: str
    user_intent: str
    candidate_action: str


SkillGenerator = Callable[[NexAUTask], Awaitable[str]]


async def _stub_generator(task: NexAUTask) -> str:
    return (
        "---\n"
        f"name: nexau-{task.task_id}\n"
        f"description: Action-utility skill for '{task.user_intent[:60]}'\n"
        "when-to-use: next-action ranking under user-intent shift\n"
        "---\n\n"
        f"Plan: candidate action = `{task.candidate_action}`.\n"
    )


def load_split(path: Path) -> List[NexAUTask]:
    tasks: List[NexAUTask] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks.append(NexAUTask(
                task_id=row["task_id"],
                user_intent=row["user_intent"],
                candidate_action=row["candidate_action"],
            ))
    return tasks


async def run_split(
    tasks: List[NexAUTask],
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
