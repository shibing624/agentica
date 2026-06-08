# -*- coding: utf-8 -*-
"""Utilities for planning real Terminal-Bench 2.0 runs through Harbor.

The Mac workstation should normally orchestrate these runs only. The actual
Terminal-Bench task containers are Linux/Docker workloads and should run on a
Linux host, Docker Desktop, or Harbor's remote environment such as Daytona.
"""
from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_DATASET = "terminal-bench/terminal-bench-2"
DEFAULT_HF_DATASET = "harborframework/terminal-bench-2.0"


@dataclass(frozen=True)
class HarborCommandPlan:
    name: str
    command: List[str]
    notes: List[str] = field(default_factory=list)

    def shell_command(self) -> str:
        return " ".join(self.command)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "command": self.command,
            "shell_command": self.shell_command(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class TB2Split:
    event: List[str]
    holdout: List[str]
    test: List[str]

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "event": list(self.event),
            "holdout": list(self.holdout),
            "test": list(self.test),
        }


def discover_task_ids(dataset_root: Path) -> List[str]:
    """Return task directory names from a local HF dataset checkout."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Terminal-Bench dataset root does not exist: {dataset_root}")
    task_ids = []
    for child in dataset_root.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            task_ids.append(child.name)
    return sorted(task_ids)


def split_task_ids(
    task_ids: Iterable[str],
    seed: int = 42,
    event_size: int = 50,
    holdout_size: int = 14,
    test_size: int = 25,
) -> TB2Split:
    """Create deterministic Event/Holdout/Test splits.

    If fewer than 89 tasks are available (e.g. adapter tests), the requested
    proportions are scaled down while preserving non-overlap.
    """
    ids = sorted(set(task_ids))
    if not ids:
        raise ValueError("task_ids must not be empty")
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    requested_total = event_size + holdout_size + test_size
    if len(shuffled) >= requested_total:
        event_count = event_size
        holdout_count = holdout_size
        test_count = test_size
    else:
        event_count = max(1, round(len(shuffled) * event_size / requested_total))
        holdout_count = max(1, round(len(shuffled) * holdout_size / requested_total))
        remaining = len(shuffled) - event_count - holdout_count
        test_count = max(1, remaining)
        while event_count + holdout_count + test_count > len(shuffled):
            if event_count >= holdout_count and event_count > 1:
                event_count -= 1
            elif holdout_count > 1:
                holdout_count -= 1
            else:
                test_count -= 1

    event = shuffled[:event_count]
    holdout = shuffled[event_count:event_count + holdout_count]
    test = shuffled[event_count + holdout_count:event_count + holdout_count + test_count]
    return TB2Split(event=event, holdout=holdout, test=test)


def write_split_files(split: TB2Split, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "event": out_dir / "tbench2_split_event.txt",
        "holdout": out_dir / "tbench2_split_holdout.txt",
        "test": out_dir / "tbench2_split_test.txt",
        "manifest": out_dir / "tbench2_split_manifest.json",
    }
    for name in ["event", "holdout", "test"]:
        paths[name].write_text("\n".join(split.to_dict()[name]) + "\n", encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(split.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def build_oracle_smoke_plan(dataset: str = DEFAULT_DATASET) -> HarborCommandPlan:
    return HarborCommandPlan(
        name="tb2_oracle_smoke",
        command=["harbor", "run", "-d", dataset, "-a", "oracle"],
        notes=[
            "Run this first on the Linux/Docker host to verify Harbor + Docker + TB2 access.",
        ],
    )


def build_agent_run_plan(
    dataset: str = DEFAULT_DATASET,
    agent: str = "claude-code",
    model: Optional[str] = None,
    env: Optional[str] = None,
    n_tasks: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
) -> HarborCommandPlan:
    command = ["harbor", "run", "-d", dataset, "-a", agent]
    if model:
        command.extend(["-m", model])
    if env:
        command.extend(["--env", env])
    if n_tasks is not None:
        command.extend(["-n", str(n_tasks)])
    command.extend(extra_args or [])
    return HarborCommandPlan(
        name=f"tb2_{agent}_run",
        command=command,
        notes=[
            "Execute on a Linux/Docker host or with --env daytona; do not rely on macOS for every TB2 image.",
            "If using a custom Agentica/VaG agent, register it with Harbor and replace -a accordingly.",
            "Harbor task filtering is intentionally left in extra_args because the CLI flag may change.",
        ],
    )


def run_plan(plan: HarborCommandPlan, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        plan.command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_HF_DATASET",
    "HarborCommandPlan",
    "TB2Split",
    "discover_task_ids",
    "split_task_ids",
    "write_split_files",
    "build_oracle_smoke_plan",
    "build_agent_run_plan",
    "run_plan",
]
