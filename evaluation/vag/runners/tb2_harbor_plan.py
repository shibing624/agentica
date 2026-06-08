# -*- coding: utf-8 -*-
"""Create a reproducible execution plan for real Terminal-Bench 2.0 runs.

This runner is safe by default: it writes commands and split files but does not
start Docker/Harbor unless ``--execute-oracle-smoke`` is explicitly provided.
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

from evaluation.vag.adapters.harbor_terminal_bench import (
    DEFAULT_DATASET,
    DEFAULT_HF_DATASET,
    build_agent_run_plan,
    build_oracle_smoke_plan,
    discover_task_ids,
    run_plan,
    split_task_ids,
    write_split_files,
)


HF_TASKS_FROM_DOC_PAGE = [
    "adaptive-rejection-sampler",
    "bn-fit-modify",
    "break-filter-js-from-html",
    "build-cython-ext",
    "build-pmars",
    "build-pov-ray",
    "caffe-cifar-10",
    "cancel-async-tasks",
    "chess-best-move",
    "circuit-fibsqrt",
    "cobol-modernization",
    "code-from-image",
    "compile-compcert",
    "configure-git-webserver",
    "constraints-scheduling",
    "count-dataset-tokens",
    "crack-7z-hash",
    "custom-memory-heap-crash",
    "db-wal-recovery",
    "distribution-search",
    "dna-assembly",
    "dna-insert",
    "extract-elf",
    "extract-moves-from-video",
    "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis",
    "filter-js-from-html",
    "financial-document-processor",
    "fix-code-vulnerability",
    "fix-git",
    "fix-ocaml-gc",
    "gcode-to-text",
    "git-leak-recovery",
    "git-multibranch",
    "gpt2-codegolf",
    "headless-terminal",
    "hf-model-inference",
    "install-windows-3.11",
    "kv-store-grpc",
    "large-scale-text-editing",
    "largest-eigenval",
    "llm-inference-batching-scheduler",
    "log-summary-date-ranges",
    "mailman",
    "make-doom-for-mips",
    "make-mips-interpreter",
    "mcmc-sampling-stan",
    "merge-diff-arc-agi-task",
    "model-extraction-relu-logits",
    "modernize-scientific-stack",
]


def _task_ids_from_args(dataset_root: str) -> List[str]:
    if dataset_root:
        return discover_task_ids(Path(dataset_root))
    return HF_TASKS_FROM_DOC_PAGE


def build_plan(args: argparse.Namespace) -> Dict:
    task_ids = _task_ids_from_args(args.dataset_root)
    split = split_task_ids(
        task_ids,
        seed=args.seed,
        event_size=args.event_size,
        holdout_size=args.holdout_size,
        test_size=args.test_size,
    )
    split_paths = write_split_files(split, Path(args.split_out_dir))
    plans = [
        build_oracle_smoke_plan(dataset=args.dataset),
        build_agent_run_plan(
            dataset=args.dataset,
            agent=args.agent,
            model=args.model,
            env=args.env,
            n_tasks=args.n_tasks,
            extra_args=args.extra_arg,
        ),
    ]
    return {
        "dataset": args.dataset,
        "hf_dataset": DEFAULT_HF_DATASET,
        "dataset_root": args.dataset_root or None,
        "task_source": "local_dataset_root" if args.dataset_root else "doc_page_snapshot",
        "n_discovered_tasks": len(task_ids),
        "split": split.to_dict(),
        "split_files": {name: str(path) for name, path in split_paths.items()},
        "commands": [plan.to_dict() for plan in plans],
        "execution_note": (
            "Run Harbor commands on a Linux/Docker host or via Harbor remote env such as Daytona. "
            "The macOS repo can generate plans and ingest results, but should not be assumed to build every TB2 image."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument(
        "--split-out-dir",
        default=str(ROOT / "evaluation" / "vag" / "data"),
    )
    parser.add_argument("--out", default=str(ROOT / "results" / "vag" / "tb2_harbor_plan.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-size", type=int, default=50)
    parser.add_argument("--holdout-size", type=int, default=14)
    parser.add_argument("--test-size", type=int, default=25)
    parser.add_argument("--agent", default="claude-code")
    parser.add_argument("--model", default="anthropic/claude-haiku-4-5")
    parser.add_argument("--env", default="daytona")
    parser.add_argument("--n-tasks", type=int, default=32)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--execute-oracle-smoke", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if args.execute_oracle_smoke:
        oracle_plan = build_oracle_smoke_plan(dataset=args.dataset)
        completed = run_plan(oracle_plan)
        print(json.dumps({
            "oracle_smoke_returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
