# -*- coding: utf-8 -*-
"""Aider Polyglot — agent edits files until pytest is green. No Docker."""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional

from .cli_agents import run_cli_agent
from .common import (
    INSTRUCTIONS_ADDENDUM,
    TEST_FAILURES,
    SampleResult,
    build_coding_agent,
    cache_path,
    isolated_home,
    slice_items,
)
from .execute import run_command, run_pytest
from .metrics import (
    add_usage,
    empty_metrics,
    finalize_metrics,
    inspect_run_response,
    measure_collateral,
    snapshot_worktree,
)

POLYGLOT_REPO = "https://github.com/Aider-AI/polyglot-benchmark.git"
PRACTICE_REL = Path("python") / "exercises" / "practice"


def ensure_polyglot_repo(root: Optional[Path] = None) -> Path:
    dest = root or cache_path("polyglot-benchmark")
    marker = dest / PRACTICE_REL
    if marker.is_dir() and any(marker.iterdir()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    clone = run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            POLYGLOT_REPO,
            str(dest),
        ],
        timeout=180,
    )
    if not clone.ok:
        raise RuntimeError(f"git clone polyglot-benchmark failed:\n{clone.output}")
    sparse = run_command(
        ["git", "sparse-checkout", "set", "python"],
        cwd=dest,
        timeout=60,
    )
    if not sparse.ok:
        raise RuntimeError(f"sparse-checkout failed:\n{sparse.output}")
    return dest


def list_exercises(repo: Path, language: str = "python") -> List[Path]:
    practice = repo / language / "exercises" / "practice"
    if not practice.is_dir():
        return []
    return sorted(p for p in practice.iterdir() if p.is_dir() and (p / ".meta" / "config.json").exists())


def _read_config(exercise: Path) -> dict:
    return json.loads((exercise / ".meta" / "config.json").read_text(encoding="utf-8"))


def build_instructions(exercise: Path, solution_files: List[str]) -> str:
    parts: List[str] = []
    intro = exercise / ".docs" / "introduction.md"
    if intro.exists():
        parts.append(intro.read_text(encoding="utf-8"))
    instructions = exercise / ".docs" / "instructions.md"
    if instructions.exists():
        parts.append(instructions.read_text(encoding="utf-8"))
    extra = exercise / ".docs" / "instructions.append.md"
    if extra.exists():
        parts.append(extra.read_text(encoding="utf-8"))
    file_list = " ".join(Path(name).name for name in solution_files)
    parts.append(INSTRUCTIONS_ADDENDUM.format(file_list=file_list))
    return "\n".join(parts)


def copy_exercise(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def score_exercise(work: Path, test_files: List[str], timeout: int = 180):
    return run_pytest(work, test_files, timeout=timeout)


def apply_example_solution(src: Path, work: Path) -> bool:
    """Copy .meta/example.py over the stub — used by dry-run to prove the judge."""
    config = _read_config(src)
    examples = config.get("files", {}).get("example", [])
    solutions = config.get("files", {}).get("solution", [])
    if not examples or not solutions:
        return False
    example = src / examples[0]
    if not example.exists():
        return False
    target = work / solutions[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(example, target)
    return True


async def run_one_exercise(
    src: Path,
    *,
    model,
    work_root: Path,
    home: Path,
    tries: int,
    tool_call_limit: int,
    test_timeout: int,
    agent_kind: str = "agentica",
    agent_timeout: int = 300,
    cli_env: Optional[dict] = None,
    cli_model: str = "",
) -> SampleResult:
    config = _read_config(src)
    solution_files = list(config.get("files", {}).get("solution", []))
    test_files = list(config.get("files", {}).get("test", []))
    work = copy_exercise(src, work_root / src.name)
    prompt = build_instructions(work, solution_files)
    file_list = " ".join(Path(name).name for name in solution_files)
    started = time.time()
    last_error = ""
    passed = False
    prediction = ""
    used_tries = 0
    timed_out = False
    crashed = False
    abort_reason = ""
    had_error = False
    metrics = empty_metrics() if agent_kind == "agentica" else None
    model_id = (model.id if model is not None else "") or agent_kind
    snapshot_worktree(work)

    agent = None
    if agent_kind == "agentica":
        agent = build_coding_agent(model, work, home, tool_call_limit=tool_call_limit)
    message = prompt
    for attempt in range(1, tries + 1):
        used_tries = attempt
        if agent_kind == "agentica":
            try:
                response = await agent.run(message)
            except Exception as exc:
                crashed = True
                abort_reason = type(exc).__name__
                last_error = str(exc)
                break
            prediction = response.content or ""
            metrics = add_usage(metrics, inspect_run_response(response))
            if response.break_reason:
                abort_reason = str(response.break_reason)
        else:
            text, ext, crashed_now, abort = await asyncio.to_thread(
                run_cli_agent,
                agent_kind,
                work,
                message,
                agent_timeout,
                model=cli_model,
                env=cli_env,
            )
            prediction = text or prediction
            metrics = ext if metrics is None else add_usage(metrics, ext)
            if (ext.get("abort_reason") or "").endswith("_not_found"):
                crashed = True
                abort_reason = abort
                last_error = text
                break
            if ext.get("timed_out"):
                timed_out = True
                abort_reason = abort
            elif crashed_now:
                crashed = True
                abort_reason = abort
        judged = score_exercise(work, test_files, timeout=test_timeout)
        if judged.timed_out:
            timed_out = True
            last_error = judged.output
            break
        if judged.ok:
            passed = True
            last_error = ""
            break
        had_error = True
        last_error = judged.output
        if timed_out:
            break
        message = last_error + TEST_FAILURES.format(file_list=file_list)

    if metrics is None:
        metrics = empty_metrics()
    collateral = measure_collateral(work, solution_files)
    metrics = add_usage(metrics, collateral)
    metrics["model"] = metrics.get("model") or model_id
    duration_s = round(time.time() - started, 2)
    metrics = finalize_metrics(
        metrics,
        passed=passed,
        duration_s=duration_s,
        prediction=prediction,
        timed_out=timed_out,
        crashed=crashed,
        abort_reason=abort_reason,
        had_error=had_error,
    )

    return SampleResult(
        bench="polyglot",
        task_id=src.name,
        passed=passed,
        duration_s=duration_s,
        error=last_error[-2000:],
        prediction=prediction,
        tries=used_tries,
        extra={
            "solution_files": solution_files,
            "test_files": test_files,
            "agent": agent_kind,
        },
        metrics=metrics,
    )


async def run_polyglot(
    *,
    model,
    max_samples: int,
    offset: int,
    tries: int,
    tool_call_limit: int,
    language: str,
    keywords: Optional[str],
    repo: Optional[Path],
    output_dir: Path,
    test_timeout: int = 180,
    agent_kind: str = "agentica",
    agent_timeout: int = 300,
    cli_env: Optional[dict] = None,
    cli_model: str = "",
) -> List[SampleResult]:
    repo_dir = ensure_polyglot_repo(repo)
    exercises = list_exercises(repo_dir, language=language)
    if keywords:
        keys = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        exercises = [p for p in exercises if any(k in p.name.lower() for k in keys)]
    exercises = slice_items(exercises, max_samples, offset)
    if not exercises:
        raise RuntimeError(f"no polyglot exercises matched language={language!r} keywords={keywords!r}")

    results: List[SampleResult] = []
    run_root = output_dir / "workdir"
    with isolated_home(output_dir / "home") as home:
        for src in exercises:
            print(f"[polyglot] {src.name}")
            result = await run_one_exercise(
                src,
                model=model,
                work_root=run_root,
                home=home,
                tries=tries,
                tool_call_limit=tool_call_limit,
                test_timeout=test_timeout,
                agent_kind=agent_kind,
                agent_timeout=agent_timeout,
                cli_env=cli_env,
                cli_model=cli_model,
            )
            status = "PASS" if result.passed else "FAIL"
            m = result.metrics or {}
            print(
                f"  {status}  {result.duration_s}s  tries={result.tries}  "
                f"tools={'-' if m.get('tool_calls') is None else m.get('tool_calls')}  "
                f"api={'-' if m.get('api_calls') is None else m.get('api_calls')}  "
                f"in={m.get('input_tokens', 0)}  out={m.get('output_tokens', 0)}  "
                f"${m.get('cost_usd', 0):.4f}"
            )
            results.append(result)
    return results


def dry_run_polyglot(max_samples: int = 1, repo: Optional[Path] = None) -> dict:
    """No LLM: stub must fail pytest, official example must pass."""
    repo_dir = ensure_polyglot_repo(repo)
    exercises = slice_items(list_exercises(repo_dir), max_samples)
    if not exercises:
        raise RuntimeError("polyglot repo has no python exercises")
    src = exercises[0]
    config = _read_config(src)
    test_files = list(config.get("files", {}).get("test", []))
    work_stub = cache_path("dry-run", "polyglot-stub", src.name)
    copy_exercise(src, work_stub)
    stub = score_exercise(work_stub, test_files, timeout=60)
    work_gold = cache_path("dry-run", "polyglot-gold", src.name)
    copy_exercise(src, work_gold)
    applied = apply_example_solution(src, work_gold)
    gold = score_exercise(work_gold, test_files, timeout=60) if applied else None
    return {
        "task_id": src.name,
        "stub_failed": not stub.ok,
        "example_applied": applied,
        "example_passed": bool(gold and gold.ok),
        "stub_output": stub.output[-500:],
        "gold_output": (gold.output[-500:] if gold else "no .meta/example.py"),
    }
