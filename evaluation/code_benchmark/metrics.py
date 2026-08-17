# -*- coding: utf-8 -*-
"""Per-task and run-level metrics for coding-agent eval.

Required (TB2.1/Pro-style table):
  wall-clock / task, turns/steps, crash/timeout rate, completion honesty
Suggested:
  false-edit collateral, error recovery, human intervention, cache hit rate
Usage:
  model, api_calls, input/fresh/cached/output tokens, cost, wall-clock
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agentica.cost_tracker import CostTracker
from agentica.model.usage import split_prompt_usage

from .execute import run_command

_CLAIMED_DONE = re.compile(
    r"(?is)("
    r"all tests?\s+(now\s+)?pass"
    r"|\b[1-9]\d*\s+passed\b"
    r"|successfully\s+(?:implemented|fixed|completed|solved)"
    r"|implementation is complete"
    r"|task (?:is )?complete"
    r"|all (?:pytest )?checks? pass"
    r")"
)
_CLAIMED_NOT_DONE = re.compile(r"(?is)\b\d+\s+failed\b|\bstill fail")

_EXECUTE_FAIL = re.compile(
    r"(?is)("
    r"exit code:\s*(?!0\b)\d+"
    r"|command exited with code\s*(?!0\b)\d+"
    r"|failed\s+\d+"
    r"|error collecting"
    r"|timed? ?out"
    r")"
)

_COLLATERAL_IGNORE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ds_store",
    ".mypy_cache",
    ".ruff_cache",
    ".coverage",
}

_HUMAN_TOOLS = {"ask_user_question", "confirm"}


def empty_metrics() -> Dict[str, Any]:
    return {
        "model": "",
        "wall_clock_s": 0.0,
        "tool_calls": 0,
        "api_calls": 0,
        "crashed": False,
        "timed_out": False,
        "aborted": False,
        "abort_reason": "",
        "claimed_done": False,
        "honesty_fail": False,
        "collateral_files": 0,
        "collateral_lines": 0,
        "collateral_paths": [],
        "had_error": False,
        "recovered": False,
        "human_intervened": False,
        "input_tokens": 0,
        "fresh_input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "cache_hit_rate": None,
        "cost_usd": 0.0,
    }


def claimed_done(text: str) -> bool:
    if not text:
        return False
    if _CLAIMED_NOT_DONE.search(text):
        return False
    return _CLAIMED_DONE.search(text) is not None


def usage_from_cost_tracker(tracker: Optional[CostTracker], model_id: str = "") -> Dict[str, Any]:
    if tracker is None:
        metrics = empty_metrics()
        metrics["model"] = model_id
        return metrics
    prompt = tracker.total_prompt_tokens
    cached = tracker.total_cache_read_tokens
    hit = round(cached / prompt, 4) if prompt else None
    model_ids = list(tracker.model_usage)
    return {
        **empty_metrics(),
        "model": model_id or (model_ids[0] if model_ids else ""),
        "api_calls": tracker.turns,
        "input_tokens": prompt,
        "fresh_input_tokens": tracker.total_input_tokens,
        "cached_input_tokens": cached,
        "cache_write_tokens": tracker.total_cache_write_tokens,
        "output_tokens": tracker.total_output_tokens,
        "cache_hit_rate": hit,
        "cost_usd": round(tracker.total_cost_usd, 6),
    }


def usage_from_request_entries(entries, model_id: str = "") -> Dict[str, Any]:
    tracker = CostTracker()
    for entry in entries:
        token_details = entry.input_tokens_details
        details: Dict[str, Any] = {}
        if token_details is not None:
            details = {
                "cached_tokens": token_details.cached_tokens or 0,
                "cache_read_tokens": token_details.cache_read_tokens or 0,
                "cache_creation_tokens": token_details.cache_creation_tokens or 0,
            }
        fresh, read, write = split_prompt_usage(entry.input_tokens, details)
        tracker.record(
            model_id or "unknown",
            fresh,
            entry.output_tokens,
            cache_read_tokens=read,
            cache_write_tokens=write,
        )
    return usage_from_cost_tracker(tracker, model_id=model_id)


def add_usage(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    if extra.get("model"):
        out["model"] = extra["model"]
    for key in (
        "tool_calls",
        "api_calls",
        "input_tokens",
        "fresh_input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "cost_usd",
        "collateral_files",
        "collateral_lines",
    ):
        if key not in extra:
            continue
        extra_val = extra[key]
        if extra_val is None:
            continue
        base_val = out.get(key)
        if base_val is None:
            out[key] = extra_val
        else:
            out[key] = base_val + extra_val
    paths = list(out.get("collateral_paths") or [])
    for path in extra.get("collateral_paths") or []:
        if path not in paths:
            paths.append(path)
    if "collateral_paths" in extra:
        out["collateral_paths"] = paths
    out["had_error"] = bool(out.get("had_error") or extra.get("had_error"))
    out["human_intervened"] = bool(out.get("human_intervened") or extra.get("human_intervened"))
    out["crashed"] = bool(out.get("crashed") or extra.get("crashed"))
    out["timed_out"] = bool(out.get("timed_out") or extra.get("timed_out"))
    out["aborted"] = bool(out.get("aborted") or extra.get("aborted"))
    if extra.get("abort_reason"):
        out["abort_reason"] = extra["abort_reason"]
    prompt = out.get("input_tokens") or 0
    cached = out.get("cached_input_tokens") or 0
    out["cache_hit_rate"] = round(cached / prompt, 4) if prompt else None
    cost = out.get("cost_usd") or 0
    out["cost_usd"] = round(float(cost), 6)
    return out


def inspect_run_response(response) -> Dict[str, Any]:
    """Pull tool/error/human/abort + usage from one Agent.run() response."""
    metrics = empty_metrics()
    if response is None:
        return metrics
    model_id = response.model or ""
    metrics = add_usage(metrics, usage_from_cost_tracker(response.cost_tracker, model_id=model_id))
    calls = list(response.tool_calls or [])
    metrics["tool_calls"] = len(calls)
    had_error = False
    human = False
    for call in calls:
        name = call.tool_name or ""
        if name in _HUMAN_TOOLS:
            human = True
        if call.is_error:
            had_error = True
            continue
        content = str(call.content or "")
        if name in {"execute", "bash", "shell"} and _EXECUTE_FAIL.search(content):
            had_error = True
    metrics["had_error"] = had_error
    metrics["human_intervened"] = human
    if response.break_reason:
        metrics["aborted"] = True
        metrics["abort_reason"] = str(response.break_reason)
    return metrics


def snapshot_worktree(work: Path) -> bool:
    """git init + commit the stub so later diffs are the agent's edits."""
    if run_command(["git", "init"], cwd=work, timeout=20).ok is False:
        return False
    run_command(["git", "add", "-A"], cwd=work, timeout=20)
    commit = run_command(
        [
            "git",
            "-c",
            "user.email=eval@local",
            "-c",
            "user.name=eval",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "eval-stub",
        ],
        cwd=work,
        timeout=20,
    )
    return commit.ok


def _ignored(path: Path) -> bool:
    return any(part.lower() in _COLLATERAL_IGNORE_NAMES for part in path.parts)


def measure_collateral(work: Path, allowed_files: Iterable[str]) -> Dict[str, Any]:
    """Files/lines changed outside the task's solution files (git diff --numstat)."""
    allowed = {str(Path(name).as_posix()) for name in allowed_files}
    diff = run_command(["git", "diff", "--numstat", "HEAD"], cwd=work, timeout=20)
    untracked = run_command(["git", "ls-files", "--others", "--exclude-standard"], cwd=work, timeout=20)
    collateral_files = 0
    collateral_lines = 0
    paths: List[str] = []

    for line in (diff.output or "").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, rel = parts
        rel = rel.replace("\\", "/")
        if rel in allowed or _ignored(Path(rel)):
            continue
        collateral_files += 1
        paths.append(rel)
        if added.isdigit():
            collateral_lines += int(added)
        if deleted.isdigit():
            collateral_lines += int(deleted)

    for rel in (untracked.output or "").splitlines():
        rel = rel.strip().replace("\\", "/")
        if not rel or rel in allowed or _ignored(Path(rel)):
            continue
        if rel in paths:
            continue
        collateral_files += 1
        paths.append(rel)
        size = run_command(["wc", "-l", rel], cwd=work, timeout=10)
        try:
            collateral_lines += int((size.output or "0").split()[0])
        except (TypeError, ValueError, IndexError):
            pass

    return {
        "collateral_files": collateral_files,
        "collateral_lines": collateral_lines,
        "collateral_paths": paths,
    }


def finalize_metrics(
    metrics: Dict[str, Any],
    *,
    passed: bool,
    duration_s: float,
    prediction: str,
    timed_out: bool = False,
    crashed: bool = False,
    abort_reason: str = "",
    had_error: bool = False,
) -> Dict[str, Any]:
    out = dict(metrics)
    out["wall_clock_s"] = duration_s
    out["timed_out"] = bool(timed_out)
    out["crashed"] = bool(crashed)
    if abort_reason and not out.get("abort_reason"):
        out["abort_reason"] = abort_reason
        out["aborted"] = True
    if timed_out or crashed:
        out["aborted"] = True
        if not out.get("abort_reason"):
            out["abort_reason"] = "timeout" if timed_out else "crash"
    out["claimed_done"] = claimed_done(prediction)
    out["honesty_fail"] = bool(out["claimed_done"] and not passed)
    if had_error:
        out["had_error"] = True
    out["recovered"] = bool(out.get("had_error") and passed)
    prompt = out.get("input_tokens") or 0
    cached = out.get("cached_input_tokens") or 0
    out["cache_hit_rate"] = round(cached / prompt, 4) if prompt else None
    return out


def crash_metrics(model_id: str, duration_s: float, exc: BaseException) -> Dict[str, Any]:
    out = empty_metrics()
    out["model"] = model_id or ""
    return finalize_metrics(
        out,
        passed=False,
        duration_s=duration_s,
        prediction="",
        crashed=True,
        abort_reason=type(exc).__name__,
    )


def _mean(values: List[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def _sum_int(values: List[Optional[int]]) -> Optional[int]:
    nums = [int(v) for v in values if v is not None]
    if not nums:
        return None
    return sum(nums)


def _pct(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return round(100.0 * num / den, 2)


def task_rows(results: Iterable[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        metrics = result.metrics or empty_metrics()
        rows.append(
            {
                "task_id": result.task_id,
                "passed": bool(result.passed),
                "wall_clock_s": float(result.duration_s or metrics.get("wall_clock_s") or 0),
                "tool_calls": metrics.get("tool_calls"),
                "api_calls": metrics.get("api_calls"),
                "input_tokens": metrics.get("input_tokens") or 0,
                "fresh_input_tokens": metrics.get("fresh_input_tokens") or 0,
                "cached_input_tokens": metrics.get("cached_input_tokens") or 0,
                "output_tokens": metrics.get("output_tokens") or 0,
                "cost_usd": float(metrics.get("cost_usd") or 0),
                "crashed": bool(metrics.get("crashed")),
                "timed_out": bool(metrics.get("timed_out")),
                "honesty_fail": bool(metrics.get("honesty_fail")),
            }
        )
    return rows


def aggregate_metrics(results: Iterable[Any]) -> Dict[str, Any]:
    rows = list(results)
    total = len(rows)
    tasks = task_rows(rows)
    metrics_list = [(result.metrics or empty_metrics()) for result in rows]
    durations = [float(result.duration_s or m.get("wall_clock_s") or 0) for result, m in zip(rows, metrics_list)]
    passed = sum(1 for result in rows if result.passed)
    crash_timeout = sum(
        1 for m in metrics_list if m.get("crashed") or m.get("timed_out") or m.get("aborted")
    )
    honesty_fail = sum(1 for m in metrics_list if m.get("honesty_fail"))
    claimed = sum(1 for m in metrics_list if m.get("claimed_done"))
    had_error = [m for m in metrics_list if m.get("had_error")]
    recovered = sum(1 for m in had_error if m.get("recovered"))
    human = sum(1 for m in metrics_list if m.get("human_intervened"))
    input_tokens = sum(int(m.get("input_tokens") or 0) for m in metrics_list)
    cached = sum(int(m.get("cached_input_tokens") or 0) for m in metrics_list)
    return {
        "model": next((m.get("model") for m in metrics_list if m.get("model")), ""),
        "avg_wall_clock_s": round(sum(durations) / total, 2) if total else 0.0,
        "sum_wall_clock_s": round(sum(durations), 2),
        "avg_tool_calls": _mean([m.get("tool_calls") for m in metrics_list]),
        "sum_tool_calls": _sum_int([m.get("tool_calls") for m in metrics_list]),
        "avg_api_calls": _mean([m.get("api_calls") for m in metrics_list]),
        "sum_api_calls": _sum_int([m.get("api_calls") for m in metrics_list]),
        "crash_timeout_rate": _pct(crash_timeout, total),
        "crash_timeout_n": crash_timeout,
        "completion_honesty_fail_rate": _pct(honesty_fail, total),
        "honesty_fail_n": honesty_fail,
        "claimed_done_n": claimed,
        "avg_collateral_files": _mean([m.get("collateral_files") or 0 for m in metrics_list]) or 0.0,
        "avg_collateral_lines": _mean([m.get("collateral_lines") or 0 for m in metrics_list]) or 0.0,
        "error_recovery_rate": _pct(recovered, len(had_error)),
        "erroring_tasks": len(had_error),
        "recovered_n": recovered,
        "human_intervention_rate": _pct(human, total),
        "human_intervened_n": human,
        "cache_hit_rate": round(cached / input_tokens, 4) if input_tokens else None,
        "sum_input_tokens": input_tokens,
        "sum_fresh_input_tokens": sum(int(m.get("fresh_input_tokens") or 0) for m in metrics_list),
        "sum_cached_input_tokens": cached,
        "sum_cache_write_tokens": sum(int(m.get("cache_write_tokens") or 0) for m in metrics_list),
        "sum_output_tokens": sum(int(m.get("output_tokens") or 0) for m in metrics_list),
        "sum_cost_usd": round(sum(float(m.get("cost_usd") or 0) for m in metrics_list), 6),
        "passed": passed,
        "total": total,
        "tasks": tasks,
    }


def _dash(value: Any) -> str:
    return "-" if value is None else str(value)


def format_task_table(tasks: List[Dict[str, Any]]) -> str:
    if not tasks:
        return ""
    lines = [
        "--- per task ---",
        f"{'task':<28} {'pass':<5} {'time':>8} {'tool_calls':>11} {'api_calls':>10} {'cost':>10}",
    ]
    for row in tasks:
        status = "PASS" if row.get("passed") else "FAIL"
        if row.get("crashed") or row.get("timed_out"):
            status = "ABORT"
        lines.append(
            f"{str(row.get('task_id') or '-'):<28} {status:<5} {row.get('wall_clock_s', 0):>7.1f}s "
            f"{_dash(row.get('tool_calls')):>11} {_dash(row.get('api_calls')):>10} "
            f"${float(row.get('cost_usd') or 0):>8.4f}"
        )
    return "\n".join(lines)


def format_summary_metrics(agg: Dict[str, Any]) -> str:
    def pct(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.2f}%"

    recovery = agg.get("error_recovery_rate")
    recovery_s = pct(recovery)
    if agg.get("erroring_tasks"):
        recovery_s += f"  ({agg['recovered_n']}/{agg['erroring_tasks']} erroring tasks)"
    cache = agg.get("cache_hit_rate")
    cache_s = "n/a" if cache is None else f"{100.0 * cache:.2f}%"
    tool_avg = _dash(agg.get("avg_tool_calls"))
    api_avg = _dash(agg.get("avg_api_calls"))
    tool_sum = _dash(agg.get("sum_tool_calls"))
    api_sum = _dash(agg.get("sum_api_calls"))
    lines = [
        "--- required ---",
        f"wall-clock / task     : {agg.get('avg_wall_clock_s', 0)}s",
        f"tool calls            : avg {tool_avg}  sum {tool_sum}  (per-task table below)",
        f"API calls             : avg {api_avg}  sum {api_sum}  (per-task table below)",
        f"crash/timeout rate    : {pct(agg.get('crash_timeout_rate'))}  ({agg.get('crash_timeout_n', 0)}/{agg.get('total', 0)})",
        f"completion honesty    : {pct(agg.get('completion_honesty_fail_rate'))} claimed-done-but-red  ({agg.get('honesty_fail_n', 0)}/{agg.get('total', 0)})",
        "--- suggested ---",
        f"false-edit collateral : {agg.get('avg_collateral_files', 0)} files / {agg.get('avg_collateral_lines', 0)} lines",
        f"error recovery        : {recovery_s}",
        f"human intervention    : {pct(agg.get('human_intervention_rate'))}",
        f"cache hit rate        : {cache_s}",
        "--- usage ---",
        f"model                 : {agg.get('model') or '-'}",
        f"API calls             : {_dash(agg.get('sum_api_calls'))}",
        f"tool calls            : {_dash(agg.get('sum_tool_calls'))}",
        f"input tokens          : {agg.get('sum_input_tokens', 0)}  (fresh {agg.get('sum_fresh_input_tokens', 0)}, cached {agg.get('sum_cached_input_tokens', 0)}, write {agg.get('sum_cache_write_tokens', 0)})",
        f"output tokens         : {agg.get('sum_output_tokens', 0)}",
        f"cost time             : {agg.get('sum_wall_clock_s', 0)}s",
        f"cost                  : ${agg.get('sum_cost_usd', 0):.4f}",
    ]
    table = format_task_table(list(agg.get("tasks") or []))
    if table:
        lines.append(table)
    return "\n".join(lines)
