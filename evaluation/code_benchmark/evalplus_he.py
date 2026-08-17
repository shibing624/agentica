# -*- coding: utf-8 -*-
"""EvalPlus HumanEval+ — function-level generate + local execute. Smoke / baseline only."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    SampleResult,
    cache_path,
    download_file,
    extract_code,
    generate_text,
    slice_items,
)
from .execute import judge_check_function, judge_plus_inputs
from .metrics import crash_metrics, finalize_metrics

HUMANEVAL_PLUS_URL = (
    "https://github.com/evalplus/humanevalplus/releases/download/"
    "v0.1.10/HumanEvalPlus-v0.1.10.jsonl.gz"
)
HUMANEVAL_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

GENERATE_PROMPT = """Complete the following Python function. Return only the full function implementation, no explanation.

{prompt}
"""


def _read_jsonl_gz(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".jsonl.gz") else open
    rows: List[Dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_humaneval_plus() -> List[Dict[str, Any]]:
    dest = cache_path("evalplus", "HumanEvalPlus-v0.1.10.jsonl.gz")
    try:
        download_file(HUMANEVAL_PLUS_URL, dest, timeout=120)
        return _read_jsonl_gz(dest)
    except Exception:
        fallback = cache_path("evalplus", "HumanEval.jsonl.gz")
        download_file(HUMANEVAL_URL, fallback, timeout=60)
        return _read_jsonl_gz(fallback)


def assemble_code(prompt: str, completion: str) -> str:
    text = extract_code(completion)
    head = prompt.strip()[:48]
    if head and head in text:
        return text
    return prompt + text


async def run_evalplus(
    *,
    model,
    max_samples: int,
    offset: int,
    plus: bool,
    timeout: int,
) -> List[SampleResult]:
    rows = slice_items(load_humaneval_plus(), max_samples, offset)
    results: List[SampleResult] = []
    for row in rows:
        task_id = str(row.get("task_id") or "unknown")
        prompt = row.get("prompt") or ""
        print(f"[evalplus] {task_id}")
        started = time.time()
        try:
            raw, usage = await generate_text(model, GENERATE_PROMPT.format(prompt=prompt))
        except Exception as exc:
            duration_s = round(time.time() - started, 2)
            results.append(
                SampleResult(
                    bench="evalplus",
                    task_id=task_id,
                    passed=False,
                    duration_s=duration_s,
                    error=str(exc)[-2000:],
                    prediction="",
                    extra={"base_ok": False, "plus_ok": False},
                    metrics=crash_metrics(model.id or "", duration_s, exc),
                )
            )
            print(f"  CRASH  {duration_s}s")
            continue
        code = assemble_code(prompt, raw)
        base = judge_check_function(
            code,
            row.get("test") or "",
            row.get("entry_point") or "",
            timeout=timeout,
        )
        plus_ok = True
        plus_output = ""
        if plus and base.ok and row.get("plus_input") and row.get("canonical_solution"):
            plus_result = judge_plus_inputs(
                prompt,
                code[len(prompt):] if code.startswith(prompt) else extract_code(raw),
                row["canonical_solution"],
                row.get("entry_point") or "",
                row["plus_input"],
                timeout=timeout,
            )
            plus_ok = plus_result.ok
            plus_output = plus_result.output
        passed = base.ok and plus_ok
        duration_s = round(time.time() - started, 2)
        metrics = finalize_metrics(
            usage,
            passed=passed,
            duration_s=duration_s,
            prediction=raw,
            timed_out=bool(base.timed_out),
        )
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {duration_s}s")
        results.append(
            SampleResult(
                bench="evalplus",
                task_id=task_id,
                passed=passed,
                duration_s=duration_s,
                error="" if passed else (plus_output or base.output)[-2000:],
                prediction=code,
                extra={"base_ok": base.ok, "plus_ok": plus_ok},
                metrics=metrics,
            )
        )
    return results


def dry_run_evalplus() -> dict:
    rows = slice_items(load_humaneval_plus(), 1)
    if not rows:
        raise RuntimeError("HumanEval(+) download is empty")
    row = rows[0]
    gold = (row.get("prompt") or "") + (row.get("canonical_solution") or "")
    good = judge_check_function(gold, row.get("test") or "", row.get("entry_point") or "", timeout=15)
    bad = judge_check_function(
        (row.get("prompt") or "") + "    return None\n",
        row.get("test") or "",
        row.get("entry_point") or "",
        timeout=15,
    )
    return {
        "task_id": row.get("task_id"),
        "canonical_passed": good.ok,
        "broken_failed": not bad.ok,
        "gold_output": good.output[-300:],
        "broken_output": bad.output[-300:],
    }
