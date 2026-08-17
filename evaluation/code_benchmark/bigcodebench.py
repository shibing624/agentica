# -*- coding: utf-8 -*-
"""BigCodeBench — function-level generate with real-library calls. Local execute."""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .common import SampleResult, extract_code, generate_text, slice_items
from .execute import run_python_source
from .metrics import crash_metrics, finalize_metrics

GENERATE_PROMPT = """Write a Python solution for the following programming task.
Use the libraries mentioned. Return only code, no explanation.

{prompt}
"""


def load_bigcodebench(subset: str = "full") -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "BigCodeBench needs the optional `datasets` package: pip install datasets"
        ) from exc
    name = "bigcode/bigcodebench" if subset == "full" else f"bigcode/bigcodebench-{subset}"
    try:
        dataset = load_dataset(name, split="v0.1.4")
    except Exception:
        dataset = load_dataset(name, split="train")
    return [dict(row) for row in dataset]


def assemble_solution(row: Dict[str, Any], completion: str, split: str) -> str:
    text = extract_code(completion)
    prefix = row.get("complete_prompt") or ""
    if split == "complete" and prefix:
        head = prefix.strip()[:48]
        if head and head in text:
            return text
        return prefix + text
    return text


def judge_bcb(code: str, test: str, timeout: int = 30) -> tuple[bool, str]:
    source = f"""
import unittest
import sys

code = {code!r}
test = {test!r}
ns = {{}}
exec(code, ns)
exec(test, ns)
loader = unittest.defaultTestLoader
suite = unittest.TestSuite()
for value in list(ns.values()):
    if isinstance(value, type) and issubclass(value, unittest.TestCase) and value is not unittest.TestCase:
        suite.addTests(loader.loadTestsFromTestCase(value))
if suite.countTestCases() == 0 and "check" in ns:
    entry = None
    for key, value in ns.items():
        if callable(value) and key not in ("check",):
            entry = value
    if entry is not None:
        ns["check"](entry)
        print("PASSED")
        sys.exit(0)
result = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(suite)
if result.failures or result.errors or suite.countTestCases() == 0:
    sys.exit(1)
print("PASSED")
"""
    judged = run_python_source(source, timeout=timeout)
    return judged.ok, judged.output


async def run_bigcodebench(
    *,
    model,
    max_samples: int,
    offset: int,
    subset: str,
    split: str,
    timeout: int,
) -> List[SampleResult]:
    rows = load_bigcodebench(subset)
    rows = slice_items(rows, max_samples, offset)
    results: List[SampleResult] = []
    for row in rows:
        task_id = str(row.get("task_id") or "unknown")
        prompt = row.get("instruct_prompt") if split == "instruct" else row.get("complete_prompt")
        print(f"[bigcodebench] {task_id}")
        started = time.time()
        try:
            raw, usage = await generate_text(model, GENERATE_PROMPT.format(prompt=prompt or ""))
        except Exception as exc:
            duration_s = round(time.time() - started, 2)
            results.append(
                SampleResult(
                    bench="bigcodebench",
                    task_id=task_id,
                    passed=False,
                    duration_s=duration_s,
                    error=str(exc)[-2000:],
                    prediction="",
                    extra={"subset": subset, "split": split},
                    metrics=crash_metrics(model.id or "", duration_s, exc),
                )
            )
            print(f"  CRASH  {duration_s}s")
            continue
        code = assemble_solution(row, raw, split)
        ok, output = judge_bcb(code, row.get("test") or "", timeout=timeout)
        duration_s = round(time.time() - started, 2)
        metrics = finalize_metrics(
            usage,
            passed=ok,
            duration_s=duration_s,
            prediction=raw,
            timed_out=(not ok) and "timeout" in (output or "").lower(),
        )
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {duration_s}s")
        results.append(
            SampleResult(
                bench="bigcodebench",
                task_id=task_id,
                passed=ok,
                duration_s=duration_s,
                error="" if ok else output[-2000:],
                prediction=code,
                extra={"subset": subset, "split": split},
                metrics=metrics,
            )
        )
    return results


def dry_run_bigcodebench() -> dict:
    rows = slice_items(load_bigcodebench("full"), 1)
    if not rows:
        raise RuntimeError("BigCodeBench download is empty")
    row = rows[0]
    gold = (row.get("complete_prompt") or "") + (row.get("canonical_solution") or "")
    good_ok, good_out = judge_bcb(gold, row.get("test") or "", timeout=25)
    bad_ok, bad_out = judge_bcb("def broken():\n    return None\n", row.get("test") or "", timeout=15)
    return {
        "task_id": row.get("task_id"),
        "canonical_passed": good_ok,
        "broken_failed": not bad_ok,
        "gold_output": good_out[-300:],
        "broken_output": bad_out[-300:],
    }
