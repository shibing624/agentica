# -*- coding: utf-8 -*-
"""LiveCodeBench code-generation (lite) — generate + local execute. No Docker."""
from __future__ import annotations

import base64
import json
import pickle
import time
import zlib
from typing import Any, Dict, List, Optional

from .common import SampleResult, extract_code, generate_text, iter_jsonl_url, slice_items
from .execute import run_python_source
from .metrics import crash_metrics, finalize_metrics

# release_v1 is a single jsonl; enough for a first score. Newer releases need
# extra shards (test2.jsonl …) or `datasets.load_dataset`.
LCB_TEST_JSONL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
    "resolve/main/test.jsonl"
)

GENERATE_PROMPT = """You are a competitive programmer. Solve the problem in Python 3.

Write a complete solution. If a function signature or class Solution is given, keep it.
Do not explain. Output only code.

# Problem
{question}

{starter}
"""


def decode_test_cases(raw: Any) -> List[Dict[str, Any]]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    text = raw if isinstance(raw, str) else raw.decode("utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass
    blob = pickle.loads(zlib.decompress(base64.b64decode(text.encode("utf-8"))))
    if isinstance(blob, (bytes, str)):
        blob = json.loads(blob)
    if isinstance(blob, list):
        return blob
    return [blob]


def load_lcb_problems(
    release: str = "release_v1",
    *,
    limit: int = 0,
    offset: int = 0,
    start_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load LCB rows. The official jsonl is ~1.2GB — stream and stop at limit."""
    if release not in {"release_v1", "v1"}:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "release_v2+ needs the optional `datasets` package, "
                "or pass --release release_v1 to stream the first shard"
            ) from exc
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag=release,
            split="test",
            trust_remote_code=True,
        )
        rows = [dict(row) for row in dataset]
        if start_date:
            rows = [r for r in rows if str(r.get("contest_date") or "") >= start_date]
        return slice_items(rows, limit, offset)

    print("[lcb] streaming first matching rows from the lite jsonl (not downloading 1.2GB)")
    collected: List[Dict[str, Any]] = []
    skipped = 0
    for row in iter_jsonl_url(LCB_TEST_JSONL, limit=0, timeout=60):
        if start_date and str(row.get("contest_date") or "") < start_date:
            continue
        if skipped < offset:
            skipped += 1
            continue
        collected.append(row)
        if limit and len(collected) >= limit:
            break
    return collected


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    raw = row.get("metadata") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _all_tests(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    public = decode_test_cases(row.get("public_test_cases"))
    private = decode_test_cases(row.get("private_test_cases"))
    return public + private


def judge_lcb(code: str, row: Dict[str, Any], timeout: int = 12) -> tuple[bool, str]:
    tests = _all_tests(row)
    if not tests:
        return False, "no test cases"
    meta = _metadata(row)
    fn_name = meta.get("func_name") or meta.get("fn_name")
    payload = json.dumps(tests)
    source = f"""
import json
import io
import sys
from types import ModuleType

code = {code!r}
tests = json.loads({payload!r})
fn_name = {fn_name!r}

ns = {{}}
exec(code, ns)
obj = ns.get("Solution")
if obj is not None:
    inst = obj()
else:
    inst = None

def call_fn(args):
    if inst is not None and fn_name:
        return getattr(inst, fn_name)(*args)
    if fn_name:
        return ns[fn_name](*args)
    raise RuntimeError("no function name in metadata")

def run_stdin(stdin_text):
    buf = io.StringIO()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(stdin_text)
    sys.stdout = buf
    try:
        if "__main_guard_removed__" not in ns:
            pass
        main = ns.get("main")
        if callable(main):
            main()
        elif inst is not None and fn_name:
            getattr(inst, fn_name)()
        else:
            # re-exec so top-level stdin reads run again
            local = dict(ns)
            exec(code, local)
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    return buf.getvalue()

for i, case in enumerate(tests):
    raw_in = case.get("input", "")
    expected = case.get("output", "")
    kind = (case.get("testtype") or ("functional" if fn_name else "stdin")).lower()
    if kind in ("functional", "call", "function"):
        args = [json.loads(line) for line in str(raw_in).splitlines() if line != ""]
        got = call_fn(args)
        exp = json.loads(expected) if isinstance(expected, str) else expected
        if isinstance(got, tuple):
            got = list(got)
        if got != exp:
            raise AssertionError(f"case {{i}}: {{got!r}} != {{exp!r}}")
    else:
        got = run_stdin(str(raw_in))
        if got.strip() != str(expected).strip():
            raise AssertionError(f"stdin case {{i}}: {{got!r}} != {{expected!r}}")
print("PASSED")
"""
    result = run_python_source(source, timeout=timeout)
    return result.ok, result.output


def _starter_block(row: Dict[str, Any]) -> str:
    starter = (row.get("starter_code") or "").strip()
    if not starter:
        return ""
    return f"# Starter code\n```python\n{starter}\n```"


async def run_livecodebench(
    *,
    model,
    max_samples: int,
    offset: int,
    release: str,
    start_date: Optional[str],
    timeout: int,
) -> List[SampleResult]:
    rows = load_lcb_problems(
        release,
        limit=max_samples,
        offset=offset,
        start_date=start_date,
    )
    results: List[SampleResult] = []
    for row in rows:
        task_id = str(row.get("question_id") or row.get("question_title") or "unknown")
        print(f"[lcb] {task_id}")
        prompt = GENERATE_PROMPT.format(
            question=row.get("question_content") or "",
            starter=_starter_block(row),
        )
        started = time.time()
        crashed = False
        try:
            raw, usage = await generate_text(model, prompt)
        except Exception as exc:
            duration_s = round(time.time() - started, 2)
            results.append(
                SampleResult(
                    bench="livecodebench",
                    task_id=task_id,
                    passed=False,
                    duration_s=duration_s,
                    error=str(exc)[-2000:],
                    prediction="",
                    extra={
                        "difficulty": row.get("difficulty"),
                        "platform": row.get("platform"),
                        "contest_date": row.get("contest_date"),
                    },
                    metrics=crash_metrics(model.id or "", duration_s, exc),
                )
            )
            print(f"  CRASH  {duration_s}s")
            continue
        code = extract_code(raw)
        if row.get("starter_code") and row["starter_code"].strip() not in code:
            code = (row.get("starter_code") or "") + "\n" + code
        ok, output = judge_lcb(code, row, timeout=timeout)
        duration_s = round(time.time() - started, 2)
        timed_out = (not ok) and "timeout" in (output or "").lower()
        metrics = finalize_metrics(
            usage,
            passed=ok,
            duration_s=duration_s,
            prediction=raw,
            timed_out=timed_out,
            crashed=crashed,
        )
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {duration_s}s")
        results.append(
            SampleResult(
                bench="livecodebench",
                task_id=task_id,
                passed=ok,
                duration_s=duration_s,
                error="" if ok else output[-2000:],
                prediction=code,
                extra={
                    "difficulty": row.get("difficulty"),
                    "platform": row.get("platform"),
                    "contest_date": row.get("contest_date"),
                },
                metrics=metrics,
            )
        )
    return results


def dry_run_lcb() -> dict:
    rows = load_lcb_problems("release_v1", limit=1)
    if not rows:
        raise RuntimeError("LiveCodeBench shard is empty")
    row = rows[0]
    # A broken snippet must fail; we cannot assume a canonical solution is present.
    ok, output = judge_lcb("def impossible():\n    return None\n", row, timeout=8)
    return {
        "task_id": row.get("question_id"),
        "n_tests": len(_all_tests(row)),
        "broken_failed": not ok,
        "output": output[-400:],
    }
