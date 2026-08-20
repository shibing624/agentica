# -*- coding: utf-8 -*-
"""InfiAgent-DABench (DAEval validation) — data-analysis agent loop. No Docker.

Closed-form tags `@name[value]` are scored against official labels (exact
match, or float within 1e-6). Same DeepAgent / Codex CLI surface as Polyglot
(no todos; schema drops ls/glob/grep). Prompt names the CSV, forbids extra
cleaning, and stops as soon as the required tags are written.
"""
from __future__ import annotations

import asyncio
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from .cli_agents import run_cli_agent
from .common import (
    SampleResult,
    build_coding_agent,
    cache_path,
    download_file,
    isolated_home,
    load_jsonl,
    slice_items,
)
from .metrics import (
    add_usage,
    empty_metrics,
    finalize_metrics,
    inspect_run_response,
)

RAW = "https://raw.githubusercontent.com/InfiAgent/InfiAgent/main/examples/DA-Agent/data"
TAG_RE = re.compile(r"@(\w+)\[(.*?)\]")

PROMPT = """The CSV is already named {file_name} in the current directory. Do not list, glob, or grep the directory.

Question:
{question}

Constraints:
{constraints}

Compute with execute using pandas, numpy, scipy, or sklearn as the constraints require. Do not pip install. Prefer a short Python snippet over reading the whole CSV into context.

Follow the constraints exactly. Do not drop rows, recode columns, or skip values unless the constraints say so.

When you have the numbers, the last message must contain only the required tags (no table, no interpretation) in this exact format:
{format}

Stop immediately after that message.
"""


def data_dir() -> Path:
    return cache_path("dabench")


def extract_tags(text: str) -> Dict[str, str]:
    return dict(TAG_RE.findall(text or ""))


def is_equal(response: Any, label: Any) -> bool:
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except (TypeError, ValueError):
        return False


def score_response(text: str, common_answers: List[List[str]]) -> Tuple[bool, Dict[str, Any]]:
    """Official DAEval closed-form judge: every @name must match its label."""
    extracted = extract_tags(text)
    expected = {name: value for name, value in common_answers}
    correctness = {name: is_equal(extracted.get(name), value) for name, value in expected.items()}
    ok = bool(correctness) and all(correctness.values())
    return ok, {"expected": expected, "predicted": extracted, "correctness": correctness}


def build_prompt(question: dict) -> str:
    return PROMPT.format(
        file_name=question["file_name"],
        question=question["question"],
        constraints=question.get("constraints") or "",
        format=question["format"],
    )


def ensure_dabench_files() -> Tuple[Path, Path, Path]:
    root = data_dir()
    questions = download_file(f"{RAW}/da-dev-questions.jsonl", root / "da-dev-questions.jsonl")
    labels = download_file(f"{RAW}/da-dev-labels.jsonl", root / "da-dev-labels.jsonl")
    return questions, labels, root / "da-dev-tables"


def load_questions(*, max_samples: int, offset: int) -> Tuple[List[dict], Dict[Any, dict]]:
    qpath, lpath, tables = ensure_dabench_files()
    questions = slice_items(load_jsonl(qpath), max_samples, offset)
    labels = {row["id"]: row for row in load_jsonl(lpath)}
    for question in questions:
        name = question["file_name"]
        download_file(f"{RAW}/da-dev-tables/{quote(name)}", tables / name)
    return questions, labels


async def run_one(
    question: dict,
    gold: dict,
    *,
    model,
    work: Path,
    home: Path,
    tool_call_limit: int,
    agent_kind: str,
    agent_timeout: int,
    cli_env: Optional[dict],
    cli_model: str,
) -> SampleResult:
    tables = data_dir() / "da-dev-tables"
    work.mkdir(parents=True, exist_ok=True)
    src = tables / question["file_name"]
    shutil.copy2(src, work / question["file_name"])
    prompt = build_prompt(question)
    started = time.time()
    prediction = ""
    crashed = False
    timed_out = False
    abort_reason = ""
    last_error = ""
    metrics = empty_metrics() if agent_kind == "agentica" else None
    model_id = (model.id if model is not None else "") or agent_kind

    if agent_kind == "agentica":
        agent = build_coding_agent(model, work, home, tool_call_limit=tool_call_limit)
        try:
            response = await agent.run(prompt)
            prediction = response.content or ""
            metrics = add_usage(metrics, inspect_run_response(response))
            if response.break_reason:
                abort_reason = str(response.break_reason)
        except Exception as exc:
            crashed = True
            abort_reason = type(exc).__name__
            last_error = str(exc)[-2000:]
    else:
        text, ext, crashed_now, abort = await asyncio.to_thread(
            run_cli_agent,
            agent_kind,
            work,
            prompt,
            agent_timeout,
            model=cli_model,
            env=cli_env,
        )
        prediction = text or ""
        metrics = ext
        if (ext.get("abort_reason") or "").endswith("_not_found"):
            crashed = True
            abort_reason = abort
            last_error = text
        elif ext.get("timed_out"):
            timed_out = True
            abort_reason = abort
        elif crashed_now:
            crashed = True
            abort_reason = abort

    if metrics is None:
        metrics = empty_metrics()
    answers = list((gold or {}).get("common_answers") or [])
    passed, detail = score_response(prediction, answers)
    duration_s = round(time.time() - started, 2)
    metrics["model"] = metrics.get("model") or model_id
    metrics = finalize_metrics(
        metrics,
        passed=passed,
        duration_s=duration_s,
        prediction=prediction,
        timed_out=timed_out,
        crashed=crashed,
        abort_reason=abort_reason,
        had_error=False,
    )
    if not passed and not last_error:
        last_error = json_mismatch(detail)
    return SampleResult(
        bench="dabench",
        task_id=str(question["id"]),
        passed=passed,
        duration_s=duration_s,
        error=last_error[-2000:],
        prediction=prediction,
        extra={
            "file_name": question.get("file_name"),
            "level": question.get("level"),
            "concepts": question.get("concepts") or [],
            "agent": agent_kind,
            **detail,
        },
        metrics=metrics,
    )


def json_mismatch(detail: Dict[str, Any]) -> str:
    return (
        f"expected={detail.get('expected')} predicted={detail.get('predicted')} "
        f"correctness={detail.get('correctness')}"
    )


async def run_dabench(
    *,
    model,
    max_samples: int,
    offset: int,
    tool_call_limit: int,
    output_dir: Path,
    agent_kind: str = "agentica",
    agent_timeout: int = 300,
    cli_env: Optional[dict] = None,
    cli_model: str = "",
) -> List[SampleResult]:
    questions, labels = load_questions(max_samples=max_samples, offset=offset)
    if not questions:
        raise RuntimeError("no DABench questions after slice")
    results: List[SampleResult] = []
    with isolated_home(output_dir / "home") as home:
        for question in questions:
            qid = question["id"]
            print(f"[dabench] id={qid} level={question.get('level')} file={question['file_name']}")
            result = await run_one(
                question,
                labels.get(qid) or {},
                model=model,
                work=output_dir / "workdir" / f"q{qid}",
                home=home,
                tool_call_limit=tool_call_limit,
                agent_kind=agent_kind,
                agent_timeout=agent_timeout,
                cli_env=cli_env,
                cli_model=cli_model,
            )
            status = "PASS" if result.passed else "FAIL"
            m = result.metrics or {}
            print(
                f"  {status}  {result.duration_s}s  "
                f"tools={'-' if m.get('tool_calls') is None else m.get('tool_calls')}  "
                f"api={'-' if m.get('api_calls') is None else m.get('api_calls')}  "
                f"in={m.get('input_tokens', 0)}  out={m.get('output_tokens', 0)}"
            )
            results.append(result)
    return results


def dry_run_dabench() -> dict:
    """No LLM, no download: gold tags pass, a wrong number fails."""
    gold = "@mean_fare[34.65] extra prose"
    labels = [["mean_fare", "34.65"]]
    ok, detail = score_response(gold, labels)
    bad, bad_detail = score_response("@mean_fare[0]", labels)
    empty, _ = score_response("no tags", labels)
    return {
        "gold_passed": ok,
        "broken_failed": not bad,
        "empty_failed": not empty,
        "gold_predicted": detail["predicted"],
        "broken_predicted": bad_detail["predicted"],
    }
