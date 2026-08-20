# -*- coding: utf-8 -*-
"""Unit tests for DABench scoring / prompt / dry-run judge (no LLM)."""
import pytest

from evaluation.code_benchmark.common import cache_path, load_jsonl
from evaluation.code_benchmark.dabench import (
    build_prompt,
    dry_run_dabench,
    extract_tags,
    score_response,
)


def test_extract_and_score_closed_form():
    text = "prose @mean_fare[34.65] more"
    assert extract_tags(text) == {"mean_fare": "34.65"}
    ok, detail = score_response(text, [["mean_fare", "34.65"]])
    assert ok
    assert detail["predicted"]["mean_fare"] == "34.65"


def test_score_float_tolerance():
    ok, _ = score_response("@r[0.2100000001]", [["r", "0.21"]])
    assert ok


def test_score_rejects_wrong_and_missing():
    labels = [["mean_fare", "34.65"], ["n", "3"]]
    bad, detail = score_response("@mean_fare[0]", labels)
    assert not bad
    assert detail["correctness"]["mean_fare"] is False
    assert detail["correctness"]["n"] is False
    empty, _ = score_response("I am done", labels)
    assert not empty


def test_prompt_requires_tags():
    prompt = build_prompt(
        {
            "file_name": "a.csv",
            "question": "mean?",
            "constraints": "round 2",
            "format": "@mean_fare[x]",
        }
    )
    assert "a.csv" in prompt
    assert "@mean_fare[x]" in prompt
    assert "pandas" in prompt
    assert "Do not list, glob, or grep the directory" in prompt
    assert "Stop immediately" in prompt
    assert "Do not drop rows" in prompt


def test_cached_questions_each_name_one_csv():
    """Dropping ls/glob/grep is only valid if every DAEval row names one CSV."""
    path = cache_path("dabench") / "da-dev-questions.jsonl"
    if not path.exists():
        pytest.skip("DABench questions cache not downloaded")
    questions = load_jsonl(path)
    assert questions
    names = [q.get("file_name") for q in questions]
    assert all(isinstance(n, str) and n.lower().endswith(".csv") for n in names)
    assert all("/" not in n and "\\" not in n for n in names)


def test_dry_run_judge_no_network():
    payload = dry_run_dabench()
    assert payload["gold_passed"]
    assert payload["broken_failed"]
    assert payload["empty_failed"]
