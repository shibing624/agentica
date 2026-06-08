# -*- coding: utf-8 -*-
"""Tests for P1/P2 VaG extensions: extended seeds, ablation summary,
case studies, TB2 / InterCode / NexAU adapters, cross-model sensitivity."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.vag.adapters import harbor_terminal_bench, intercode_bash, terminal_bench
from evaluation.vag.adapters.nexau import adapter as nexau_adapter
from evaluation.vag.analysis.make_paper_tables import render_cross_model_table
from evaluation.vag.analysis.summarize_admission import summarize
from evaluation.vag.runners import (
    case_studies,
    cross_model,
    downstream_stress,
    holdout_replay,
    regression_injection,
    tb2_harbor_plan,
)
from evaluation.vag.seeds import build_extended_candidates, build_pilot_candidates


def test_extended_set_has_200_candidates_with_ground_truth():
    candidates = build_extended_candidates()
    assert len(candidates) == 200
    benign = [c for c in candidates if c.label == "benign"]
    harmful = [c for c in candidates if c.label == "harmful"]
    assert len(benign) == 120
    assert len(harmful) == 80
    assert all(c.expected_rejected_by for c in harmful), "harmful candidates need expected critic labels"
    assert all(not c.expected_rejected_by for c in benign), "benign candidates must not expect rejection"


def test_pilot_set_unchanged():
    candidates = build_pilot_candidates()
    assert len(candidates) == 60
    assert len([c for c in candidates if c.label == "benign"]) == 36
    assert len([c for c in candidates if c.label == "harmful"]) == 24


def test_regression_injection_has_full_gate_list_and_separate_files():
    gates = regression_injection.gate_configs()
    for name in [
        "ungated",
        "llm_only",
        "skillsvote_style",
        "peek_style",
        "schema",
        "exec",
        "agent",
        "schema_exec",
        "schema_agent",
        "exec_agent",
        "vag_full",
    ]:
        assert name in gates

    with tempfile.TemporaryDirectory() as td:
        candidates_path = Path(td) / "pilot_candidates.jsonl"
        labels_path = Path(td) / "pilot_labels.jsonl"
        source = build_pilot_candidates()
        regression_injection.write_dataset(source, candidates_path, labels_path)
        candidate_row = json.loads(candidates_path.read_text(encoding="utf-8").splitlines()[0])
        label_row = json.loads(labels_path.read_text(encoding="utf-8").splitlines()[0])
        assert "label" not in candidate_row
        assert "failure_type" not in candidate_row
        assert label_row["label"] in {"benign", "harmful"}
        loaded = regression_injection.load_dataset(candidates_path, labels_path)
        assert loaded == source


def test_summarize_admission_renders_table():
    fake_results = [
        {
            "config": "ungated",
            "harmful_admission_rate": 1.0,
            "benign_retention_rate": 1.0,
            "accepted": [
                {"id": "h1", "label": "harmful", "failure_type": "format"},
                {"id": "b1", "label": "benign", "failure_type": "none"},
            ],
            "rejected": [],
        },
        {
            "config": "vag_full",
            "harmful_admission_rate": 0.0,
            "benign_retention_rate": 1.0,
            "accepted": [{"id": "b1", "label": "benign", "failure_type": "none"}],
            "rejected": [{"id": "h1", "label": "harmful", "failure_type": "format"}],
        },
    ]
    table = summarize(fake_results)
    assert "Config" in table
    assert "vag_full" in table
    assert "format" in table


def test_case_studies_writes_provenance_and_report():
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        studies = asyncio.run(case_studies.run_cases(out_dir))
        provenance = (out_dir / "provenance.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(provenance) == len(case_studies.CASE_IDS)
        for line in provenance:
            event = json.loads(line)
            assert event["event"] == "gate_decision"
            assert "verdicts" in event

        report = case_studies.render_report(studies, out_dir)
        assert report.startswith("# VaG Case Studies")
        for case_id in case_studies.CASE_IDS:
            assert case_id in report

        benign_event = next(e for e in studies.values() if e["label"] == "benign")
        assert benign_event["approved"] is True
        harmful_events = [e for e in studies.values() if e["label"] == "harmful"]
        assert any(not e["approved"] for e in harmful_events)


def test_terminal_bench_adapter_runs_synthetic_split():
    split = terminal_bench.load_split(
        ROOT / "evaluation" / "vag" / "data" / "tb2_synthetic_split.jsonl"
    )
    assert len(split) >= 5
    outcomes = asyncio.run(terminal_bench.run_split(split))
    assert len(outcomes) == len(split)
    assert all("approved" in o for o in outcomes)


def test_intercode_bash_adapter_runs_synthetic_split():
    split = intercode_bash.load_split(
        ROOT / "evaluation" / "vag" / "data" / "intercode_bash_synthetic.jsonl"
    )
    assert len(split) >= 3
    outcomes = asyncio.run(intercode_bash.run_split(split))
    assert len(outcomes) == len(split)


def test_nexau_adapter_runs_synthetic_split():
    split = nexau_adapter.load_split(
        ROOT / "evaluation" / "vag" / "data" / "nexau_synthetic.jsonl"
    )
    assert split, "NexAU synthetic split is empty"
    outcomes = asyncio.run(nexau_adapter.run_split(split))
    assert all(o["approved"] for o in outcomes), "benign nexau stubs should pass vag_full"


def test_downstream_stress_reports_bad_skill_failures():
    rows = asyncio.run(downstream_stress.run(["ungated", "vag_full"]))
    by_gate = {row["gate"]: row for row in rows}
    assert by_gate["ungated"]["bad_skill_induced_failures"] > 0
    assert by_gate["vag_full"]["bad_skill_induced_failures"] == 0
    table = downstream_stress.render_table(rows)
    assert "Bad-skill-induced failures" in table
    assert "vag_full" in table


def test_cross_model_table_renders_rows():
    table = render_cross_model_table([
        {
            "model": "judge-strong",
            "miss_rate": 0.0,
            "harmful_admission_rate": 0.0,
            "benign_retention_rate": 1.0,
        }
    ])
    assert "judge-strong" in table
    assert "Harmful admitted" in table


def test_harbor_tb2_split_and_command_plan():
    task_ids = [f"task-{idx:02d}" for idx in range(12)]
    split = harbor_terminal_bench.split_task_ids(task_ids, seed=42, event_size=6, holdout_size=3, test_size=3)
    all_split_ids = set(split.event + split.holdout + split.test)
    assert all_split_ids.issubset(set(task_ids))
    assert not (set(split.event) & set(split.holdout))
    assert not (set(split.holdout) & set(split.test))

    oracle = harbor_terminal_bench.build_oracle_smoke_plan()
    assert oracle.command[:3] == ["harbor", "run", "-d"]
    agent = harbor_terminal_bench.build_agent_run_plan(env="daytona", n_tasks=5)
    assert "--env" in agent.command
    assert "daytona" in agent.command


def test_tb2_harbor_plan_builds_without_dataset_checkout():
    class Args:
        dataset = harbor_terminal_bench.DEFAULT_DATASET
        dataset_root = ""
        split_out_dir = ""
        seed = 42
        event_size = 8
        holdout_size = 4
        test_size = 4
        agent = "claude-code"
        model = "anthropic/claude-haiku-4-5"
        env = "daytona"
        n_tasks = 4
        extra_arg = []

    with tempfile.TemporaryDirectory() as td:
        args = Args()
        args.split_out_dir = td
        plan = tb2_harbor_plan.build_plan(args)
        assert plan["n_discovered_tasks"] >= 16
        assert Path(plan["split_files"]["manifest"]).exists()
        assert plan["commands"][0]["command"][0] == "harbor"


def test_holdout_replay_synthetic_rejects_exec_sensitive_harmful():
    candidates = build_extended_candidates()
    records = holdout_replay.build_synthetic_records(candidates)
    decisions = holdout_replay.evaluate_replay(records)
    by_id = {row["candidate_id"]: row for row in decisions}
    assert by_id["harmful_000_01"]["approved"] is False
    assert by_id["harmful_000_06"]["approved"] is False
    assert by_id["benign_000_00"]["approved"] is True


def test_cross_model_sensitivity_orders_by_miss_rate():
    rows = asyncio.run(cross_model.run())
    assert [r["model"] for r in rows] == ["judge-strong", "judge-medium", "judge-weak"]
    strong = next(r for r in rows if r["model"] == "judge-strong")
    weak = next(r for r in rows if r["model"] == "judge-weak")
    assert strong["harmful_admission_rate"] <= weak["harmful_admission_rate"], (
        f"weak judge should admit at least as many harmful candidates as strong: "
        f"strong={strong['harmful_admission_rate']} weak={weak['harmful_admission_rate']}"
    )
