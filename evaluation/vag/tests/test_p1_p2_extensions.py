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

from evaluation.vag.adapters import intercode_bash, terminal_bench
from evaluation.vag.adapters.nexau import adapter as nexau_adapter
from evaluation.vag.analysis.summarize_admission import summarize
from evaluation.vag.runners import case_studies, cross_model
from evaluation.vag.seeds import build_extended_candidates, build_pilot_candidates


def test_extended_set_has_200_candidates_with_ground_truth():
    candidates = build_extended_candidates()
    assert len(candidates) == 200
    benign = [c for c in candidates if c.label == "benign"]
    harmful = [c for c in candidates if c.label == "harmful"]
    assert len(benign) == 20
    assert len(harmful) == 180
    assert all(c.expected_rejected_by for c in harmful), "harmful candidates need expected critic labels"
    assert all(not c.expected_rejected_by for c in benign), "benign candidates must not expect rejection"


def test_pilot_set_unchanged():
    assert len(build_pilot_candidates()) == 60


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


def test_cross_model_sensitivity_orders_by_miss_rate():
    rows = asyncio.run(cross_model.run())
    assert [r["model"] for r in rows] == ["judge-strong", "judge-medium", "judge-weak"]
    strong = next(r for r in rows if r["model"] == "judge-strong")
    weak = next(r for r in rows if r["model"] == "judge-weak")
    assert strong["harmful_admission_rate"] <= weak["harmful_admission_rate"], (
        f"weak judge should admit at least as many harmful candidates as strong: "
        f"strong={strong['harmful_admission_rate']} weak={weak['harmful_admission_rate']}"
    )
