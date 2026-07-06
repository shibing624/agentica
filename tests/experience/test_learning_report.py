# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for LearningReport (arch_v5.md Phase 2 visibility).
"""
import json
import tempfile
from pathlib import Path

import pytest

from agentica.learning_report import (
    LearningReport,
    LearningStatus,
    write_learning_report,
)
from agentica.workspace import Workspace


class TestLearningReportShape:
    def test_default_status(self):
        r = LearningReport(run_id="r1")
        assert r.status == LearningStatus.no_action
        assert r.tool_errors_captured == 0
        assert r.cards_written == 0

    def test_mark_learned_clears_skip_reason(self):
        r = LearningReport(run_id="r1")
        r.mark_skipped("nothing")
        assert r.skip_reason == "nothing"
        r.mark_learned()
        assert r.status == LearningStatus.learned
        assert r.skip_reason is None

    def test_to_dict_serializes_status(self):
        r = LearningReport(run_id="r1", session_id="s1")
        r.mark_skipped("below_threshold")
        d = r.to_dict()
        assert d["status"] == "skipped"
        assert d["skip_reason"] == "below_threshold"
        assert d["session_id"] == "s1"

    def test_markdown_contains_counters(self):
        r = LearningReport(
            run_id="abc",
            tool_errors_captured=2,
            cards_written=1,
            skill_candidate="my_skill",
        )
        r.mark_learned()
        md = r.to_markdown()
        assert "# Learning Report — abc" in md
        assert "tool_errors_captured: 2" in md
        assert "cards_written: 1" in md
        assert "skill_candidate: my_skill" in md
        assert "Status:** `learned`" in md


class TestLearningReportPersistence:
    def test_write_returns_none_when_workspace_none(self):
        r = LearningReport(run_id="r1")
        assert write_learning_report(None, r) is None

    def test_write_persists_markdown_and_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(td)
            ws.initialize()
            r = LearningReport(run_id="run-xyz", session_id="s1")
            r.mark_learned()
            r.cards_written = 3

            path = write_learning_report(ws, r)
            assert path is not None
            md_path = Path(path)
            assert md_path.exists()
            assert "cards_written: 3" in md_path.read_text(encoding="utf-8")

            jsonl = md_path.parent / "learning.jsonl"
            assert jsonl.exists()
            line = jsonl.read_text(encoding="utf-8").strip().splitlines()[-1]
            payload = json.loads(line)
            assert payload["run_id"] == "run-xyz"
            assert payload["status"] == "learned"
            assert payload["cards_written"] == 3
