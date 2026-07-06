# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for the workspace memory evidence gate (arch_v5.md Phase 2).

Verifies:
- Trusted sources land in the canonical memory folder + MEMORY.md index
- Untrusted sources land in memory_candidates/ and DO NOT touch the index
- evidence_refs are persisted in the frontmatter for both paths
"""
import asyncio
import tempfile
from pathlib import Path

import pytest

from agentica.workspace import Workspace


def _run(coro):
    """Run a coroutine in a fresh event loop.

    Using `asyncio.get_event_loop()` here is unsafe inside the full pytest run
    because earlier async tests can leave the loop in a closed state. A fresh
    loop per call keeps these workspace tests independent.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEvidenceGate:
    def test_verified_source_writes_to_memory_dir(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(td)
            ws.initialize()
            path = _run(ws.write_memory_entry(
                title="proj_alpha",
                content="Alpha project ships on Friday",
                memory_type="project",
                source="verified",
                evidence_refs=["docs/spec.md", "issue#42"],
            ))
            p = Path(path)
            assert p.exists()
            assert "memory_candidates" not in str(p)
            text = p.read_text(encoding="utf-8")
            assert "source: verified" in text
            # JSON-encoded so YAML-special chars (`#`) don't get interpreted
            # as inline comments. The line must still parse as YAML and JSON.
            assert 'evidence_refs: ["docs/spec.md", "issue#42"]' in text
            import yaml
            front = text.split("---", 2)[1]
            parsed = yaml.safe_load(front)
            assert parsed["evidence_refs"] == ["docs/spec.md", "issue#42"]

            index = ws._get_user_memory_md()
            assert index.exists()
            assert "proj_alpha" in index.read_text(encoding="utf-8")

    def test_auto_extract_source_lands_in_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(td)
            ws.initialize()
            path = _run(ws.write_memory_entry(
                title="maybe_pref",
                content="User MIGHT like dark mode",
                memory_type="user",
                source="auto_extract",
            ))
            p = Path(path)
            assert "memory_candidates" in str(p)
            text = p.read_text(encoding="utf-8")
            assert "source: auto_extract" in text

            # Index must NOT mention quarantined entries
            index = ws._get_user_memory_md()
            if index.exists():
                assert "maybe_pref" not in index.read_text(encoding="utf-8")

    def test_manual_source_treated_as_trusted(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(td)
            ws.initialize()
            path = _run(ws.write_memory_entry(
                title="manual_note",
                content="hand written",
                memory_type="reference",
                source="manual",
            ))
            assert "memory_candidates" not in path

    def test_reports_and_candidates_dirs_lazy_create(self):
        """`reports/` and `memory_candidates/` are the only Phase 2 partitions.

        An earlier draft also created a sibling `archives/` directory; that
        was removed because RunJournal / SessionArchive are now planned to
        land under `reports/runs/` and `reports/sessions/` respectively
        (single root for structured run-time artifacts).
        """
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(td)
            ws.initialize()
            r = ws.get_user_learning_reports_dir()
            c = ws._get_user_memory_candidates_dir()
            assert r.exists() and r.is_dir()
            assert c.exists() and c.is_dir()
            # archives/ partition was deliberately removed from WorkspaceConfig
            assert not hasattr(ws.config, "archives_dir")
            assert not (Path(td) / "users" / "default" / "archives").exists()
