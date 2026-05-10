# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for generated skill provenance JSONL logs.
"""
import tempfile
import unittest
from pathlib import Path

from evaluation.vag.lifecycle import (
    PROVENANCE_FILENAME,
    append_provenance_event,
    get_provenance_path,
    read_provenance_events,
)


class TestSkillProvenance(unittest.TestCase):
    def test_first_write_creates_provenance_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "generated-skills" / "safe-skill"
            append_provenance_event(skill_dir, {
                "event": "admission",
                "skill_name": "safe-skill",
                "fingerprint": "sha256:first",
                "approved": True,
                "verdicts": [],
            })

            path = get_provenance_path(skill_dir)
            self.assertEqual(path.name, PROVENANCE_FILENAME)
            self.assertTrue(path.exists())

    def test_append_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "safe-skill"
            append_provenance_event(skill_dir, {
                "event": "admission",
                "skill_name": "safe-skill",
                "fingerprint": "sha256:first",
                "approved": True,
                "verdicts": [],
            })
            append_provenance_event(skill_dir, {
                "event": "promotion",
                "skill_name": "safe-skill",
                "from": "shadow",
                "to": "auto",
                "approved": True,
                "verdicts": [],
            })
            append_provenance_event(skill_dir, {
                "event": "rollback",
                "skill_name": "safe-skill",
                "reason": "consecutive_failures",
                "verdicts": [],
            })

            events = read_provenance_events(skill_dir)

            self.assertEqual([e["event"] for e in events], ["admission", "promotion", "rollback"])
            self.assertEqual(events[0]["fingerprint"], "sha256:first")
            self.assertIn("created_at", events[0])

    def test_read_skips_malformed_partial_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "safe-skill"
            append_provenance_event(skill_dir, {
                "event": "admission",
                "skill_name": "safe-skill",
                "approved": True,
            })
            path = get_provenance_path(skill_dir)
            with path.open("a", encoding="utf-8") as f:
                f.write('{"event": "partial"')

            events = read_provenance_events(skill_dir)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event"], "admission")

    def test_fingerprint_versions_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "safe-skill"
            append_provenance_event(skill_dir, {
                "event": "admission",
                "skill_name": "safe-skill",
                "fingerprint": "sha256:first",
                "approved": True,
            })
            append_provenance_event(skill_dir, {
                "event": "admission",
                "skill_name": "safe-skill",
                "fingerprint": "sha256:second",
                "approved": True,
            })

            events = read_provenance_events(skill_dir)

            self.assertEqual(
                [e["fingerprint"] for e in events],
                ["sha256:first", "sha256:second"],
            )


if __name__ == "__main__":
    unittest.main()
