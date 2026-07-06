# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the SkillCurator linting/catalog tool.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.skills.curator import SkillCurator, STATUS_OK, STATUS_WARNING, STATUS_BROKEN


def _write_skill(root: Path, dirname: str, frontmatter: str, body: str = "# Body\n") -> Path:
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
    return d


class _CuratorOnTmp(SkillCurator):
    """Curator that only scans a single explicit skills dir (test isolation)."""
    def __init__(self, skills_dir: Path):
        super().__init__()
        self._skills_dir = skills_dir

    def scan(self):
        # Override search paths: scan ONLY the temp dir.
        from agentica.skills.skill import Skill
        discovered = []
        for md in self.loader.discover_skills(self._skills_dir):
            discovered.append((Skill.from_skill_md(md, "project"), md, "project"))
        reports = [self._validate_one(s, p, l) for s, p, l in discovered]
        self._flag_duplicates(reports, discovered)
        for r in reports:
            self._finalize_status(r)
        return reports


class TestSkillCurator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _curator(self):
        return _CuratorOnTmp(self.root)

    def test_clean_skill_is_ok(self):
        _write_skill(self.root, "good",
                     "name: Good Skill\ndescription: Does good\nwhen_to_use: foo, bar\ntrigger: /good")
        reports = self._curator().scan()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].status, STATUS_OK)

    def test_missing_when_to_use_warns(self):
        _write_skill(self.root, "nowtu", "name: No WTU\ndescription: x")
        report = self._curator().scan()[0]
        self.assertEqual(report.status, STATUS_WARNING)
        self.assertTrue(any(i.code == "no_when_to_use" for i in report.issues))

    def test_bad_trigger_warns(self):
        _write_skill(self.root, "badtrig",
                     "name: Bad Trigger\ndescription: x\nwhen_to_use: a\ntrigger: nofslash")
        report = self._curator().scan()[0]
        self.assertTrue(any(i.code == "bad_trigger" for i in report.issues))

    def test_unparseable_is_broken(self):
        # Missing description -> from_skill_md returns None -> broken
        _write_skill(self.root, "broken", "name: OnlyName")
        report = self._curator().scan()[0]
        self.assertEqual(report.status, STATUS_BROKEN)
        self.assertTrue(any(i.code == "parse_failed" for i in report.issues))

    def test_duplicate_name_is_error(self):
        _write_skill(self.root, "a", "name: Dup\ndescription: x\nwhen_to_use: a")
        _write_skill(self.root, "b", "name: Dup\ndescription: y\nwhen_to_use: b")
        reports = self._curator().scan()
        self.assertTrue(all(r.status == STATUS_BROKEN for r in reports))
        self.assertTrue(all(any(i.code == "duplicate_name" for i in r.issues) for r in reports))

    def test_missing_resource_dir_warns(self):
        _write_skill(self.root, "res",
                     "name: Res\ndescription: x\nwhen_to_use: a",
                     body="Run scripts/build.sh to do the thing.")
        report = self._curator().scan()[0]
        self.assertTrue(any(i.code == "missing_resource_dir" for i in report.issues))

    def test_present_resource_dir_no_warn(self):
        d = _write_skill(self.root, "res2",
                         "name: Res2\ndescription: x\nwhen_to_use: a",
                         body="Run scripts/build.sh.")
        (d / "scripts").mkdir()
        report = self._curator().scan()[0]
        self.assertFalse(any(i.code == "missing_resource_dir" for i in report.issues))

    def test_catalog_generation(self):
        _write_skill(self.root, "good",
                     "name: Cat Skill\ndescription: d\nwhen_to_use: a\ntrigger: /cat")
        catalog = self._curator().generate_catalog()
        self.assertIn("# Skill Catalog", catalog)
        self.assertIn("Cat Skill", catalog)
        self.assertIn("/cat", catalog)

    def test_summary_runs(self):
        _write_skill(self.root, "good", "name: S\ndescription: d\nwhen_to_use: a")
        out = self._curator().summary()
        self.assertIn("Scanned 1 skill", out)


class TestPublicExport(unittest.TestCase):
    def test_importable(self):
        from agentica.skills import SkillCurator, SkillReport, SkillIssue
        self.assertIsNotNone(SkillCurator)


if __name__ == "__main__":
    unittest.main()
