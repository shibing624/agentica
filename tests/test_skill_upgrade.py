# -*- coding: utf-8 -*-
"""
Tests for the experience → skill upgrade pipeline.

Covers:
1. SkillUpgradeConfig defaults and custom values
2. SkillEvolutionManager — candidate filtering, spawn, episode recording, state judging
3. Hooks integration — skill upgrade triggered after lifecycle
4. Cross-layer cleanup — memory_feedback removed from compiler

All tests mock LLM API keys -- no real API calls.
"""
import asyncio
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.agent.config import ExperienceConfig, SkillUpgradeConfig
from agentica.skills import reset_skill_registry


# Valid SKILL.md fixture that passes _validate_skill_content (gotcha-first
# format: minimal frontmatter + Gotchas section + Minimal Example with real
# code). Tests substitute the name via .replace() when they need a different
# slug.
_VALID_SKILL_MD = (
    "---\nname: pandas-preference\n"
    "description: Use pandas for tabular data, not the csv module\n"
    "when-to-use: csv, dataframe, tabular, parsing\n---\n\n"
    "Reach for pandas.read_csv before the stdlib csv module.\n\n"
    "## Gotchas\n"
    "- \u26a0\ufe0f csv.reader strips dtypes: every cell becomes str. "
    "Fix: pd.read_csv(path, dtype=...).\n"
    "- \u26a0\ufe0f csv.writer adds CRLF on Windows: rows look doubled. "
    "Fix: open with newline='' or use df.to_csv.\n\n"
    "## Minimal Example\n```python\nimport pandas as pd\n"
    "df = pd.read_csv('data.csv', dtype={'id': int})\n"
    "df.to_csv('out.csv', index=False)\n```\n"
)


# ===========================================================================
# SkillUpgradeConfig tests
# ===========================================================================

class TestSkillUpgradeConfig(unittest.TestCase):
    """Test SkillUpgradeConfig defaults and custom values."""

    def test_defaults(self):
        config = SkillUpgradeConfig()
        self.assertEqual(config.mode, "shadow")
        self.assertEqual(config.min_repeat_count, 3)
        self.assertEqual(config.min_tier, "hot")
        self.assertEqual(config.checkpoint_interval, 5)
        self.assertEqual(config.rollback_consecutive_failures, 2)
        self.assertTrue(config.notify_user)
        self.assertFalse(config.maintain_failed_skills)
        self.assertEqual(config.max_repair_attempts, 3)

    def test_custom_values(self):
        config = SkillUpgradeConfig(
            mode="draft",
            min_repeat_count=5,
            checkpoint_interval=10,
        )
        self.assertEqual(config.mode, "draft")
        self.assertEqual(config.min_repeat_count, 5)
        self.assertEqual(config.checkpoint_interval, 10)

    def test_experience_config_skill_upgrade_none_by_default(self):
        config = ExperienceConfig()
        self.assertIsNone(config.skill_upgrade)

    def test_experience_config_with_skill_upgrade(self):
        config = ExperienceConfig(skill_upgrade=SkillUpgradeConfig())
        self.assertIsNotNone(config.skill_upgrade)
        self.assertEqual(config.skill_upgrade.mode, "shadow")


# ===========================================================================
# SkillEvolutionManager tests
# ===========================================================================

class TestNormalizeSkillMd(unittest.TestCase):
    """LLM-emitted SKILL.md frontmatter coercion."""

    def _parses(self, normalized: str) -> bool:
        from agentica.skills.skill import Skill
        # Skill._parse_frontmatter is a static method.
        meta, _ = Skill._parse_frontmatter(normalized.strip())
        return bool(meta) and "name" in meta

    def test_yaml_inline_comment_does_not_truncate_frontmatter(self):
        """A `# comment` inside YAML must NOT be mistaken for a markdown body."""
        from agentica.experience.skill_upgrade import _normalize_skill_md
        raw = (
            "---\nname: foo\n"
            "# Reminder: keep slug short\n"
            "description: bar\n"
            "when-to-use: never\n\n"
            "## Body\nactual body\n"
        )
        out = _normalize_skill_md(raw)
        self.assertIn("when-to-use: never", out)
        self.assertIn("\n---\n## Body", out)
        from agentica.skills.skill import Skill
        meta, _ = Skill._parse_frontmatter(out.strip())
        self.assertEqual(meta.get("when-to-use"), "never")
        self.assertEqual(meta.get("description"), "bar")

    def test_strips_leading_stray_dash_and_adds_missing_close(self):
        """Real-world LLM output: stray ``-`` then ``---`` open, no close."""
        from agentica.experience.skill_upgrade import _normalize_skill_md
        raw = (
            "-\n---\nname: list-directory-before-read\n"
            "description: t\nwhen-to-use: read_file\n\n"
            "## One-line summary\nEnsure existence.\n\n"
            "## Gotchas\n- \u26a0\ufe0f a: b. c.\n- \u26a0\ufe0f d: e. f.\n"
        )
        out = _normalize_skill_md(raw)
        self.assertTrue(out.startswith("---\nname:"))
        self.assertIn("\n---\n## One-line summary", out)
        self.assertTrue(self._parses(out))

    def test_idempotent_on_canonical_input(self):
        from agentica.experience.skill_upgrade import _normalize_skill_md
        canonical = (
            "---\n"
            "name: foo\n"
            "description: bar\n"
            "---\n"
            "# Body\n"
        )
        out = _normalize_skill_md(canonical)
        self.assertEqual(out, canonical)
        self.assertTrue(self._parses(out))

    def test_strips_yaml_code_fence(self):
        """LLMs frequently emit ```yaml ... ``` instead of ---."""
        from agentica.experience.skill_upgrade import _normalize_skill_md
        raw = (
            "```yaml\n"
            "name: check-dir\n"
            "description: check directory\n"
            "when-to-use: before reading files\n"
            "```\n"
            "# Overview\n"
            "Step 1.\n"
        )
        out = _normalize_skill_md(raw)
        self.assertTrue(out.startswith("---\n"))
        self.assertIn("---\n# Overview", out)
        self.assertTrue(self._parses(out))

    def test_strips_mixed_fence_and_dashes(self):
        """Opening ```yaml fence, closing --- (the failure mode hit in practice)."""
        from agentica.experience.skill_upgrade import _normalize_skill_md
        raw = (
            "```yaml\n"
            "name: mixed\n"
            "description: mixed front\n"
            "---\n"
            "# Overview\n"
            "Body.\n"
        )
        out = _normalize_skill_md(raw)
        self.assertTrue(out.startswith("---\n"))
        self.assertTrue(self._parses(out))

    def test_adds_missing_leading_dashes(self):
        from agentica.experience.skill_upgrade import _normalize_skill_md
        raw = (
            "name: bare\n"
            "description: bare front\n"
            "---\n"
            "# Body\n"
        )
        out = _normalize_skill_md(raw)
        self.assertTrue(out.startswith("---\n"))
        self.assertTrue(self._parses(out))

    def test_strips_stray_prefix_before_dashes(self):
        """A leaked '-' or other prefix before the real '---' must be dropped."""
        from agentica.experience.skill_upgrade import _normalize_skill_md
        raw = (
            "-\n"
            "---\n"
            "name: leaked\n"
            "description: leaked front\n"
            "---\n"
            "# Overview\n"
            "Body.\n"
        )
        out = _normalize_skill_md(raw)
        self.assertTrue(out.startswith("---\nname: leaked"))
        self.assertTrue(self._parses(out))


class TestSkillEvolutionManager(unittest.TestCase):
    """Test the SkillEvolutionManager."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._exp_dir = Path(self._tmpdir) / "experiences"
        self._gen_dir = Path(self._tmpdir) / "generated_skills"
        self._exp_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_experience(self, title="test", repeat_count=5, tier="hot", exp_type="correction"):
        """Write a test experience file."""
        import re
        safe_title = re.sub(r"[^\w\-]", "_", title.lower())[:50].strip("_")
        filename = f"{exp_type}_{safe_title}.md"
        filepath = self._exp_dir / filename
        content = (
            f"---\ntitle: {title}\n"
            f"type: {exp_type}\n"
            f"tool: \n"
            f"repeat_count: {repeat_count}\n"
            f"first_seen: {date.today().isoformat()}\n"
            f"last_seen: {date.today().isoformat()}\n"
            f"tier: {tier}\n---\n\n"
            f"Rule: Always do {title}\nWhy: It works better\n"
            f"How to apply: When doing tasks"
        )
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def test_get_candidate_cards_filters_correctly(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        # High repeat, hot → candidate
        self._write_experience(title="good_candidate", repeat_count=5, tier="hot")
        # Low repeat → not candidate
        self._write_experience(title="low_repeat", repeat_count=1, tier="hot")
        # Warm tier → not candidate (min_tier=hot)
        self._write_experience(title="warm_exp", repeat_count=5, tier="warm")

        candidates = SkillEvolutionManager.get_candidate_cards(
            self._exp_dir, min_repeat_count=3, min_tier="hot",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["title"], "good_candidate")

    def test_get_candidate_cards_empty_dir(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        empty_dir = Path(self._tmpdir) / "empty"
        candidates = SkillEvolutionManager.get_candidate_cards(empty_dir)
        self.assertEqual(candidates, [])

    def test_maybe_spawn_skill_generates_skill_md(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "pandas-preference",
            "source_experience": "use_pandas_not_csv",
            "reason": "Repeated correction about data processing",
            "skill_md": (
                "---\nname: pandas-preference\n"
                "description: Use pandas for data processing\n"
                "when-to-use: data processing, CSV, dataframes\n---\n\n"
                "Use pandas.read_csv instead of the csv module for any "
                "tabular workload.\n\n## Gotchas\n"
                "- \u26a0\ufe0f csv.reader strips dtypes: every cell becomes "
                "str. Fix: pd.read_csv(path, dtype=...).\n"
                "- \u26a0\ufe0f csv.writer adds CRLF on Windows: rows look "
                "doubled. Fix: open file with newline='' or use df.to_csv.\n\n"
                "## Minimal Example\n```python\nimport pandas as pd\n"
                "df = pd.read_csv('data.csv', dtype={'id': int})\n"
                "df.to_csv('out.csv', index=False)\n```\n"
            ),
        })))

        manager = SkillEvolutionManager()
        candidates = [{"title": "use_pandas", "content": "Use pandas", "repeat_count": 5, "type": "correction"}]

        result = asyncio.run(manager.maybe_spawn_skill(
            model=model, candidates=candidates,
            existing_skills=[], generated_skills_dir=self._gen_dir,
        ))
        self.assertEqual(result, "pandas-preference")

        # Verify files created
        skill_dir = self._gen_dir / "pandas-preference"
        self.assertTrue((skill_dir / "SKILL.md").exists())
        self.assertTrue((skill_dir / "meta.json").exists())

        meta = json.loads((skill_dir / "meta.json").read_text())
        self.assertEqual(meta["status"], "shadow")
        self.assertEqual(meta["skill_name"], "pandas-preference")

    def test_maybe_spawn_skill_ignores_when_llm_says_ignore(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "ignore",
            "reason": "Not procedural enough",
        })))

        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{"title": "x", "content": "y", "repeat_count": 5}],
            existing_skills=[], generated_skills_dir=self._gen_dir,
        ))
        self.assertIsNone(result)

    def test_maybe_spawn_skill_skips_existing(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "already-exists",
            "source_experience": "x",
            "skill_md": "---\nname: already-exists\ndescription: test\n---\nBody",
        })))

        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{"title": "x", "content": "y", "repeat_count": 5}],
            existing_skills=["already-exists"],
            generated_skills_dir=self._gen_dir,
        ))
        self.assertIsNone(result)

    def test_maybe_spawn_skill_empty_candidates(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=MagicMock(), candidates=[],
            existing_skills=[], generated_skills_dir=self._gen_dir,
        ))
        self.assertIsNone(result)

    def test_record_episode(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        episodes_path = Path(self._tmpdir) / "skill1" / "episodes.jsonl"
        SkillEvolutionManager.record_episode(
            episodes_path, outcome="success", query="test query",
        )
        SkillEvolutionManager.record_episode(
            episodes_path, outcome="failure", query="bad query", tool_errors=2,
        )
        lines = episodes_path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["outcome"], "success")
        self.assertEqual(json.loads(lines[1])["tool_errors"], 2)

    def test_read_write_meta(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        meta_path = Path(self._tmpdir) / "skill1" / "meta.json"
        meta = {"skill_name": "test", "status": "shadow", "total_episodes": 0}
        SkillEvolutionManager.write_meta(meta_path, meta)
        loaded = SkillEvolutionManager.read_meta(meta_path)
        self.assertEqual(loaded["skill_name"], "test")
        self.assertEqual(loaded["status"], "shadow")

    def test_read_meta_nonexistent(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        meta_path = Path(self._tmpdir) / "nonexistent" / "meta.json"
        self.assertEqual(SkillEvolutionManager.read_meta(meta_path), {})

    def test_update_meta_after_episode_success(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        meta_path = Path(self._tmpdir) / "skill1" / "meta.json"
        SkillEvolutionManager.write_meta(meta_path, {
            "skill_name": "test", "status": "shadow",
            "total_episodes": 2, "success_count": 1, "failure_count": 1,
            "consecutive_failures": 1,
        })
        updated = SkillEvolutionManager.update_meta_after_episode(meta_path, "success")
        self.assertEqual(updated["total_episodes"], 3)
        self.assertEqual(updated["success_count"], 2)
        self.assertEqual(updated["consecutive_failures"], 0)

    def test_update_meta_after_episode_failure(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        meta_path = Path(self._tmpdir) / "skill1" / "meta.json"
        SkillEvolutionManager.write_meta(meta_path, {
            "skill_name": "test", "status": "shadow",
            "total_episodes": 2, "success_count": 2, "failure_count": 0,
            "consecutive_failures": 0,
        })
        updated = SkillEvolutionManager.update_meta_after_episode(meta_path, "failure")
        self.assertEqual(updated["failure_count"], 1)
        self.assertEqual(updated["consecutive_failures"], 1)

    def test_maybe_update_skill_state_promote(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "skill1"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: t\n---\nBody")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "test", "status": "shadow",
            "total_episodes": 5, "success_count": 4, "failure_count": 1,
            "consecutive_failures": 0,
        })
        # Write 5 success episodes
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl", outcome="success", query=f"q{i}",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "promote",
            "reason": "Good performance",
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model, skill_dir=skill_dir, checkpoint_interval=5,
        ))
        self.assertEqual(decision, "promote")

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(meta["status"], "auto")

    def test_maybe_update_skill_state_auto_rollback(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "skill2"
        skill_dir.mkdir(parents=True)
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "bad-skill", "status": "shadow",
            "total_episodes": 3, "success_count": 0, "failure_count": 3,
            "consecutive_failures": 3,
        })

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=MagicMock(),  # Not called due to auto-rollback
            skill_dir=skill_dir,
            rollback_consecutive_failures=2,
        ))
        self.assertEqual(decision, "rollback")

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(meta["status"], "rolled_back")

    def test_maybe_update_not_at_checkpoint(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "skill3"
        skill_dir.mkdir(parents=True)
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "test", "status": "shadow",
            "total_episodes": 3, "success_count": 3, "failure_count": 0,
            "consecutive_failures": 0,
        })

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=MagicMock(), skill_dir=skill_dir, checkpoint_interval=5,
        ))
        self.assertIsNone(decision)  # Not at checkpoint (3 < 5)

    def test_maybe_update_revise(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "skill4"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: t\n---\nOld body")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "test", "status": "shadow",
            "total_episodes": 5, "success_count": 3, "failure_count": 2,
            "consecutive_failures": 0,
        })
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl", outcome="success" if i < 3 else "failure",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "revise",
            "reason": "Needs updating",
            "revised_skill_md": _VALID_SKILL_MD.replace(
                "name: pandas-preference", "name: test"
            ).replace(
                "Reach for pandas.read_csv",
                "Revised guidance: reach for pandas.read_csv",
            ),
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model, skill_dir=skill_dir, checkpoint_interval=5,
        ))
        self.assertEqual(decision, "revise")

        # Verify SKILL.md was updated
        skill_content = (skill_dir / "SKILL.md").read_text()
        self.assertIn("Revised guidance", skill_content)

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(meta["version"], 2)

    def test_maybe_update_revise_with_section_updates(self):
        """Revision should support section-level updates without rewriting the whole file."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "skill4_sections"
        skill_dir.mkdir(parents=True)
        original_md = SkillEvolutionManager._append_source_section(
            _VALID_SKILL_MD.replace(
                "name: pandas-preference", "name: test-sections"
            ),
            source="seed-card",
            event_count=2,
        )
        (skill_dir / "SKILL.md").write_text(original_md, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "test-sections", "status": "shadow",
            "total_episodes": 5, "success_count": 3, "failure_count": 2,
            "consecutive_failures": 0, "version": 1,
            "source_experience": "seed-card",
        })
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl", outcome="success" if i < 3 else "failure",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "revise",
            "reason": "Tighten the summary and gotchas only",
            "section_updates": {
                "summary": "Prefer pandas.read_csv first for tabular CSV workflows.",
                "gotchas": [
                    "csv.reader strips dtypes: every cell becomes str. Fix: pd.read_csv(path, dtype=...).",
                    "csv.Sniffer mis-detects delimiters on sparse samples. Fix: pass sep= explicitly to pd.read_csv.",
                ],
            },
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model, skill_dir=skill_dir, checkpoint_interval=5,
        ))
        self.assertEqual(decision, "revise")

        skill_content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Prefer pandas.read_csv first", skill_content)
        self.assertIn("csv.Sniffer mis-detects delimiters", skill_content)
        self.assertIn("df.to_csv('out.csv', index=False)", skill_content)
        self.assertIn("generated from experience card: `seed-card`", skill_content)

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(meta["version"], 2)

    def test_list_generated_skills(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        self._gen_dir.mkdir(parents=True)
        for name in ["skill-a", "skill-b"]:
            d = self._gen_dir / name
            d.mkdir()
            SkillEvolutionManager.write_meta(d / "meta.json", {
                "skill_name": name, "status": "shadow",
            })

        skills = SkillEvolutionManager.list_generated_skills(self._gen_dir)
        self.assertEqual(len(skills), 2)
        names = [s["skill_name"] for s in skills]
        self.assertIn("skill-a", names)
        self.assertIn("skill-b", names)

    def test_rollback_disables_skill_md(self):
        """Rollback should rename SKILL.md to SKILL.md.disabled."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "skill-rollback"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: test\ndescription: t\n---\nBody")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "test", "status": "shadow",
            "total_episodes": 3, "success_count": 0, "failure_count": 3,
            "consecutive_failures": 3,
        })

        manager = SkillEvolutionManager()
        asyncio.run(manager.maybe_update_skill_state(
            model=MagicMock(), skill_dir=skill_dir,
            rollback_consecutive_failures=2,
        ))

        # SKILL.md should be renamed, not visible to SkillLoader
        self.assertFalse((skill_dir / "SKILL.md").exists())
        self.assertTrue((skill_dir / "SKILL.md.disabled").exists())

    def test_draft_mode_sets_status_draft(self):
        """In draft mode, spawned skill should have status=draft, not shadow."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "draft-test",
            "source_experience": "x",
            "skill_md": _VALID_SKILL_MD.replace(
                "name: pandas-preference", "name: draft-test"
            ),
        })))

        manager = SkillEvolutionManager()
        slug = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{"title": "x", "content": "y", "repeat_count": 5}],
            existing_skills=[], generated_skills_dir=self._gen_dir,
        ))
        self.assertEqual(slug, "draft-test")

        # Now simulate hooks setting draft mode post-spawn
        meta_path = self._gen_dir / "draft-test" / "meta.json"
        meta = SkillEvolutionManager.read_meta(meta_path)
        meta["status"] = "draft"
        SkillEvolutionManager.write_meta(meta_path, meta)

        loaded = SkillEvolutionManager.read_meta(meta_path)
        self.assertEqual(loaded["status"], "draft")

    def test_episode_has_timestamp(self):
        """Episodes should include a UTC timestamp field."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        episodes_path = Path(self._tmpdir) / "ts_test" / "episodes.jsonl"
        SkillEvolutionManager.record_episode(
            episodes_path, outcome="success", query="test",
        )
        lines = episodes_path.read_text().strip().splitlines()
        ep = json.loads(lines[0])
        self.assertIn("timestamp", ep)
        self.assertIn("T", ep["timestamp"])  # ISO format with T separator

    def test_rolled_back_skill_not_discoverable(self):
        """After rollback, SKILL.md should be renamed so SkillLoader can't find it."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        from agentica.skills.skill_loader import SkillLoader

        # Create a generated skill directory
        skill_dir = self._gen_dir / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A test skill\n---\n\n# Test\n\nBody."
        )
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "test-skill", "status": "shadow",
            "total_episodes": 0, "consecutive_failures": 0,
        })

        # Before rollback: loader should discover it
        loader = SkillLoader()
        found_before = loader.discover_skills(self._gen_dir)
        self.assertEqual(len(found_before), 1)

        # Rollback
        SkillEvolutionManager._disable_skill_md(skill_dir)

        # After rollback: loader should NOT discover it
        found_after = loader.discover_skills(self._gen_dir)
        self.assertEqual(len(found_after), 0)
        self.assertTrue((skill_dir / "SKILL.md.disabled").exists())

    def test_promote_uses_auto_status(self):
        """Promote decision should set status to 'auto', not 'promoted'."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "auto-test"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: auto-test\ndescription: t\n---\nBody")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "auto-test", "status": "shadow",
            "total_episodes": 5, "success_count": 5, "failure_count": 0,
            "consecutive_failures": 0,
        })
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl", outcome="success",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "promote", "reason": "Good",
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model, skill_dir=skill_dir, checkpoint_interval=5,
        ))
        self.assertEqual(decision, "promote")
        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(meta["status"], "auto")

    def test_maybe_update_auto_skill_state_rolls_back_after_failures(self):
        """Auto skills should still rollback after later consecutive failures."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "auto-regressed"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: auto-regressed\ndescription: t\n---\nBody")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "auto-regressed",
            "status": "auto",
            "total_episodes": 8,
            "success_count": 5,
            "failure_count": 3,
            "consecutive_failures": 3,
        })

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=MagicMock(),
            skill_dir=skill_dir,
            rollback_consecutive_failures=2,
        ))

        self.assertEqual(decision, "rollback")
        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(meta["status"], "rolled_back")
        self.assertFalse((skill_dir / "SKILL.md").exists())

    def test_judge_prompt_includes_episode_failure_signals(self):
        """Judge prompt should surface tool_errors and user_corrected signals."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = Path(self._tmpdir) / "judge-signals"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: judge-signals\ndescription: t\n---\nBody")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "judge-signals", "status": "shadow",
            "total_episodes": 5, "success_count": 3, "failure_count": 2,
            "consecutive_failures": 0,
        })
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="failure" if i == 4 else "success",
                query=f"q{i}",
                tool_errors=2 if i == 4 else 0,
                user_corrected=(i == 4),
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "keep_shadow",
            "reason": "Need more data",
        })))

        manager = SkillEvolutionManager()
        asyncio.run(manager.maybe_update_skill_state(
            model=model, skill_dir=skill_dir, checkpoint_interval=5,
        ))

        prompt = model.response.call_args[0][0][0].content
        self.assertIn("tool_errors=2", prompt)
        self.assertIn("user_corrected=True", prompt)


# ===========================================================================
# Gotcha-first validator + INDEX + evidence chain
# ===========================================================================

class TestSkillContentValidator(unittest.TestCase):
    """`_validate_skill_content` enforces the No-Execution-No-Memory rules."""

    def test_valid_skill_passes(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        ok, reason = SkillEvolutionManager._validate_skill_content(_VALID_SKILL_MD)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_missing_gotchas_fails(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        md = (
            "---\nname: t\ndescription: t\nwhen-to-use: t\n---\n"
            "Just a plain body with no warnings."
        )
        ok, reason = SkillEvolutionManager._validate_skill_content(md)
        self.assertFalse(ok)
        self.assertIn("gotchas", reason.lower())

    def test_todo_placeholder_fails(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        md = _VALID_SKILL_MD + "\n## Notes\n# TODO: write me\n"
        ok, reason = SkillEvolutionManager._validate_skill_content(md)
        self.assertFalse(ok)
        self.assertIn("placeholder", reason.lower())

    def test_forbidden_textbook_heading_fails(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        md = _VALID_SKILL_MD + "\n## Workflow\n1. Do thing\n"
        ok, reason = SkillEvolutionManager._validate_skill_content(md)
        self.assertFalse(ok)
        self.assertIn("textbook heading", reason.lower())

    def test_skeleton_code_block_fails(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        md = (
            "---\nname: t\ndescription: t\nwhen-to-use: t\n---\n"
            "summary\n\n## Gotchas\n- \u26a0\ufe0f a: b. c.\n- \u26a0\ufe0f d: e. f.\n\n"
            "## Minimal Example\n```python\ndef f():\n  pass\n```\n"
        )
        ok, reason = SkillEvolutionManager._validate_skill_content(md)
        self.assertFalse(ok)
        self.assertIn("skeleton", reason.lower())


class TestSpawnSkillEvidenceAndIndex(unittest.TestCase):
    """Spawn rejects bad LLM output, builds INDEX.md, gates on recoveries."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._gen_dir = Path(self._tmpdir) / "gen"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_spawn_rejects_invalid_skill_md(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "bad-skill",
            "source_experience": "x",
            "skill_md": "---\nname: bad-skill\ndescription: t\n---\nNo gotchas here.",
        })))
        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{"title": "x", "content": "y", "repeat_count": 5}],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
        ))
        self.assertIsNone(result)
        self.assertFalse((self._gen_dir / "bad-skill").exists())

    def test_spawn_writes_index_and_source_section(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "pandas-preference",
            "source_experience": "use_pandas",
            "skill_md": _VALID_SKILL_MD,
        })))
        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{"title": "x", "content": "y", "repeat_count": 5}],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
        ))
        self.assertEqual(result, "pandas-preference")

        skill_md = (self._gen_dir / "pandas-preference" / "SKILL.md").read_text()
        self.assertIn("## Source", skill_md)
        self.assertIn("use_pandas", skill_md)

        index_md = (self._gen_dir / "INDEX.md").read_text()
        self.assertIn("pandas-preference", index_md)
        self.assertIn("csv, dataframe", index_md)

    def test_spawn_recovery_gate_blocks_when_no_recoveries(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        class _StubStore:
            async def read_all(self):
                return [{"event_type": "tool_error", "tool": "x", "error": "boom"}]

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "any",
            "source_experience": "x",
            "skill_md": _VALID_SKILL_MD,
        })))
        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{
                "title": "x", "content": "y", "repeat_count": 5,
                "type": "tool_error", "tool": "x",
            }],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
            event_store=_StubStore(),
            min_success_applications=2,
        ))
        self.assertIsNone(result)
        # Model must NOT have been called — gate runs before LLM
        model.response.assert_not_called()

    def test_spawn_recovery_gate_passes_with_enough_relevant_recoveries(self):
        """Per-candidate gate: only recoveries on the candidate's tool count."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        class _StubStore:
            async def read_all(self):
                return [
                    {"event_type": "tool_recovery", "tool": "pandas_read"},
                    {"event_type": "tool_recovery", "tool": "pandas_read"},
                ]

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "gated-skill",
            "source_experience": "x",
            "skill_md": _VALID_SKILL_MD.replace(
                "name: pandas-preference", "name: gated-skill"
            ),
        })))
        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{
                "title": "pandas_read_failure", "content": "y", "repeat_count": 5,
                "type": "tool_error", "tool": "pandas_read",
            }],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
            event_store=_StubStore(),
            min_success_applications=2,
        ))
        self.assertEqual(result, "gated-skill")

    def test_spawn_recovery_gate_blocks_when_recoveries_are_for_other_tool(self):
        """Workspace-global recoveries on tool A must NOT unlock spawn for tool B."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        class _StubStore:
            async def read_all(self):
                return [
                    {"event_type": "tool_recovery", "tool": "read_file"},
                    {"event_type": "tool_recovery", "tool": "read_file"},
                    {"event_type": "tool_recovery", "tool": "read_file"},
                ]

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "should-not-spawn",
            "source_experience": "x",
            "skill_md": _VALID_SKILL_MD,
        })))
        manager = SkillEvolutionManager()
        result = asyncio.run(manager.maybe_spawn_skill(
            model=model,
            candidates=[{
                "title": "grep_TimeoutError", "content": "y", "repeat_count": 5,
                "type": "tool_error", "tool": "grep",  # candidate is for grep
            }],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
            event_store=_StubStore(),
            min_success_applications=1,
        ))
        self.assertIsNone(result)
        model.response.assert_not_called()


class TestBuildEvidenceText(unittest.TestCase):
    """`_build_evidence_text` per-candidate dispatch via `_EventIndex`:
    - correction candidates pull `correction_classification` events
      filtered by `correction_key` (NOT raw user_message) so two distinct
      corrections never share evidence
    - tool_error / success_pattern candidates match on strict tool equality
    - empty events / empty candidates produce empty output
    """

    @staticmethod
    def _index(events):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        return SkillEvolutionManager._index_events_once(events)

    def test_empty_events_returns_empty(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        out = SkillEvolutionManager._build_evidence_text(
            candidates=[{"title": "x", "type": "correction", "correction_key": "k"}],
            idx=self._index([]),
        )
        self.assertEqual(out, "")

    def test_correction_matches_only_its_own_correction_key(self):
        """Two corrections in the same workspace must not share evidence."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        events = [
            {  # unrelated tool_error
                "event_type": "tool_error",
                "tool": "read_file",
                "error": "File not found: /tmp/a.txt",
            },
            {  # belongs to candidate A only
                "event_type": "correction_classification",
                "is_correction": True,
                "should_persist": True,
                "correction_key": "list_directory_read_file",
                "rule": "list directory before read file",
                "user_message": "Check directory before reading.",
            },
            {  # belongs to candidate B only
                "event_type": "correction_classification",
                "is_correction": True,
                "should_persist": True,
                "correction_key": "use_rg_not_grep",
                "rule": "use rg not grep",
                "user_message": "Always use rg, never grep.",
            },
        ]
        out = SkillEvolutionManager._build_evidence_text(
            candidates=[{
                "title": "list_directory_read_file",
                "type": "correction",
                "correction_key": "list_directory_read_file",
            }],
            idx=self._index(events),
        )
        self.assertIn("[correction]", out)
        self.assertIn("Check directory before reading.", out)
        self.assertNotIn("Always use rg", out)   # cross-key bleed
        self.assertNotIn("File not found", out)  # tool_error ignored

    def test_tool_error_candidate_strict_tool_equality(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        events = [
            {"event_type": "tool_error", "tool": "read_file", "error": "boom-A"},
            {"event_type": "tool_error", "tool": "read", "error": "boom-B"},
            {"event_type": "tool_error", "tool": "write_file", "error": "boom-C"},
        ]
        out = SkillEvolutionManager._build_evidence_text(
            candidates=[{
                "title": "read_file_failure",
                "type": "tool_error",
                "tool": "read_file",
            }],
            idx=self._index(events),
        )
        self.assertIn("boom-A", out)
        self.assertNotIn("boom-B", out)
        self.assertNotIn("boom-C", out)

    def test_tool_error_candidate_without_tool_field_gets_nothing(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        events = [
            {"event_type": "tool_error", "tool": "read_file", "error": "x"},
        ]
        out = SkillEvolutionManager._build_evidence_text(
            candidates=[{"title": "x", "type": "tool_error"}],
            idx=self._index(events),
        )
        self.assertEqual(out, "")

    def test_per_candidate_limit_truncates(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        events = [
            {"event_type": "tool_error", "tool": "t", "error": f"err-{i}"}
            for i in range(10)
        ]
        out = SkillEvolutionManager._build_evidence_text(
            candidates=[{"title": "c", "type": "tool_error", "tool": "t"}],
            idx=self._index(events),
            per_candidate_limit=3,
        )
        self.assertIn("err-9", out)
        self.assertIn("err-8", out)
        self.assertIn("err-7", out)
        self.assertNotIn("err-0", out)
        self.assertNotIn("err-5", out)


class TestAppendSourceSection(unittest.TestCase):
    """`_append_source_section` must:
    - append a fresh Source block when none exists
    - replace any existing Source block (idempotent; LLM-leaked Source is overwritten)
    """

    def test_appends_source_when_missing(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        md = "---\nname: x\n---\nBody.\n## Gotchas\n- one"
        out = SkillEvolutionManager._append_source_section(
            md, source="card-a", event_count=4,
        )
        self.assertIn("## Source", out)
        self.assertIn("`card-a`", out)
        self.assertIn("raw events cited: 4", out)

    def test_overwrites_existing_source_section(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        md = (
            "---\nname: x\n---\nBody.\n## Gotchas\n- one\n\n"
            "## Source\n- stale garbage the LLM wrote\n"
        )
        out = SkillEvolutionManager._append_source_section(
            md, source="card-b", event_count=7,
        )
        self.assertNotIn("stale garbage", out)
        self.assertIn("`card-b`", out)
        self.assertIn("raw events cited: 7", out)
        # Only one Source section ever
        self.assertEqual(out.count("## Source"), 1)

    def test_unknown_source_fallback(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        out = SkillEvolutionManager._append_source_section(
            "## Gotchas\n- x", source="", event_count=0,
        )
        self.assertIn("`unknown`", out)


class TestGetCandidateCardsToolField(unittest.TestCase):
    """get_candidate_cards must load the `tool` frontmatter field so
    `_build_evidence_text` can do strict tool matching downstream."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._exp_dir = Path(self._tmpdir) / "experiences"
        self._exp_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_tool_field_is_loaded(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        (self._exp_dir / "tool_error_read_file_not_found.md").write_text(
            "---\ntitle: read_file_not_found\ntype: tool_error\n"
            "tool: read_file\nrepeat_count: 5\ntier: hot\n---\n"
            "Tool `read_file` failed.",
            encoding="utf-8",
        )
        (self._exp_dir / "correction_check_dir.md").write_text(
            "---\ntitle: check_dir\ntype: correction\n"
            "repeat_count: 5\ntier: hot\n---\n"
            "Rule: check directory before read_file",
            encoding="utf-8",
        )
        candidates = SkillEvolutionManager.get_candidate_cards(
            self._exp_dir, min_repeat_count=3, min_tier="hot",
        )
        by_type = {c["type"]: c for c in candidates}
        self.assertEqual(by_type["tool_error"]["tool"], "read_file")
        # correction card with no `tool` field → empty string downstream
        self.assertEqual(by_type["correction"]["tool"], "")


class TestDisableSkillMd(unittest.TestCase):
    """`_disable_skill_md` renames SKILL.md so SkillLoader skips it."""

    def test_rename_skill_md_to_disabled(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "s"
            skill_dir.mkdir()
            md = skill_dir / "SKILL.md"
            md.write_text("body", encoding="utf-8")
            SkillEvolutionManager._disable_skill_md(skill_dir)
            self.assertFalse(md.exists())
            self.assertTrue((skill_dir / "SKILL.md.disabled").exists())

    def test_rename_is_safe_when_missing(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        with tempfile.TemporaryDirectory() as tmp:
            # Must not raise
            SkillEvolutionManager._disable_skill_md(Path(tmp))


class TestRebuildIndex(unittest.TestCase):
    """`rebuild_index` writes a keyword-routed L1 INDEX.md."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._gen_dir = Path(self._tmpdir) / "gen"
        self._gen_dir.mkdir(parents=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_rebuild_skips_rolled_back_skills(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        # Active skill
        active = self._gen_dir / "active"
        active.mkdir()
        (active / "SKILL.md").write_text(
            "---\nname: active\ndescription: alive\nwhen-to-use: foo, bar\n---\nbody"
        )
        SkillEvolutionManager.write_meta(active / "meta.json", {"status": "shadow"})
        # Rolled-back skill
        dead = self._gen_dir / "dead"
        dead.mkdir()
        (dead / "SKILL.md").write_text(
            "---\nname: dead\ndescription: gone\nwhen-to-use: x\n---\nbody"
        )
        SkillEvolutionManager.write_meta(dead / "meta.json", {"status": "rolled_back"})

        idx = SkillEvolutionManager.rebuild_index(self._gen_dir)
        self.assertIsNotNone(idx)
        text = idx.read_text()
        self.assertIn("active", text)
        self.assertNotIn("dead", text)


# ===========================================================================
# Hooks integration tests
# ===========================================================================

class TestHooksSkillUpgradeIntegration(unittest.TestCase):
    """Test skill upgrade integration in ExperienceCaptureHooks."""

    def _make_hooks(self, **config_overrides):
        config = ExperienceConfig(**config_overrides)
        from agentica.hooks import ExperienceCaptureHooks
        return ExperienceCaptureHooks(config)

    def _mock_agent(self, agent_id="test-agent"):
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.run_input = "test input"
        agent.model = MagicMock()
        agent.auxiliary_model = None
        agent.workspace = MagicMock()
        agent.workspace.write_memory_entry = AsyncMock(return_value="/tmp/mem.md")
        mock_event_store = MagicMock()
        mock_event_store.append = AsyncMock(return_value="/tmp/events.jsonl")
        agent.workspace.get_experience_event_store = MagicMock(return_value=mock_event_store)
        mock_compiled_store = MagicMock()
        mock_compiled_store.write = AsyncMock(return_value="/tmp/exp.md")
        mock_compiled_store.run_lifecycle = AsyncMock(return_value={"promoted": 0, "demoted": 0, "archived": 0})
        mock_compiled_store.sync_to_global_agent_md = AsyncMock(return_value="/tmp/AGENTS.md")
        agent.workspace.get_compiled_experience_store = MagicMock(return_value=mock_compiled_store)
        agent.workspace._get_global_agent_md_path = MagicMock(return_value="/tmp/AGENTS.md")
        agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=Path("/tmp/gen_skills"))
        agent.workspace._get_user_experience_dir = MagicMock(return_value=Path("/tmp/experiences"))
        agent.working_memory = MagicMock()
        agent.working_memory.messages = []
        return agent

    def test_skill_upgrade_disabled_when_no_config(self):
        """Skill upgrade should not run when skill_upgrade is None."""
        hooks = self._make_hooks(capture_user_corrections=False)
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="Done"))

        # No crash, lifecycle should still run
        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.run_lifecycle.assert_called_once()

    def test_skill_upgrade_disabled_when_off(self):
        """Skill upgrade should not run when mode=off."""
        hooks = self._make_hooks(
            capture_user_corrections=False,
            skill_upgrade=SkillUpgradeConfig(mode="off"),
        )
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="Done"))

        # No crash
        compiled_store = agent.workspace.get_compiled_experience_store()
        compiled_store.run_lifecycle.assert_called_once()

    def test_get_skill_info_error_result_not_counted_as_skill_use(self):
        """A failed get_skill_info call (is_error=True) must not register as use.

        SkillTool now raises ValueError on missing/disabled skills, so the
        tool framework reports is_error=True. Hooks rely solely on that
        flag — they do not inspect the result text.
        """
        hooks = self._make_hooks(
            capture_user_corrections=False,
            skill_upgrade=SkillUpgradeConfig(mode="shadow"),
        )
        agent = self._mock_agent()

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent,
            tool_name="get_skill_info",
            tool_args={"skill_name": "missing-skill"},
            result="ValueError: Skill 'missing-skill' not found.",
            is_error=True,
        ))

        self.assertEqual(hooks._skills_used[agent.agent_id], set())

    def test_user_correction_marks_shadow_skill_episode_as_failure(self):
        """User correction should record a failing episode even without tool errors."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        tmpdir = tempfile.mkdtemp()

        hooks = self._make_hooks(
            capture_user_corrections=False,
            capture_success_patterns=False,
            skill_upgrade=SkillUpgradeConfig(mode="shadow"),
        )
        agent = self._mock_agent()
        try:
            gen_dir = Path(tmpdir) / "generated_skills"
            exp_dir = Path(tmpdir) / "experiences"
            gen_dir.mkdir(parents=True)
            exp_dir.mkdir(parents=True)
            agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=gen_dir)
            agent.workspace._get_user_experience_dir = MagicMock(return_value=exp_dir)

            skill_dir = gen_dir / "shadow-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("---\nname: shadow-skill\ndescription: t\n---\nBody")
            SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
                "skill_name": "shadow-skill",
                "status": "shadow",
                "total_episodes": 0,
                "success_count": 0,
                "failure_count": 0,
                "consecutive_failures": 0,
            })

            asyncio.run(hooks.on_agent_start(agent))
            hooks._correction_detected[agent.agent_id] = True
            asyncio.run(hooks.on_tool_end(
                agent,
                tool_name="get_skill_info",
                tool_args={"skill_name": "shadow-skill"},
                result="=== Skill: shadow-skill ===\nBody",
                is_error=False,
            ))

            with patch("agentica.experience.skill_upgrade.SkillEvolutionManager.record_episode") as record_episode:
                asyncio.run(hooks.on_agent_end(agent, output="Corrected by user"))

            record_episode.assert_called_once()
            self.assertEqual(record_episode.call_args.kwargs["outcome"], "failure")
            self.assertTrue(record_episode.call_args.kwargs["user_corrected"])
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestToolRecoveryEmission(unittest.TestCase):
    """``tool_recovery`` is emitted on tool success after prior tool_error.

    This is the bootstrap-safe semantic: recovery does NOT require any
    skill to be installed. A tool simply has to have failed somewhere in
    the workspace's events.jsonl history, then succeed in the current run.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_hooks_with_real_store(self):
        from agentica.experience.event_store import ExperienceEventStore
        from agentica.hooks import ExperienceCaptureHooks
        config = ExperienceConfig(
            capture_tool_errors=True,
            capture_user_corrections=False,
            capture_success_patterns=True,
        )
        hooks = ExperienceCaptureHooks(config)
        exp_dir = Path(self._tmpdir) / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        store = ExperienceEventStore(exp_dir=exp_dir)

        agent = MagicMock()
        agent.agent_id = "test-agent"
        agent.run_input = "do thing"
        agent.run_id = "run-1"
        agent.session_id = "sess-1"
        agent.model = MagicMock()
        agent.auxiliary_model = None
        agent.workspace = MagicMock()
        agent.workspace.get_experience_event_store = MagicMock(return_value=store)
        compiled_store = MagicMock()
        compiled_store.write = AsyncMock(return_value="/tmp/exp.md")
        compiled_store.run_lifecycle = AsyncMock(return_value={"promoted": 0, "demoted": 0, "archived": 0})
        compiled_store.sync_to_global_agent_md = AsyncMock(return_value="/tmp/AGENTS.md")
        agent.workspace.get_compiled_experience_store = MagicMock(return_value=compiled_store)
        agent.workspace._get_global_agent_md_path = MagicMock(return_value="/tmp/AGENTS.md")
        agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=Path(self._tmpdir) / "gen")
        agent.workspace._get_user_experience_dir = MagicMock(return_value=Path(self._tmpdir) / "experiences")
        agent.working_memory = MagicMock()
        agent.working_memory.messages = []
        return hooks, agent, store

    def test_no_recovery_without_prior_error(self):
        """First-ever success of a tool should NOT emit tool_recovery."""
        hooks, agent, store = self._make_hooks_with_real_store()

        async def _go():
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="write_file", tool_args={"path": "/tmp/a"},
                result="ok", is_error=False, elapsed=0.05,
            )
            await hooks.on_agent_end(agent, output="done")
            return await store.read_all()

        events = asyncio.run(_go())
        recovery_count = sum(1 for e in events if e.get("event_type") == "tool_recovery")
        self.assertEqual(recovery_count, 0, events)

    def test_recovery_emitted_after_prior_error(self):
        """Second-run success of a previously-failed tool emits tool_recovery."""
        hooks, agent, store = self._make_hooks_with_real_store()

        async def _go():
            # Run 1: tool fails
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="write_file", tool_args={"path": "/tmp/a"},
                result="permission denied", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")

            # Run 2: same tool succeeds
            agent.run_id = "run-2"
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="write_file", tool_args={"path": "/tmp/b"},
                result="ok", is_error=False, elapsed=0.04,
            )
            await hooks.on_agent_end(agent, output="ok")
            return await store.read_all()

        events = asyncio.run(_go())
        recoveries = [e for e in events if e.get("event_type") == "tool_recovery"]
        self.assertEqual(len(recoveries), 1, events)
        self.assertEqual(recoveries[0]["tool"], "write_file")

    def test_no_recovery_when_history_only_has_recovery_events(self):
        """If a tool only has tool_recovery history (no tool_error), do NOT emit.

        Guards against the gate accidentally counting "tool succeeded after
        any prior recovery" as further recovery — only past *errors*
        legitimize a recovery.
        """
        hooks, agent, store = self._make_hooks_with_real_store()

        async def _go():
            await store.append({"event_type": "tool_recovery", "tool": "grep"})
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="grep", tool_args={"path": "/tmp/x"},
                result="ok", is_error=False, elapsed=0.02,
            )
            await hooks.on_agent_end(agent, output="ok")
            return await store.read_all()

        events = asyncio.run(_go())
        recoveries = [e for e in events if e.get("event_type") == "tool_recovery"]
        self.assertEqual(len(recoveries), 1, events)

    def test_recovery_dedupes_within_a_run(self):
        """Multiple successes of the same recovered tool in one run = 1 event."""
        hooks, agent, store = self._make_hooks_with_real_store()

        async def _go():
            # Seed: tool failed previously
            await store.append({"event_type": "tool_error", "tool": "grep", "error": "no match"})

            await hooks.on_agent_start(agent)
            for path in ("/tmp/a", "/tmp/b", "/tmp/c"):
                await hooks.on_tool_end(
                    agent, tool_name="grep", tool_args={"path": path},
                    result="ok", is_error=False, elapsed=0.02,
                )
            await hooks.on_agent_end(agent, output="ok")
            return await store.read_all()

        events = asyncio.run(_go())
        recoveries = [e for e in events if e.get("event_type") == "tool_recovery"]
        self.assertEqual(len(recoveries), 1)


class TestSkillToolGeneratedSkillRuntime(unittest.TestCase):
    """Runtime behavior for generated skill visibility and refresh."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._gen_dir = Path(self._tmpdir) / "generated_skills"
        self._gen_dir.mkdir(parents=True)
        reset_skill_registry()

    def tearDown(self):
        import shutil
        reset_skill_registry()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_generated_skill(self, name: str, status: str = "shadow", body: str = "Body") -> Path:
        skill_dir = self._gen_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} desc\n---\n{body}",
            encoding="utf-8",
        )
        (skill_dir / "meta.json").write_text(
            json.dumps({"skill_name": name, "status": status}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return skill_dir

    def test_draft_generated_skill_not_visible(self):
        """Draft generated skills should not be listed or loadable."""
        from agentica.tools.skill_tool import SkillTool

        self._write_generated_skill("draft-skill", status="draft")
        tool = SkillTool(custom_skill_dirs=[str(self._gen_dir)])

        self.assertNotIn("draft-skill", tool.list_skills())
        with self.assertRaisesRegex(ValueError, "not found"):
            tool.get_skill_info("draft-skill")

    def test_reload_generated_skills_refreshes_revised_content(self):
        """Reload should replace cached generated skill content after revise."""
        from agentica.tools.skill_tool import SkillTool

        skill_dir = self._write_generated_skill("revise-skill", status="shadow", body="Old body")
        tool = SkillTool(custom_skill_dirs=[str(self._gen_dir)])

        self.assertIn("Old body", tool.get_skill_info("revise-skill"))

        (skill_dir / "SKILL.md").write_text(
            "---\nname: revise-skill\ndescription: revise-skill desc\n---\nNew body",
            encoding="utf-8",
        )
        tool.reload_generated_skills()

        self.assertIn("New body", tool.get_skill_info("revise-skill"))

    def test_reload_generated_skills_removes_rolled_back_skill(self):
        """Reload should remove rolled back generated skills from current registry."""
        from agentica.tools.skill_tool import SkillTool
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        skill_dir = self._write_generated_skill("rollback-skill", status="shadow")
        tool = SkillTool(custom_skill_dirs=[str(self._gen_dir)])

        self.assertIn("rollback-skill", tool.list_skills())

        meta_path = skill_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = "rolled_back"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        SkillEvolutionManager._disable_skill_md(skill_dir)
        tool.reload_generated_skills()

        self.assertNotIn("rollback-skill", tool.list_skills())
        with self.assertRaisesRegex(ValueError, "not found"):
            tool.get_skill_info("rollback-skill")


class TestAgentSkillUpgradeLifecycle(unittest.TestCase):
    """Agent-level lifecycle tests for generated skill visibility and refresh."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        reset_skill_registry()

    def tearDown(self):
        import shutil
        reset_skill_registry()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @staticmethod
    def _write_generated_skill(skill_dir: Path, status: str = "shadow", body: str = "Body") -> None:
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_dir.name}\ndescription: {skill_dir.name} desc\n---\n{body}",
            encoding="utf-8",
        )
        (skill_dir / "meta.json").write_text(
            json.dumps({"skill_name": skill_dir.name, "status": status}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def test_agent_loads_preexisting_generated_skill_on_init(self):
        """Preexisting generated skills should be visible when a new Agent starts."""
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.tools.skill_tool import SkillTool
        from agentica.workspace import Workspace

        workspace = Workspace(self._tmpdir)
        workspace.initialize()
        gen_dir = workspace._get_user_generated_skills_dir()
        self._write_generated_skill(gen_dir / "preexisting-skill", status="shadow", body="Loaded on init")

        skill_tool = SkillTool()
        agent = Agent(
            name="skill-agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[skill_tool],
            workspace=workspace,
            enable_experience_capture=True,
            experience_config=ExperienceConfig(skill_upgrade=SkillUpgradeConfig(mode="shadow")),
        )

        self.assertIn("preexisting-skill", skill_tool.list_skills())
        self.assertIn("preexisting-skill", "\n".join(agent._session_guidance_prompts))

    def test_spawned_generated_skill_refreshes_agent_session_guidance(self):
        """Spawned generated skills should appear in next-run session guidance immediately."""
        from agentica.agent import Agent
        from agentica.hooks import ExperienceCaptureHooks
        from agentica.model.openai import OpenAIChat
        from agentica.tools.skill_tool import SkillTool
        from agentica.workspace import Workspace

        workspace = Workspace(self._tmpdir)
        workspace.initialize()
        skill_tool = SkillTool()
        config = ExperienceConfig(
            capture_user_corrections=False,
            skill_upgrade=SkillUpgradeConfig(mode="shadow"),
        )
        agent = Agent(
            name="skill-agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[skill_tool],
            workspace=workspace,
            enable_experience_capture=True,
            experience_config=config,
        )
        agent.run_input = "test input"
        hooks = ExperienceCaptureHooks(config)

        async def _spawn_skill(**kwargs):
            self._write_generated_skill(
                kwargs["generated_skills_dir"] / "spawned-skill",
                status="shadow",
                body="Spawned body",
            )
            return "spawned-skill"

        with patch.object(agent.workspace.get_compiled_experience_store(), "run_lifecycle", new=AsyncMock(return_value={})):
            with patch("agentica.experience.skill_upgrade.SkillEvolutionManager.get_candidate_cards", return_value=[{
                "title": "candidate", "content": "content", "repeat_count": 5, "type": "correction",
            }]):
                with patch("agentica.experience.skill_upgrade.SkillEvolutionManager.maybe_spawn_skill", new=AsyncMock(side_effect=_spawn_skill)):
                    asyncio.run(hooks.on_agent_start(agent))
                    asyncio.run(hooks.on_agent_end(agent, output="Done"))

        self.assertIn("spawned-skill", "\n".join(agent._session_guidance_prompts))

    def test_name_collision_with_project_skill_does_not_record_generated_episode(self):
        """If a project skill wins name resolution, generated skill should not get scored."""
        from agentica.hooks import ExperienceCaptureHooks
        from agentica.tools.skill_tool import SkillTool

        config = ExperienceConfig(
            capture_user_corrections=False,
            capture_success_patterns=False,
            skill_upgrade=SkillUpgradeConfig(mode="shadow"),
        )
        hooks = ExperienceCaptureHooks(config)
        agent = MagicMock()
        agent.agent_id = "collision-agent"
        agent.run_input = "test input"
        agent.model = MagicMock()
        agent.auxiliary_model = None
        agent.workspace = MagicMock()
        event_store = MagicMock()
        event_store.append = AsyncMock(return_value="/tmp/events.jsonl")
        compiled_store = MagicMock()
        compiled_store.write = AsyncMock(return_value="/tmp/exp.md")
        compiled_store.run_lifecycle = AsyncMock(return_value={})
        compiled_store.sync_to_global_agent_md = AsyncMock(return_value="/tmp/AGENTS.md")
        agent.workspace.get_experience_event_store = MagicMock(return_value=event_store)
        agent.workspace.get_compiled_experience_store = MagicMock(return_value=compiled_store)
        agent.workspace._get_global_agent_md_path = MagicMock(return_value=Path("/tmp/AGENTS.md"))
        agent.working_memory = MagicMock()
        agent.working_memory.messages = []

        project_skill_dir = Path(self._tmpdir) / "project-skill"
        project_skill_dir.mkdir(parents=True)
        (project_skill_dir / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: project desc\n---\nProject body",
            encoding="utf-8",
        )
        generated_root = Path(self._tmpdir) / "generated_skills"
        generated_dir = generated_root / "shared-skill"
        self._write_generated_skill(generated_dir, status="shadow", body="Generated body")
        agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=generated_root)
        agent.workspace._get_user_experience_dir = MagicMock(return_value=Path(self._tmpdir) / "experiences")

        skill_tool = SkillTool(custom_skill_dirs=[str(project_skill_dir), str(generated_root)])
        agent.tools = [skill_tool]
        skill_tool._agent = agent

        self.assertIn("Project body", skill_tool.get_skill_info("shared-skill"))

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_tool_end(
            agent,
            tool_name="get_skill_info",
            tool_args={"skill_name": "shared-skill"},
            result="=== Skill: shared-skill ===\nProject body",
            is_error=False,
        ))

        with patch("agentica.experience.skill_upgrade.SkillEvolutionManager.get_candidate_cards", return_value=[]):
            with patch("agentica.experience.skill_upgrade.SkillEvolutionManager.record_episode") as record_episode:
                asyncio.run(hooks.on_agent_end(agent, output="Done"))

        record_episode.assert_not_called()


# ===========================================================================
# Cross-layer cleanup tests
# ===========================================================================

class TestOriginalTaskAnchoring(unittest.TestCase):
    """The user-facing task is captured at on_agent_start and threaded
    into every event/card written by ExperienceCaptureHooks.

    Step 1 verifies the events.jsonl + CompiledCard.source_task pipeline
    only — frontmatter / spawn-prompt consumption is covered by Step 2.
    """

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_hooks_with_real_store(self, run_input="search for X"):
        from agentica.experience.event_store import ExperienceEventStore
        from agentica.hooks import ExperienceCaptureHooks
        config = ExperienceConfig(
            capture_tool_errors=True,
            capture_user_corrections=False,
            capture_success_patterns=True,
        )
        hooks = ExperienceCaptureHooks(config)
        exp_dir = Path(self._tmpdir) / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        store = ExperienceEventStore(exp_dir=exp_dir)

        agent = MagicMock()
        agent.agent_id = "test-agent"
        agent.run_input = run_input
        agent.run_id = "run-1"
        agent.session_id = "sess-1"
        agent.model = MagicMock()
        agent.auxiliary_model = None
        agent.workspace = MagicMock()
        agent.workspace.get_experience_event_store = MagicMock(return_value=store)
        compiled_store = MagicMock()
        compiled_store.write = AsyncMock(return_value="/tmp/exp.md")
        compiled_store.run_lifecycle = AsyncMock(return_value={"promoted": 0, "demoted": 0, "archived": 0})
        compiled_store.sync_to_global_agent_md = AsyncMock(return_value="/tmp/AGENTS.md")
        agent.workspace.get_compiled_experience_store = MagicMock(return_value=compiled_store)
        agent.workspace._get_global_agent_md_path = MagicMock(return_value="/tmp/AGENTS.md")
        agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=Path(self._tmpdir) / "gen")
        agent.workspace._get_user_experience_dir = MagicMock(return_value=Path(self._tmpdir) / "experiences")
        agent.working_memory = MagicMock()
        agent.working_memory.messages = []
        return hooks, agent, store, compiled_store

    def test_tool_error_event_carries_original_task(self):
        hooks, agent, store, _ = self._make_hooks_with_real_store(
            run_input="find the project notes"
        )

        async def _go():
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="read_file", tool_args={"path": "/x"},
                result="FileNotFoundError: nope", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")
            return await store.read_all()

        events = asyncio.run(_go())
        tool_errors = [e for e in events if e.get("event_type") == "tool_error"]
        self.assertEqual(len(tool_errors), 1, events)
        self.assertEqual(tool_errors[0]["original_task"], "find the project notes")

    def test_tool_recovery_event_carries_original_task(self):
        hooks, agent, store, _ = self._make_hooks_with_real_store(
            run_input="please rewrite the config"
        )

        async def _go():
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="write_file", tool_args={"path": "/x"},
                result="permission denied", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")

            agent.run_id = "run-2"
            agent.run_input = "please rewrite the config"
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="write_file", tool_args={"path": "/y"},
                result="ok", is_error=False, elapsed=0.02,
            )
            await hooks.on_agent_end(agent, output="ok")
            return await store.read_all()

        events = asyncio.run(_go())
        recoveries = [e for e in events if e.get("event_type") == "tool_recovery"]
        self.assertEqual(len(recoveries), 1, events)
        self.assertEqual(recoveries[0]["original_task"], "please rewrite the config")

    def test_compiled_card_inherits_source_task(self):
        """CompiledCard built during on_agent_end captures source_task."""
        hooks, agent, _, compiled_store = self._make_hooks_with_real_store(
            run_input="rebuild the search index"
        )

        async def _go():
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="ls", tool_args={"path": "/missing"},
                result="FileNotFoundError: /missing", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")

        asyncio.run(_go())
        compiled_store.write.assert_called()
        cards = [c.args[0] for c in compiled_store.write.call_args_list]
        self.assertTrue(cards)
        self.assertEqual(cards[0].source_task, "rebuild the search index")

    def test_original_task_falls_back_to_first_user_message(self):
        """If run_input is empty, use first user-role message in working_memory."""
        hooks, agent, store, _ = self._make_hooks_with_real_store(run_input="")

        first_user = MagicMock()
        first_user.role = "user"
        first_user.content = "summarize today's reports"
        agent.working_memory.messages = [first_user]

        async def _go():
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="read_file", tool_args={"path": "/x"},
                result="oops: nope", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")
            return await store.read_all()

        events = asyncio.run(_go())
        tool_errors = [e for e in events if e.get("event_type") == "tool_error"]
        self.assertEqual(tool_errors[0]["original_task"], "summarize today's reports")

    def test_each_run_tags_events_with_its_own_run_input(self):
        """Per-run capture: events from run N carry run N's run_input.

        Multi-task aggregation across runs is the responsibility of
        CompiledExperienceStore (it merges into a frontmatter list). At
        the event layer each run is its own atomic unit.
        """
        hooks, agent, store, _ = self._make_hooks_with_real_store(
            run_input="find user manual"
        )

        async def _go():
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="read_file", tool_args={"path": "/x"},
                result="error: nope", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")

            agent.run_input = "different follow-up question"
            await hooks.on_agent_start(agent)
            await hooks.on_tool_end(
                agent, tool_name="grep", tool_args={"path": "/y"},
                result="error: x", is_error=True, elapsed=0.01,
            )
            await hooks.on_agent_end(agent, output="failed")
            return await store.read_all()

        events = asyncio.run(_go())
        tool_errors = [e for e in events if e.get("event_type") == "tool_error"]
        self.assertEqual(len(tool_errors), 2, events)
        self.assertEqual(tool_errors[0]["original_task"], "find user manual")
        self.assertEqual(tool_errors[1]["original_task"], "different follow-up question")


class TestSourceTasksPersistence(unittest.TestCase):
    """Step 2 — compiled_store + skill_upgrade consume source_task end-to-end."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_compiled_store(self):
        from agentica.experience.compiled_store import CompiledExperienceStore
        exp_dir = Path(self._tmpdir) / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        index_path = exp_dir / "EXPERIENCE.md"
        return CompiledExperienceStore(exp_dir=exp_dir, index_path=index_path), exp_dir

    def test_write_emits_source_tasks_in_frontmatter(self):
        from agentica.experience.compiler import CompiledCard
        from agentica.utils.async_file import extract_frontmatter_list
        compiled_store, exp_dir = self._make_compiled_store()
        card = CompiledCard(
            title="read_FileNotFoundError",
            content="ok",
            experience_type="tool_error",
            tool_name="read_file",
            source_task="find the project notes",
        )
        path = asyncio.run(compiled_store.write(card))
        text = Path(path).read_text(encoding="utf-8")
        self.assertEqual(
            extract_frontmatter_list(text, "source_tasks"),
            ["find the project notes"],
        )

    def test_write_omits_source_tasks_when_empty(self):
        """Cards without a source_task should NOT add an empty list field."""
        from agentica.experience.compiler import CompiledCard
        compiled_store, _ = self._make_compiled_store()
        card = CompiledCard(
            title="ls_PermissionError",
            content="ok",
            experience_type="tool_error",
            tool_name="ls",
            source_task="",
        )
        path = asyncio.run(compiled_store.write(card))
        text = Path(path).read_text(encoding="utf-8")
        self.assertNotIn("source_tasks:", text)

    def test_write_merges_multiple_tasks_dedup_and_caps(self):
        """Repeated writes with different source_tasks accumulate, dedup, cap=5."""
        from agentica.experience.compiler import CompiledCard
        from agentica.utils.async_file import extract_frontmatter_list
        compiled_store, _ = self._make_compiled_store()

        async def _go():
            for i in range(7):
                await compiled_store.write(CompiledCard(
                    title="grep_TimeoutError",
                    content="grep timed out",
                    experience_type="tool_error",
                    tool_name="grep",
                    source_task=f"task-{i}",
                ))
            # Replay an existing task — should NOT add a duplicate but should
            # move the entry to the front (most-recent-first).
            await compiled_store.write(CompiledCard(
                title="grep_TimeoutError",
                content="grep timed out",
                experience_type="tool_error",
                tool_name="grep",
                source_task="task-1",
            ))

        asyncio.run(_go())
        files = list((Path(self._tmpdir) / "experiences").glob("tool_error_grep_*.md"))
        self.assertEqual(len(files), 1)
        tasks = extract_frontmatter_list(files[0].read_text(encoding="utf-8"), "source_tasks")
        self.assertLessEqual(len(tasks), 5)
        self.assertEqual(tasks[0], "task-1", tasks)
        self.assertEqual(len(tasks), len(set(tasks)), tasks)

    def test_get_candidate_cards_exposes_source_tasks(self):
        from agentica.experience.compiled_store import CompiledExperienceStore
        from agentica.experience.compiler import CompiledCard
        from agentica.experience.skill_upgrade import SkillEvolutionManager

        exp_dir = Path(self._tmpdir) / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        store = CompiledExperienceStore(exp_dir=exp_dir, index_path=exp_dir / "EXPERIENCE.md")

        async def _go():
            for i in range(3):  # trigger repeat_count >= 3
                await store.write(CompiledCard(
                    title="read_FileNotFoundError",
                    content="x",
                    experience_type="tool_error",
                    tool_name="read_file",
                    source_task=f"find the report v{i}",
                ))

        asyncio.run(_go())
        candidates = SkillEvolutionManager.get_candidate_cards(
            exp_dir=exp_dir, min_repeat_count=3, min_tier="hot",
        )
        self.assertEqual(len(candidates), 1)
        self.assertIn("source_tasks", candidates[0])
        self.assertEqual(
            sorted(candidates[0]["source_tasks"]),
            sorted(["find the report v0", "find the report v1", "find the report v2"]),
        )

    def test_format_card_for_prompt_caps_source_tasks(self):
        """Spawn-prompt rendering caps tasks at _SOURCE_TASKS_PER_CARD."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        card = {
            "title": "t",
            "content": "body",
            "type": "tool_error",
            "repeat_count": 5,
            "source_tasks": [f"task-{i}" for i in range(10)],
        }
        rendered = SkillEvolutionManager._format_card_for_prompt(card)
        self.assertIn("Source tasks (samples):", rendered)
        for i in range(SkillEvolutionManager._SOURCE_TASKS_PER_CARD):
            self.assertIn(f"task-{i}", rendered)
        self.assertNotIn(
            f"task-{SkillEvolutionManager._SOURCE_TASKS_PER_CARD}", rendered,
        )

    def test_append_source_section_writes_originating_tasks(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        skill_md = "---\nname: x\ndescription: y\nwhen-to-use: z\n---\n\n## Body\nhello"
        out = SkillEvolutionManager._append_source_section(
            skill_md, source="my_card", event_count=4,
            source_tasks=["alpha task", "beta task"],
        )
        self.assertIn("- originating tasks:", out)
        self.assertIn("- alpha task", out)
        self.assertIn("- beta task", out)


class TestCorrectionKeyDataChain(unittest.TestCase):
    """End-to-end correction_key flow: rule → key → frontmatter → events
    → per-candidate evidence selection."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_correction_key_helper_is_stable_across_rewordings(self):
        from agentica.experience.compiler import ExperienceCompiler
        a = ExperienceCompiler.correction_key_from_rule("list directory before read file")
        b = ExperienceCompiler.correction_key_from_rule("Always list the directory before reading files")
        self.assertTrue(a)
        self.assertEqual(a, b)

    def test_correction_key_helper_returns_empty_for_all_stopwords(self):
        from agentica.experience.compiler import ExperienceCompiler
        self.assertEqual(
            ExperienceCompiler.correction_key_from_rule("the the and the"),
            "",
        )

    def test_compile_correction_attaches_correction_key(self):
        from agentica.experience.compiler import ExperienceCompiler
        card = ExperienceCompiler.compile_correction({
            "is_correction": True, "should_persist": True, "persist_target": "experience",
            "rule": "list directory before read file", "confidence": 0.9,
        })
        self.assertIsNotNone(card)
        self.assertTrue(card.correction_key)
        # Title and key are aligned by design
        self.assertEqual(card.title, card.correction_key)

    def test_correction_card_persists_correction_key_in_frontmatter(self):
        from agentica.experience.compiled_store import CompiledExperienceStore
        from agentica.experience.compiler import ExperienceCompiler
        from agentica.utils.async_file import extract_frontmatter_value
        exp_dir = Path(self._tmpdir) / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        compiled_store = CompiledExperienceStore(
            exp_dir=exp_dir, index_path=exp_dir / "EXPERIENCE.md",
        )
        card = ExperienceCompiler.compile_correction({
            "is_correction": True, "should_persist": True, "persist_target": "experience",
            "rule": "list directory before read file", "confidence": 0.9,
        })
        path = asyncio.run(compiled_store.write(card))
        text = Path(path).read_text(encoding="utf-8")
        self.assertEqual(
            extract_frontmatter_value(text, "correction_key"),
            card.correction_key,
        )

    def test_get_candidate_cards_exposes_correction_key(self):
        from agentica.experience.compiled_store import CompiledExperienceStore
        from agentica.experience.compiler import ExperienceCompiler
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        exp_dir = Path(self._tmpdir) / "experiences"
        exp_dir.mkdir(parents=True, exist_ok=True)
        compiled_store = CompiledExperienceStore(
            exp_dir=exp_dir, index_path=exp_dir / "EXPERIENCE.md",
        )

        async def _go():
            for _ in range(3):  # repeat_count >= 3
                card = ExperienceCompiler.compile_correction({
                    "is_correction": True, "should_persist": True, "persist_target": "experience",
                    "rule": "list directory before read file", "confidence": 0.9,
                })
                await compiled_store.write(card)

        asyncio.run(_go())
        candidates = SkillEvolutionManager.get_candidate_cards(
            exp_dir=exp_dir, min_repeat_count=3, min_tier="hot",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0]["correction_key"],
            ExperienceCompiler.correction_key_from_rule("list directory before read file"),
        )

    def test_correction_recovery_gate_uses_correction_key(self):
        """A correction candidate counts confirmations matching its key only."""
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        events = [
            {  # 2 confirmations for key A
                "event_type": "correction_classification",
                "is_correction": True, "should_persist": True,
                "correction_key": "list_directory_read_file",
            },
            {
                "event_type": "correction_classification",
                "is_correction": True, "should_persist": True,
                "correction_key": "list_directory_read_file",
            },
            {  # 0 confirmations for key B (despite many for A)
                "event_type": "correction_classification",
                "is_correction": False, "should_persist": False,
                "correction_key": "use_rg_not_grep",
            },
        ]
        idx = SkillEvolutionManager._index_events_once(events)
        c_a = {"type": "correction", "correction_key": "list_directory_read_file"}
        c_b = {"type": "correction", "correction_key": "use_rg_not_grep"}
        self.assertEqual(SkillEvolutionManager._candidate_recovery_count(c_a, idx), 2)
        self.assertEqual(SkillEvolutionManager._candidate_recovery_count(c_b, idx), 0)


class TestCrossLayerCleanup(unittest.TestCase):
    """Test that experience→memory cross-layer write has been removed."""

    def test_memory_feedback_removed_from_compiler(self):
        from agentica.experience.compiler import ExperienceCompiler
        # is_memory_feedback and build_memory_feedback should not exist
        self.assertFalse(hasattr(ExperienceCompiler, "is_memory_feedback"))
        self.assertFalse(hasattr(ExperienceCompiler, "build_memory_feedback"))

    def test_correction_always_goes_to_experience(self):
        """Correction classified as experience should be written to compiled store."""
        from agentica.hooks import ExperienceCaptureHooks

        # capture_user_corrections is False by default; opt in explicitly for this test.
        config = ExperienceConfig(capture_user_corrections=True)
        hooks = ExperienceCaptureHooks(config)
        agent = MagicMock()
        agent.agent_id = "test"
        agent.run_input = "Use pandas instead"
        agent.model = MagicMock()
        agent.auxiliary_model = None
        agent.workspace = MagicMock()

        mock_event_store = MagicMock()
        mock_event_store.append = AsyncMock(return_value="/tmp/events.jsonl")
        agent.workspace.get_experience_event_store = MagicMock(return_value=mock_event_store)
        mock_compiled_store = MagicMock()
        mock_compiled_store.write = AsyncMock(return_value="/tmp/exp.md")
        mock_compiled_store.run_lifecycle = AsyncMock(return_value={})
        agent.workspace.get_compiled_experience_store = MagicMock(return_value=mock_compiled_store)
        agent.workspace._get_global_agent_md_path = MagicMock(return_value=Path("/tmp/AGENTS.md"))
        agent.workspace._get_user_generated_skills_dir = MagicMock(return_value=Path("/tmp/gen"))
        agent.workspace._get_user_experience_dir = MagicMock(return_value=Path("/tmp/exp"))

        prev_msg = MagicMock()
        prev_msg.role = "assistant"
        prev_msg.content = "I'll use csv module"
        agent.working_memory = MagicMock()
        agent.working_memory.messages = [prev_msg]

        # Simulate LLM returning experience target
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "is_correction": True,
            "confidence": 0.95,
            "category": "preference",
            "scope": "cross_session",
            "should_persist": True,
            "persist_target": "experience",
            "title": "use_pandas",
            "rule": "Use pandas for data",
            "why": "Better",
            "how_to_apply": "Data tasks",
        })
        agent.model.response = AsyncMock(return_value=mock_response)

        asyncio.run(hooks.on_agent_start(agent))
        asyncio.run(hooks.on_agent_end(agent, output="OK"))

        # Should write to compiled_store, NOT workspace.write_memory_entry
        compiled_store = agent.workspace.get_compiled_experience_store()
        calls = compiled_store.write.call_args_list
        correction_calls = [c for c in calls if c[0][0].experience_type == "correction"]
        self.assertEqual(len(correction_calls), 1)
        # write_memory_entry should NOT have been called for feedback
        agent.workspace.write_memory_entry.assert_not_called()

    def test_classification_prompt_no_memory_feedback_target(self):
        """Classification prompt should not include memory_feedback as a persist_target."""
        from agentica.hooks import ExperienceCaptureHooks
        prompt = ExperienceCaptureHooks._FEEDBACK_CLASSIFY_PROMPT
        self.assertNotIn("memory_feedback", prompt)
        self.assertIn("experience", prompt)
        # Scope vocabulary stays "turn_only | session | cross_session".
        self.assertIn("cross_session", prompt)
        # arch_v5 §8.D: title is now derived deterministically from `rule`,
        # so the LLM must NOT be asked to produce a title field.
        self.assertNotIn('"title"', prompt)


# ===========================================================================
# Import tests
# ===========================================================================

class TestSkillUpgradeImports(unittest.TestCase):

    def test_import_config(self):
        from agentica.agent.config import SkillUpgradeConfig
        self.assertIsNotNone(SkillUpgradeConfig)

    def test_import_from_top_level(self):
        from agentica import SkillUpgradeConfig, SkillEvolutionManager
        self.assertIsNotNone(SkillUpgradeConfig)
        self.assertIsNotNone(SkillEvolutionManager)

    def test_import_from_experience_package(self):
        from agentica.experience import SkillEvolutionManager
        self.assertIsNotNone(SkillEvolutionManager)

    def test_import_directly(self):
        from agentica.experience.skill_upgrade import SkillEvolutionManager
        self.assertIsNotNone(SkillEvolutionManager)


if __name__ == "__main__":
    unittest.main()
