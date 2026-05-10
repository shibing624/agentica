# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for VaG integration in the skill upgrade lifecycle.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from agentica.critic import CritiqueResult
from agentica.experience.skill_upgrade import SkillEvolutionManager
from agentica.skills.provenance import read_provenance_events


VALID_SKILL_MD = """---
name: gated-skill
description: Check paths before reading files
when-to-use: file operations
---

Check that a path exists before reading it.

## Gotchas
- Missing files trigger noisy retries; list the parent directory first.

## Minimal Example
```python
from pathlib import Path
path = Path("data/input.txt")
if path.exists():
    content = path.read_text()
```
"""


class ApproveCritic:
    name = "approve"

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        return CritiqueResult(approved=True, critic_name=self.name)


class RejectCritic:
    name = "reject"

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        return CritiqueResult(
            approved=False,
            issues="unsafe candidate",
            critic_name=self.name,
        )


class TailRiskCritic:
    name = "tail_risk"

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        return CritiqueResult(
            approved="tail-risk" not in answer,
            issues="tail risk marker found" if "tail-risk" in answer else "",
            critic_name=self.name,
        )


class TestSkillUpgradeVaG(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._gen_dir = Path(self._tmpdir.name) / "generated_skills"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _spawn_model(self):
        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "action": "install_shadow",
            "skill_name": "gated-skill",
            "source_experience": "path-check-card",
            "skill_md": VALID_SKILL_MD,
        })))
        return model

    def test_admission_gate_approve_installs_shadow_and_records_provenance(self):
        manager = SkillEvolutionManager()

        slug = asyncio.run(manager.maybe_spawn_skill(
            model=self._spawn_model(),
            candidates=[{"title": "path-check", "content": "Check paths", "repeat_count": 5}],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
            admission_critics=[ApproveCritic()],
        ))

        skill_dir = self._gen_dir / "gated-skill"
        self.assertEqual(slug, "gated-skill")
        self.assertTrue((skill_dir / "SKILL.md").exists())
        events = read_provenance_events(skill_dir)
        self.assertEqual(events[0]["event"], "admission")
        self.assertTrue(events[0]["approved"])

    def test_admission_gate_reject_does_not_install(self):
        manager = SkillEvolutionManager()

        slug = asyncio.run(manager.maybe_spawn_skill(
            model=self._spawn_model(),
            candidates=[{"title": "path-check", "content": "Check paths", "repeat_count": 5}],
            existing_skills=[],
            generated_skills_dir=self._gen_dir,
            admission_critics=[RejectCritic()],
        ))

        skill_dir = self._gen_dir / "gated-skill"
        self.assertIsNone(slug)
        self.assertFalse((skill_dir / "SKILL.md").exists())
        events = read_provenance_events(skill_dir)
        self.assertEqual(events[0]["event"], "admission")
        self.assertFalse(events[0]["approved"])
        self.assertEqual(events[0]["rejected_by"], ["reject"])

    def test_promotion_gate_reject_keeps_shadow(self):
        skill_dir = self._gen_dir / "gated-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "gated-skill",
            "status": "shadow",
            "source_experience": "path-check-card",
            "total_episodes": 5,
            "success_count": 5,
            "failure_count": 0,
            "consecutive_failures": 0,
        })
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="success",
                query=f"q{i}",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "promote",
            "reason": "Strong performance",
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model,
            skill_dir=skill_dir,
            checkpoint_interval=5,
            promotion_critics=[RejectCritic()],
        ))

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(decision, "keep_shadow")
        self.assertEqual(meta["status"], "shadow")
        events = read_provenance_events(skill_dir)
        self.assertEqual(events[0]["event"], "promotion")
        self.assertFalse(events[0]["approved"])

    def test_promotion_gate_evaluates_full_skill_not_prompt_preview(self):
        skill_dir = self._gen_dir / "long-skill"
        skill_dir.mkdir(parents=True)
        long_body = VALID_SKILL_MD + "\n" + ("safe guidance\n" * 250) + "tail-risk\n"
        (skill_dir / "SKILL.md").write_text(long_body, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "long-skill",
            "status": "shadow",
            "source_experience": "long-card",
            "total_episodes": 5,
            "success_count": 5,
            "failure_count": 0,
            "consecutive_failures": 0,
        })
        for i in range(5):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="success",
                query=f"q{i}",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "promote",
            "reason": "Strong performance",
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model,
            skill_dir=skill_dir,
            checkpoint_interval=5,
            promotion_critics=[TailRiskCritic()],
        ))

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        self.assertEqual(decision, "keep_shadow")
        self.assertEqual(meta["status"], "shadow")

    def test_repeated_failures_can_repair_skill(self):
        skill_dir = self._gen_dir / "repairable-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "repairable-skill",
            "status": "auto",
            "source_experience": "path-check-card",
            "version": 1,
            "total_episodes": 3,
            "success_count": 1,
            "failure_count": 2,
            "consecutive_failures": 2,
        })
        for i in range(3):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="failure" if i else "success",
                query=f"q{i}",
            )

        repaired_md = VALID_SKILL_MD.replace(
            "Check that a path exists before reading it.",
            "Check that a path exists and list its parent directory before reading it.",
        )
        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "repair",
            "reason": "The failure trace shows missing parent directory inspection.",
            "revised_skill_md": repaired_md,
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model,
            skill_dir=skill_dir,
            rollback_consecutive_failures=2,
            promotion_critics=[ApproveCritic()],
            maintain_failed_skills=True,
        ))

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        skill_content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        events = read_provenance_events(skill_dir)
        self.assertEqual(decision, "repair")
        self.assertIn("list its parent directory", skill_content)
        self.assertEqual(meta["version"], 2)
        self.assertEqual(meta["consecutive_failures"], 0)
        self.assertEqual(meta["repair_attempts"], 0)
        self.assertEqual(events[0]["event"], "repair")
        self.assertTrue(events[0]["approved"])

    def test_repair_uses_repair_critics(self):
        skill_dir = self._gen_dir / "repair-gated-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "repair-gated-skill",
            "status": "auto",
            "source_experience": "path-check-card",
            "version": 1,
            "total_episodes": 3,
            "success_count": 1,
            "failure_count": 2,
            "consecutive_failures": 2,
        })
        for i in range(3):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="failure",
                query=f"q{i}",
            )

        repaired_md = VALID_SKILL_MD.replace(
            "Check that a path exists before reading it.",
            "This repaired body should not be installed.",
        )
        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "repair",
            "reason": "Repair attempt",
            "revised_skill_md": repaired_md,
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model,
            skill_dir=skill_dir,
            rollback_consecutive_failures=2,
            promotion_critics=[ApproveCritic()],
            repair_critics=[RejectCritic()],
            maintain_failed_skills=True,
        ))

        skill_content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        events = read_provenance_events(skill_dir)
        self.assertEqual(decision, "keep_shadow")
        self.assertNotIn("This repaired body should not be installed", skill_content)
        self.assertEqual(events[0]["event"], "repair")
        self.assertFalse(events[0]["approved"])
        self.assertEqual(events[0]["rejected_by"], ["reject"])

    def test_repeated_failures_can_discard_skill(self):
        skill_dir = self._gen_dir / "discard-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "discard-skill",
            "status": "auto",
            "source_experience": "path-check-card",
            "total_episodes": 3,
            "success_count": 1,
            "failure_count": 2,
            "consecutive_failures": 2,
        })
        for i in range(3):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="failure",
                query=f"q{i}",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content="DISCARD: tool no longer exists"))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model,
            skill_dir=skill_dir,
            rollback_consecutive_failures=2,
            maintain_failed_skills=True,
        ))

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        events = read_provenance_events(skill_dir)
        self.assertEqual(decision, "retired")
        self.assertEqual(meta["status"], "retired")
        self.assertFalse((skill_dir / "SKILL.md").exists())
        self.assertTrue((skill_dir / "SKILL.md.disabled").exists())
        self.assertEqual(events[0]["event"], "discard")

    def test_failed_repairs_retire_after_budget(self):
        skill_dir = self._gen_dir / "failed-repair-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(VALID_SKILL_MD, encoding="utf-8")
        SkillEvolutionManager.write_meta(skill_dir / "meta.json", {
            "skill_name": "failed-repair-skill",
            "status": "auto",
            "source_experience": "path-check-card",
            "repair_attempts": 2,
            "total_episodes": 3,
            "success_count": 1,
            "failure_count": 2,
            "consecutive_failures": 2,
        })
        for i in range(3):
            SkillEvolutionManager.record_episode(
                skill_dir / "episodes.jsonl",
                outcome="failure",
                query=f"q{i}",
            )

        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps({
            "decision": "repair",
            "reason": "Try repair but omit the body",
        })))

        manager = SkillEvolutionManager()
        decision = asyncio.run(manager.maybe_update_skill_state(
            model=model,
            skill_dir=skill_dir,
            rollback_consecutive_failures=2,
            maintain_failed_skills=True,
            max_repair_attempts=3,
        ))

        meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
        events = read_provenance_events(skill_dir)
        self.assertEqual(decision, "retired")
        self.assertEqual(meta["status"], "retired")
        self.assertEqual([event["event"] for event in events], ["repair", "discard"])


if __name__ == "__main__":
    unittest.main()
