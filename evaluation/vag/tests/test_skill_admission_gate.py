# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for VaG skill admission gate behavior.
"""
import asyncio
import unittest

from agentica.critic import CritiqueResult, ExecCritic, SchemaCritic
from evaluation.vag.lifecycle import (
    SkillAdmissionGate,
    SkillCandidate,
    skill_fingerprint,
)


VALID_SKILL_MD = """---
name: safe-skill
description: Check file paths before reading them
when-to-use: file operations, path validation
---

Check that a path exists before reading it.

## Gotchas
- Missing files produce noisy retries; list the parent directory first.
"""


class MockAgentCritic:
    name = "agent"

    async def __call__(self, task: str, answer: str) -> CritiqueResult:
        if "rm -rf" in answer or "secret" in answer.lower():
            return CritiqueResult(
                approved=False,
                issues="semantic risk: destructive or secret-bearing guidance",
                critic_name=self.name,
            )
        return CritiqueResult(approved=True, critic_name=self.name)


class TestSkillAdmissionGate(unittest.TestCase):
    def test_all_critics_approve(self):
        gate = SkillAdmissionGate(critics=[
            SchemaCritic(SkillCandidate),
            ExecCritic(lambda task, answer: "## Gotchas" in answer),
            MockAgentCritic(),
        ])

        result = asyncio.run(gate.evaluate(VALID_SKILL_MD, task="admit"))

        self.assertTrue(result.approved)
        self.assertEqual(result.rejected_by, [])
        self.assertEqual([v.critic_name for v in result.verdicts], ["schema", "exec", "agent"])
        self.assertEqual(result.fingerprint, skill_fingerprint(VALID_SKILL_MD))

    def test_single_critic_reject(self):
        gate = SkillAdmissionGate(critics=[
            SchemaCritic(SkillCandidate),
            ExecCritic(lambda task, answer: False, name="holdout"),
        ])

        result = asyncio.run(gate.evaluate(VALID_SKILL_MD, task="admit"))

        self.assertFalse(result.approved)
        self.assertEqual(result.rejected_by, ["holdout"])

    def test_multiple_critics_reject(self):
        destructive = VALID_SKILL_MD + "\nRun rm -rf /tmp/cache when confused.\n"
        gate = SkillAdmissionGate(critics=[
            ExecCritic(lambda task, answer: False, name="holdout"),
            MockAgentCritic(),
        ])

        result = asyncio.run(gate.evaluate(destructive, task="admit"))

        self.assertFalse(result.approved)
        self.assertEqual(set(result.rejected_by), {"holdout", "agent"})

    def test_schema_critic_rejects_invalid_frontmatter(self):
        invalid = """---
name: broken-skill
---

Body exists but description is missing.
"""
        gate = SkillAdmissionGate(critics=[SchemaCritic(SkillCandidate)])

        result = asyncio.run(gate.evaluate(invalid, task="admit"))

        self.assertFalse(result.approved)
        self.assertEqual(result.rejected_by, ["schema"])

    def test_fingerprint_is_stable(self):
        gate = SkillAdmissionGate(critics=[])

        first = asyncio.run(gate.evaluate(VALID_SKILL_MD))
        second = asyncio.run(gate.evaluate(VALID_SKILL_MD))

        self.assertEqual(first.fingerprint, second.fingerprint)


if __name__ == "__main__":
    unittest.main()
