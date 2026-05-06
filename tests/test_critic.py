# -*- coding: utf-8 -*-
"""Unit tests for agentica.critic — Protocol, adapters, and refine()."""
import os
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from pydantic import BaseModel, Field

from agentica.critic import (
    CritiqueResult,
    CritiqueStyle,
    RefineResult,
    SchemaCritic,
    AgentCritic,
    refine,
)


# ---- Helpers ----

def _mock_agent(responses):
    """Build a mock Agent whose .run() returns each response in order."""
    agent = MagicMock()
    agent.name = "mock"
    iter_resp = iter(responses)

    async def _run(message, **_kwargs):
        text = next(iter_resp)
        resp = MagicMock()
        resp.content = text
        return resp

    agent.run = AsyncMock(side_effect=_run)
    return agent


# ---- CritiqueResult ----

class TestCritiqueResult(unittest.TestCase):
    def test_default_fields(self):
        r = CritiqueResult(approved=True)
        self.assertTrue(r.approved)
        self.assertEqual(r.issues, "")
        self.assertEqual(r.critic_name, "")

    def test_with_issues(self):
        r = CritiqueResult(approved=False, issues="bad", critic_name="schema")
        self.assertFalse(r.approved)
        self.assertEqual(r.issues, "bad")


# ---- SchemaCritic ----

class _Reply(BaseModel):
    name: str
    age: int = Field(ge=0)


class TestSchemaCritic(unittest.TestCase):
    def test_approves_valid_json(self):
        c = SchemaCritic(_Reply, name="reply_schema")
        result = asyncio.run(c("task", '{"name": "alice", "age": 30}'))
        self.assertTrue(result.approved)
        self.assertEqual(result.critic_name, "reply_schema")

    def test_rejects_invalid_json(self):
        c = SchemaCritic(_Reply, name="reply_schema")
        result = asyncio.run(c("task", '{"name": "alice", "age": -5}'))
        self.assertFalse(result.approved)
        self.assertIn("age", result.issues.lower())

    def test_rejects_unparseable_text(self):
        c = SchemaCritic(_Reply)
        result = asyncio.run(c("task", "not json at all"))
        self.assertFalse(result.approved)
        self.assertTrue(len(result.issues) > 0)


# ---- AgentCritic ----

class TestAgentCritic(unittest.TestCase):
    def test_approves_when_agent_says_approved(self):
        critic_agent = _mock_agent(["APPROVED — looks good"])
        c = AgentCritic(critic_agent, name="reviewer")
        result = asyncio.run(c("task", "draft text"))
        self.assertTrue(result.approved)
        self.assertEqual(result.critic_name, "reviewer")

    def test_rejects_when_agent_lists_issues(self):
        critic_agent = _mock_agent(["Issue: too short.\nIssue: no evidence."])
        c = AgentCritic(critic_agent)
        result = asyncio.run(c("task", "draft"))
        self.assertFalse(result.approved)
        self.assertIn("too short", result.issues)


# ---- refine() ----

class TestRefine(unittest.TestCase):
    def test_single_critic_first_draft_approved(self):
        actor = _mock_agent(["Draft A"])
        critic_agent = _mock_agent(["APPROVED"])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic]))
        self.assertIsInstance(result, RefineResult)
        self.assertEqual(result.final_draft, "Draft A")
        self.assertTrue(result.approved)
        self.assertEqual(result.stopped_reason, "approved")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(actor.run.call_count, 1)
        self.assertEqual(critic_agent.run.call_count, 1)

    def test_single_critic_revision_then_approved(self):
        actor = _mock_agent(["Draft A", "Draft B (revised)"])
        critic_agent = _mock_agent(["Issue: too short", "APPROVED"])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=2))
        self.assertEqual(result.final_draft, "Draft B (revised)")
        self.assertTrue(result.approved)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(actor.run.call_count, 2)

    def test_multi_critic_all_approve(self):
        actor = _mock_agent(['{"name": "x", "age": 1}'])
        schema_c = SchemaCritic(_Reply)
        agent_c = AgentCritic(_mock_agent(["APPROVED"]))
        result = asyncio.run(refine(actor, "task", [schema_c, agent_c]))
        self.assertEqual(actor.run.call_count, 1)
        self.assertIn("name", result.final_draft)
        self.assertTrue(result.approved)

    def test_multi_critic_one_rejects_triggers_revision(self):
        actor = _mock_agent([
            '{"name": "x", "age": -1}',
            '{"name": "x", "age": 5}',
        ])
        agent_critic_agent = _mock_agent(["APPROVED", "APPROVED"])
        schema_c = SchemaCritic(_Reply)
        agent_c = AgentCritic(agent_critic_agent)
        result = asyncio.run(
            refine(actor, "task", [schema_c, agent_c], max_iter=2)
        )
        self.assertEqual(actor.run.call_count, 2)
        self.assertIn('"age": 5', result.final_draft)
        self.assertTrue(result.approved)

    def test_max_iter_exhausted(self):
        actor = _mock_agent(["v1", "v2", "v3"])
        # Distinct issues each round so loop-detection does NOT fire;
        # verify max_iter is the genuine stop reason.
        critic_agent = _mock_agent(["issue 1", "issue 2", "issue 3"])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=2))
        self.assertEqual(result.final_draft, "v3")
        self.assertFalse(result.approved)
        self.assertEqual(result.stopped_reason, "max_iter")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(actor.run.call_count, 3)

    def test_zero_iter_skips_critic_loop(self):
        actor = _mock_agent(["Draft A"])
        critic_agent = _mock_agent([])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=0))
        self.assertEqual(result.final_draft, "Draft A")
        self.assertEqual(result.stopped_reason, "no_critics")
        self.assertEqual(critic_agent.run.call_count, 0)

    def test_empty_critics_list_is_passthrough(self):
        actor = _mock_agent(["Draft A"])
        result = asyncio.run(refine(actor, "task", []))
        self.assertEqual(result.final_draft, "Draft A")
        self.assertEqual(result.stopped_reason, "no_critics")
        self.assertEqual(actor.run.call_count, 1)

    def test_loop_detection_stops_early_on_identical_verdicts(self):
        # Critic returns the SAME issue text twice → loop detected, stop.
        # Without loop detection, max_iter=5 would call actor 6 times.
        actor = _mock_agent(["v1", "v2", "v3", "v4", "v5", "v6"])
        critic_agent = _mock_agent([
            "missing evidence",
            "missing evidence",
            "missing evidence",
        ])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=5))
        self.assertEqual(result.stopped_reason, "loop_detected")
        self.assertFalse(result.approved)
        # Iteration 1: verdict recorded, actor revises -> v2
        # Iteration 2: same verdict -> loop detected, no further revision.
        self.assertEqual(result.iterations, 2)
        self.assertEqual(actor.run.call_count, 2)

    def test_history_contains_drafts_and_verdicts_per_round(self):
        actor = _mock_agent(["v1", "v2"])
        critic_agent = _mock_agent(["needs more detail", "APPROVED"])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=2))
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.history[0].draft, "v1")
        self.assertFalse(result.history[0].verdicts[0].approved)
        self.assertEqual(result.history[1].draft, "v2")
        self.assertTrue(result.history[1].verdicts[0].approved)


# ---- CritiqueStyle ----

class TestCritiqueStyle(unittest.TestCase):
    def test_style_is_string_enum_with_three_values(self):
        self.assertEqual(CritiqueStyle.STRICT.value, "strict")
        self.assertEqual(CritiqueStyle.NEUTRAL.value, "neutral")
        self.assertEqual(CritiqueStyle.LENIENT.value, "lenient")

    def test_default_style_is_neutral(self):
        c = AgentCritic(_mock_agent(["APPROVED"]))
        self.assertEqual(c.style, CritiqueStyle.NEUTRAL)

    def test_strict_style_is_injected_into_prompt(self):
        critic_agent = _mock_agent(["APPROVED"])
        c = AgentCritic(critic_agent, style=CritiqueStyle.STRICT)
        asyncio.run(c("task", "answer"))
        sent_prompt = critic_agent.run.call_args.args[0]
        self.assertIn("strict", sent_prompt.lower())

    def test_lenient_style_is_injected_into_prompt(self):
        critic_agent = _mock_agent(["APPROVED"])
        c = AgentCritic(critic_agent, style=CritiqueStyle.LENIENT)
        asyncio.run(c("task", "answer"))
        sent_prompt = critic_agent.run.call_args.args[0]
        self.assertIn("lenient", sent_prompt.lower())


if __name__ == "__main__":
    unittest.main()
