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
        self.assertEqual(result, "Draft A")
        self.assertEqual(actor.run.call_count, 1)
        self.assertEqual(critic_agent.run.call_count, 1)

    def test_single_critic_revision_then_approved(self):
        actor = _mock_agent(["Draft A", "Draft B (revised)"])
        critic_agent = _mock_agent(["Issue: too short", "APPROVED"])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=2))
        self.assertEqual(result, "Draft B (revised)")
        self.assertEqual(actor.run.call_count, 2)

    def test_multi_critic_all_approve(self):
        # Both critics approve immediately — actor only called once.
        actor = _mock_agent(['{"name": "x", "age": 1}'])
        schema_c = SchemaCritic(_Reply)
        agent_c = AgentCritic(_mock_agent(["APPROVED"]))
        result = asyncio.run(refine(actor, "task", [schema_c, agent_c]))
        self.assertEqual(actor.run.call_count, 1)
        self.assertIn("name", result)

    def test_multi_critic_one_rejects_triggers_revision(self):
        # First draft fails schema; second draft passes both.
        actor = _mock_agent([
            '{"name": "x", "age": -1}',  # fails schema
            '{"name": "x", "age": 5}',   # passes schema
        ])
        agent_critic_agent = _mock_agent(["APPROVED", "APPROVED"])
        schema_c = SchemaCritic(_Reply)
        agent_c = AgentCritic(agent_critic_agent)
        result = asyncio.run(
            refine(actor, "task", [schema_c, agent_c], max_iter=2)
        )
        self.assertEqual(actor.run.call_count, 2)
        self.assertIn('"age": 5', result)

    def test_max_iter_exhausted(self):
        actor = _mock_agent(["v1", "v2", "v3"])
        critic_agent = _mock_agent(["bad", "still bad", "no good"])
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=2))
        # initial + 2 revisions = 3 actor calls; final = "v3"
        self.assertEqual(result, "v3")
        self.assertEqual(actor.run.call_count, 3)

    def test_zero_iter_skips_critic_loop(self):
        actor = _mock_agent(["Draft A"])
        critic_agent = _mock_agent([])  # never called
        critic = AgentCritic(critic_agent)
        result = asyncio.run(refine(actor, "task", [critic], max_iter=0))
        self.assertEqual(result, "Draft A")
        self.assertEqual(critic_agent.run.call_count, 0)

    def test_empty_critics_list_is_passthrough(self):
        actor = _mock_agent(["Draft A"])
        result = asyncio.run(refine(actor, "task", []))
        self.assertEqual(result, "Draft A")
        self.assertEqual(actor.run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
