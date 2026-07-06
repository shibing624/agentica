# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Integration tests for arch_v5.md Phase 1/2/3 invariants.

Covers gaps the dataclass-shape tests in `test_run_context.py` /
`test_learning_report.py` / `test_evidence_gate.py` left open:

- Phase 1 / Issue B / C1: TaskAnchor MUST survive across multiple
  `agent.run()` calls in the same session (otherwise the "frozen
  original goal" promise is silently broken on every multi-turn
  conversation).
- Issue A: a misbehaving event_callback must not crash the run; the
  failure must be visible at WARNING level (no silent debug noise).
- Issue D / I5: a failing learning-report write must not crash the
  hook; the operator must see a WARNING.
- I6: the candidate review API (list / promote / reject) must round-trip.

All tests stub the LLM call site so no real API key is required.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock, patch

from agentica.agent import Agent
from agentica.hooks import ExperienceCaptureHooks
from agentica.learning_report import LearningReport
from agentica.model.message import Message
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.run_context import RunSource
from agentica.workspace import Workspace


def _make_agent(name: str = "anchor-agent", session_id: str = "session-A") -> Agent:
    return Agent(
        name=name,
        session_id=session_id,
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
    )


def _stub_model_response(agent: Agent, content: str = "ok") -> None:
    """Replace the model's response() with a no-tool-call canned reply.

    The Runner appends the assistant reply to messages_for_model and exits the
    loop because no tool calls remain.
    """

    async def _response(messages: List[Message]) -> ModelResponse:
        messages.append(Message(role="assistant", content=content))
        return ModelResponse(content=content)

    agent.update_model()
    agent.model.response = _response


class TestSessionAnchorReuse(unittest.TestCase):
    """Phase 1 invariant: anchor is pinned to the FIRST user message of the
    session and reused for every subsequent run in the same session."""

    def test_anchor_survives_two_runs_in_same_session(self):
        agent = _make_agent()
        _stub_model_response(agent)

        first = "Build a CLI tool for parsing nginx logs"
        second = "Now also count 4xx by path"

        asyncio.run(agent.run(message=first))
        anchor_after_turn_1 = agent.task_anchor
        self.assertIsNotNone(anchor_after_turn_1)
        self.assertEqual(anchor_after_turn_1.source_query, first)

        asyncio.run(agent.run(message=second))
        anchor_after_turn_2 = agent.task_anchor

        # Same object identity AND same source_query: turn-2 input must NOT
        # overwrite the original goal. This is the regression Issue B caught.
        self.assertIs(anchor_after_turn_1, anchor_after_turn_2)
        self.assertEqual(anchor_after_turn_2.source_query, first)

        # run_context for turn-2 must reference the same anchor object.
        self.assertIs(agent.run_context.task_anchor, anchor_after_turn_1)

    def test_session_change_resets_anchor(self):
        agent = _make_agent(session_id="session-A")
        _stub_model_response(agent)

        asyncio.run(agent.run(message="task A goal"))
        anchor_a = agent.task_anchor
        self.assertEqual(anchor_a.source_query, "task A goal")

        # Switch to a fresh session: a new conversation is allowed to
        # establish its own original task.
        agent.session_id = "session-B"
        asyncio.run(agent.run(message="task B goal"))
        anchor_b = agent.task_anchor

        self.assertIsNot(anchor_a, anchor_b)
        self.assertEqual(anchor_b.source_query, "task B goal")

    def test_subagent_run_records_parent_run_id(self):
        """Phase 0 lineage: a child agent spawned with `_parent_run_id` set
        must record it on its RunContext and switch source to subagent."""
        child = _make_agent(name="child-agent", session_id="child-session")
        _stub_model_response(child)
        child._parent_run_id = "parent-run-xyz"

        asyncio.run(child.run(message="do the child task"))

        self.assertIsNotNone(child.run_context)
        self.assertEqual(child.run_context.parent_run_id, "parent-run-xyz")
        self.assertEqual(child.run_context.source, RunSource.subagent)


class TestEventCallbackFailureIsolation(unittest.TestCase):
    """Issue A: event bus failures must be loud-but-not-fatal."""

    def test_failing_event_callback_does_not_crash_run(self):
        agent = _make_agent()
        _stub_model_response(agent)

        def boom(_event_dict):
            raise RuntimeError("event bus is on fire")

        agent._event_callback = boom

        with self.assertLogs("agentica", level="WARNING") as cm:
            response = asyncio.run(agent.run(message="hello"))

        self.assertEqual(response.content, "ok")
        joined = "\n".join(cm.output)
        # WARNING (not DEBUG) and includes the event type so the operator
        # can find which callback misbehaved.
        self.assertIn("event callback failed", joined.lower())
        self.assertIn("run.started", joined)


class TestLearningReportWriteFailureVisible(unittest.TestCase):
    """I5 / Issue D: silent learning-report failures violate observability."""

    def test_learning_report_write_failure_logs_warning(self):
        from agentica.learning_report import write_learning_report

        # A workspace stub whose reports dir resolves but whose md write fails.
        class _BrokenWorkspace:
            def __init__(self, path: Path):
                self._path = path

            def get_user_learning_reports_dir(self) -> Path:
                return self._path

        with tempfile.TemporaryDirectory() as td:
            ws = _BrokenWorkspace(Path(td))
            report = LearningReport(run_id="r-fail", agent_id="a-1")

            # Force md write to fail at the OS level.
            with patch.object(Path, "write_text", side_effect=OSError("disk full")):
                with self.assertLogs("agentica", level="WARNING") as cm:
                    result = write_learning_report(ws, report)

        self.assertIsNone(result)
        joined = "\n".join(cm.output)
        self.assertIn("learning report md write failed", joined)
        self.assertIn("r-fail.md", joined)


class TestMemoryCandidateReviewAPI(unittest.TestCase):
    """I6: list / promote / reject round-trip for quarantined memories."""

    def test_promote_moves_candidate_into_canonical_memory(self):
        async def _run():
            with tempfile.TemporaryDirectory() as td:
                ws = Workspace(td)
                ws.initialize()

                cand_path = await ws.write_memory_entry(
                    title="user_pref",
                    content="User MIGHT prefer dark mode",
                    memory_type="user",
                    source="auto_extract",
                    evidence_refs=["chat#123"],
                )
                self.assertIn("memory_candidates", cand_path)

                listing = ws.list_memory_candidates()
                self.assertEqual(len(listing), 1)
                self.assertEqual(listing[0]["name"], "user_pref")
                self.assertEqual(listing[0]["evidence_refs"], ["chat#123"])

                promoted = await ws.promote_memory_candidate(listing[0]["filename"])
                self.assertIsNotNone(promoted)
                self.assertNotIn("memory_candidates", promoted)

                # Original candidate file is gone (M7 cleanup).
                self.assertFalse(Path(cand_path).exists())
                self.assertEqual(ws.list_memory_candidates(), [])

                # Index now mentions the promoted entry.
                idx = ws._get_user_memory_md().read_text(encoding="utf-8")
                self.assertIn("user_pref", idx)

                # Frontmatter on the canonical entry kept evidence_refs
                # round-tripped through JSON.
                text = Path(promoted).read_text(encoding="utf-8")
                self.assertIn('evidence_refs: ["chat#123"]', text)
                self.assertIn("source: user_confirmed", text)

        asyncio.run(_run())

    def test_reject_deletes_candidate_and_is_idempotent(self):
        async def _run():
            with tempfile.TemporaryDirectory() as td:
                ws = Workspace(td)
                ws.initialize()
                await ws.write_memory_entry(
                    title="trash",
                    content="ignore me",
                    memory_type="reference",
                    source="auto_extract",
                )
                fname = ws.list_memory_candidates()[0]["filename"]
                self.assertTrue(ws.reject_memory_candidate(fname))
                self.assertEqual(ws.list_memory_candidates(), [])
                # Second reject is a no-op, not an error.
                self.assertFalse(ws.reject_memory_candidate(fname))

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
