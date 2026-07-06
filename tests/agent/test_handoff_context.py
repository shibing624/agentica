# -*- coding: utf-8 -*-
"""
Tests for HandoffContext + default_handoff_mapper.

P1-6: structured context handed off from parent to a Subagent / Swarm
sub-task. Replaces the previous ad-hoc string concatenation in
``Subagent._build_inherited_context``.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from agentica.agent import Agent
from agentica.handoff import HandoffContext, default_handoff_mapper


class TestHandoffContext(unittest.TestCase):
    def test_render_includes_required_sections(self):
        ctx = HandoffContext(
            parent_name="Planner",
            task="Write a unit test for the parser.",
            parent_instructions="Be concise and rigorous.",
            extra_context="Repo uses pytest.",
        )
        rendered = ctx.render()
        self.assertIn("Handoff from Planner", rendered)
        self.assertIn("Write a unit test for the parser.", rendered)
        self.assertIn("Be concise and rigorous.", rendered)
        self.assertIn("Repo uses pytest.", rendered)

    def test_render_omits_empty_sections(self):
        ctx = HandoffContext(parent_name="P", task="T")
        rendered = ctx.render()
        self.assertNotIn("Workspace Summary", rendered)
        self.assertNotIn("Recent History", rendered)
        self.assertNotIn("Additional Context", rendered)
        self.assertIn("Handoff from P", rendered)
        self.assertIn("T", rendered)


class TestDefaultHandoffMapper(unittest.TestCase):
    def test_uses_parent_name_and_instructions(self):
        parent = Agent(name="Coordinator", instructions="Always cite sources.")
        ctx = default_handoff_mapper(parent, task="Find latest figures.")
        self.assertEqual(ctx.parent_name, "Coordinator")
        self.assertEqual(ctx.task, "Find latest figures.")
        self.assertEqual(ctx.parent_instructions, "Always cite sources.")

    def test_handles_list_instructions(self):
        parent = Agent(name="C", instructions=["Step 1", "Step 2"])
        ctx = default_handoff_mapper(parent, task="x")
        self.assertIn("Step 1", ctx.parent_instructions or "")
        self.assertIn("Step 2", ctx.parent_instructions or "")

    def test_handles_callable_instructions(self):
        parent = Agent(name="C", instructions=lambda: "dyn")
        ctx = default_handoff_mapper(parent, task="x")
        # Callable instructions are out of scope for static handoff —
        # mapper must NOT crash and may return None.
        self.assertIsNone(ctx.parent_instructions)

    def test_extra_context_passthrough(self):
        parent = Agent(name="C")
        ctx = default_handoff_mapper(parent, task="t", extra_context="hint!")
        self.assertEqual(ctx.extra_context, "hint!")

    def test_default_parent_name_when_none(self):
        parent = Agent()
        ctx = default_handoff_mapper(parent, task="t")
        self.assertTrue(ctx.parent_name)


class TestSubagentUsesMapper(unittest.TestCase):
    """Smoke test: SubagentRegistry.spawn() routes through default_handoff_mapper."""

    def test_spawn_calls_mapper_when_inherit_context(self):
        from agentica.subagent import SubagentRegistry
        from unittest.mock import patch

        with patch("agentica.subagent.default_handoff_mapper") as mock_mapper:
            mock_mapper.return_value = HandoffContext(
                parent_name="P", task="t", extra_context="ctx-mark"
            )
            # Parent needs a model so spawn() reaches the mapper
            # invocation. Subsequent LLM call is short-circuited by
            # patching the child's run via _run_child_streaming.
            from agentica.model.openai import OpenAIChat
            parent = Agent(
                name="P",
                instructions="parent inst",
                model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            )
            sub = SubagentRegistry()
            with patch.object(
                SubagentRegistry,
                "_run_child_streaming",
                return_value={"content": "ok", "tool_calls_summary": []},
            ):
                import asyncio
                asyncio.run(sub.spawn(parent_agent=parent, task="do thing"))

            mock_mapper.assert_called_once()
            args, kwargs = mock_mapper.call_args
            received = kwargs.get("parent_agent") or (args[0] if args else None)
            self.assertIs(received, parent)


if __name__ == "__main__":
    unittest.main()
