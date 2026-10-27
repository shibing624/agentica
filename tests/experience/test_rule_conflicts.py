# -*- coding: utf-8 -*-
"""A new correction that contradicts a stored one is marked on both cards."""
import asyncio
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

from agentica.experience.compiler import ExperienceCompiler, rule_terms
from agentica.experience.conflicts import check_rule_conflicts, shortlist_rivals
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.workspace import Workspace


def _card(rule: str):
    return ExperienceCompiler.compile_correction({
        "is_correction": True, "should_persist": True, "persist_target": "experience",
        "rule": rule, "why": "user said so", "how_to_apply": "always", "category": "tool_usage",
        "confidence": 0.95, "scope": "cross_session",
    })


class TestRuleConflicts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = Workspace(self.tmp).get_compiled_experience_store()
        self.model = OpenAIChat(id="gpt-4o", api_key="fake_openai_key")

    def _write(self, rule):
        return asyncio.run(self.store.write(_card(rule)))

    def test_contradiction_marks_both_cards_and_prompt_says_so(self):
        old = self._write("Use pip to install python packages")
        new = self._write("Never use pip for python packages, use uv instead")
        old_name, new_name = os.path.basename(old), os.path.basename(new)
        reply = ModelResponse(content=f'{{"conflicts": ["{old_name}"]}}')
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=reply)) as m:
            found = asyncio.run(check_rule_conflicts(
                self.model, self.store, new, "Never use pip for python packages, use uv instead",
            ))
        self.assertEqual(found, [old_name])
        self.assertIn("Use pip to install", m.await_args.kwargs["messages"][1].content)
        for path, other in ((old, new_name), (new, old_name)):
            with open(path, encoding="utf-8") as f:
                self.assertIn(f'conflict_with: ["{other}"]', f.read())
        prompt = asyncio.run(self.store.get_relevant("pip", limit=5))
        self.assertIn("Conflicts with:", prompt)
        self.assertIn("confirm with the user", prompt)

    def test_mark_survives_a_bump_of_the_same_card(self):
        old = self._write("Use pip to install python packages")
        new = self._write("Never use pip for python packages, use uv instead")
        asyncio.run(self.store.mark_conflict(os.path.basename(old), os.path.basename(new)))
        self._write("Use pip to install python packages")
        with open(old, encoding="utf-8") as f:
            raw = f.read()
        self.assertIn("repeat_count: 2", raw)
        self.assertIn("conflict_with:", raw)

    def test_unrelated_rules_cost_no_call(self):
        self._write("Write commit messages in one line")
        new = self._write("Run the linter before opening a pull request")
        with patch.object(OpenAIChat, "response", new=AsyncMock()) as m:
            found = asyncio.run(check_rule_conflicts(
                self.model, self.store, new, "Run the linter before opening a pull request",
            ))
        self.assertEqual(found, [])
        m.assert_not_awaited()

    def test_failed_call_marks_nothing(self):
        self._write("Use pip to install python packages")
        new = self._write("Never use pip for python packages, use uv instead")
        with patch.object(OpenAIChat, "response", new=AsyncMock(side_effect=ConnectionError("down"))):
            found = asyncio.run(check_rule_conflicts(self.model, self.store, new, "Never use pip, use uv"))
        self.assertEqual(found, [])
        with open(new, encoding="utf-8") as f:
            self.assertNotIn("conflict_with", f.read())

    def test_shortlist_uses_shared_terms_including_cjk(self):
        cards = [
            {"filename": "a.md", "title": "a", "rule": "回复一律用中文"},
            {"filename": "b.md", "title": "b", "rule": "Use pip"},
        ]
        self.assertEqual([c["filename"] for c in shortlist_rivals("回复请用英文", cards)], ["a.md"])
        self.assertIn("pip", rule_terms("Never use pip"))


if __name__ == "__main__":
    unittest.main()
