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

    def test_first_correction_costs_no_call(self):
        new = self._write("Run the linter before opening a pull request")
        with patch.object(OpenAIChat, "response", new=AsyncMock()) as m:
            found = asyncio.run(check_rule_conflicts(
                self.model, self.store, new, "Run the linter before opening a pull request",
            ))
        self.assertEqual(found, [])
        m.assert_not_awaited()

    def test_cross_language_pair_is_still_offered(self):
        old = self._write("回复一律用中文")
        new = self._write("Always answer in English")
        reply = ModelResponse(content=f'{{"conflicts": ["{os.path.basename(old)}"]}}')
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=reply)) as m:
            found = asyncio.run(check_rule_conflicts(self.model, self.store, new, "Always answer in English"))
        self.assertEqual(found, [os.path.basename(old)])
        self.assertIn("回复一律用中文", m.await_args.kwargs["messages"][1].content)

    def test_failed_call_marks_nothing(self):
        self._write("Use pip to install python packages")
        new = self._write("Never use pip for python packages, use uv instead")
        with patch.object(OpenAIChat, "response", new=AsyncMock(side_effect=ConnectionError("down"))):
            found = asyncio.run(check_rule_conflicts(self.model, self.store, new, "Never use pip, use uv"))
        self.assertEqual(found, [])
        with open(new, encoding="utf-8") as f:
            self.assertNotIn("conflict_with", f.read())

    def test_chinese_rules_get_cards_and_their_conflict_is_marked(self):
        old = self._write("以后都用 pip 装依赖")
        new = self._write("不要用 pip 装依赖，改用 uv")
        self.assertNotEqual(old, new, "a rule and its reversal must not share a card")
        old_name = os.path.basename(old)
        reply = ModelResponse(content=f'{{"conflicts": ["{old_name}"]}}')
        with patch.object(OpenAIChat, "response", new=AsyncMock(return_value=reply)) as m:
            found = asyncio.run(check_rule_conflicts(self.model, self.store, new, "不要用 pip 装依赖，改用 uv"))
        self.assertEqual(found, [old_name])
        self.assertIn("以后都用 pip 装依赖", m.await_args.kwargs["messages"][1].content)
        prompt = asyncio.run(self.store.get_relevant("pip 装依赖", limit=5))
        self.assertIn("Conflicts with:", prompt)

    def test_shortlist_uses_shared_terms_including_cjk(self):
        cards = [
            {"filename": "a.md", "title": "a", "rule": "回复一律用中文"},
            {"filename": "b.md", "title": "b", "rule": "Use pip"},
        ]
        self.assertEqual([c["filename"] for c in shortlist_rivals("Use uv not pip", cards)], ["b.md", "a.md"])
        self.assertEqual([c["filename"] for c in shortlist_rivals("回复请用英文", cards)], ["a.md", "b.md"])
        many = [{"filename": f"{i}.md", "title": "", "rule": f"rule {i}"} for i in range(9)]
        self.assertEqual(len(shortlist_rivals("anything", many)), 5)
        self.assertIn("pip", rule_terms("Never use pip"))


class TestChineseCorrectionKey(unittest.TestCase):
    def test_rewordings_collide_and_reversals_do_not(self):
        key = ExperienceCompiler.correction_key_from_rule
        self.assertEqual(key("以后请先跑测试再提交"), key("务必先跑测试再提交"))
        self.assertEqual(key("不要用pip装依赖"), key("别用 pip 装依赖"))
        self.assertNotEqual(key("用 pip 装依赖"), key("不要用 pip 装依赖"))
        self.assertEqual(key("以后都用 uv 装依赖"), "uv_用装依赖")
        self.assertEqual(key("的了请"), "", "all filler is still refused")

    def test_english_keys_are_unchanged(self):
        self.assertEqual(
            ExperienceCompiler.correction_key_from_rule("Always list the directory before reading files"),
            "list_directory_before_read",
        )

    def test_prefiltered_chinese_rule_is_persisted(self):
        from agentica.agent.hooks import ExperienceCaptureHooks

        result = ExperienceCaptureHooks._prefilter_feedback_classification("下次请先跑测试再提交")
        self.assertIsNotNone(result)
        card = ExperienceCompiler.compile_correction(result)
        self.assertIsNotNone(card)
        self.assertEqual(card.correction_key, "先跑测试再提交")


if __name__ == "__main__":
    unittest.main()
