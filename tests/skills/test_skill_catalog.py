# -*- coding: utf-8 -*-
"""Tests for the budgeted skills catalog renderer."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agentica.skills.catalog import (
    MAX_SKILL_DESCRIPTION_CHARS,
    approx_catalog_tokens,
    cap_skill_description,
    render_skills_catalog,
    skill_catalog_budget,
)
from agentica.skills.skill import Skill
from agentica.skills.skill_registry import SkillRegistry, reset_skill_registry
from agentica.tools.skill_tool import SkillTool


def _skill(name: str, description: str = "", trigger: str = None) -> Skill:
    skill = Skill(
        name=name,
        description=description or f"Desc for {name}",
        path=Path("/tmp"),
        trigger=trigger,
    )
    skill.content = "BODY MUST NOT APPEAR IN CATALOG"
    return skill


class TestSkillCatalogBudget(unittest.TestCase):
    def test_window_percent(self):
        unit, limit = skill_catalog_budget(100_000)
        self.assertEqual(unit, "tokens")
        self.assertEqual(limit, 2_000)

    def test_no_window_falls_back_to_chars(self):
        unit, limit = skill_catalog_budget(None)
        self.assertEqual(unit, "chars")
        self.assertEqual(limit, 8000)

    def test_description_cap_front_loads(self):
        long = "TRIGGER " + ("x" * 2000)
        capped = cap_skill_description(long)
        self.assertEqual(len(capped), MAX_SKILL_DESCRIPTION_CHARS)
        self.assertTrue(capped.startswith("TRIGGER "))
        self.assertTrue(capped.endswith("..."))


class TestRenderSkillsCatalog(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(render_skills_catalog([]), "")

    def test_does_not_inline_skill_body_or_path(self):
        prompt = render_skills_catalog([_skill("review")])
        self.assertIn("**review**", prompt)
        self.assertNotIn("BODY MUST NOT APPEAR IN CATALOG", prompt)
        self.assertNotIn("/tmp", prompt)
        self.assertIn("Do not carry skills across turns unless re-mentioned", prompt)
        self.assertIn("get_skill_info", prompt)

    def test_summary_omits_workflow(self):
        prompt = render_skills_catalog([_skill("review")], include_workflow=False)
        self.assertIn("**review**", prompt)
        self.assertNotIn("get_skill_info", prompt)

    def test_drops_tail_descriptions_before_omitting_names(self):
        skills = [
            _skill("keep-me", "alpha " + ("A" * 400)),
            _skill("maybe", "beta " + ("B" * 400)),
            _skill("tail", "gamma " + ("C" * 400)),
        ]
        # 16k window → 320 tokens, no workflow: 3 full descriptions overflow,
        # 3 name lines do not. Descriptions drop from the tail first.
        prompt = render_skills_catalog(
            skills, context_window=16_000, include_workflow=False
        )
        self.assertIn("keep-me", prompt)
        self.assertIn("maybe", prompt)
        self.assertIn("tail", prompt)
        self.assertLessEqual(approx_catalog_tokens(prompt), 320)
        self.assertIn("**keep-me**", prompt)
        self.assertNotIn("gamma ", prompt)

    def test_omits_tail_when_names_overflow(self):
        skills = [_skill(f"s{i:03d}", "d") for i in range(80)]
        prompt = render_skills_catalog(
            skills, context_window=8_000, include_workflow=False
        )
        self.assertIn("s000", prompt)
        self.assertIn("omitted to fit the catalog budget", prompt)
        self.assertNotIn("s079", prompt)
        self.assertLessEqual(approx_catalog_tokens(prompt), 160)

    def test_registry_summary_uses_catalog(self):
        registry = SkillRegistry()
        registry.register(_skill("A", trigger="/a"))
        summary = registry.get_skills_summary()
        self.assertIn("Available skills", summary)
        self.assertIn("**A**", summary)
        self.assertIn("(`/a`)", summary)
        self.assertNotIn("get_skill_instruction", dir(registry))
        self.assertFalse(hasattr(SkillRegistry, "get_skill_instruction"))
        self.assertFalse(hasattr(SkillRegistry, "generate_skills_prompt"))


class TestSkillToolCatalog(unittest.TestCase):
    def setUp(self):
        reset_skill_registry()

    def tearDown(self):
        reset_skill_registry()

    def test_get_system_prompt_uses_model_window(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        skill_dir = Path(tmp) / "wide"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: wide\ndescription: " + ("Z" * 800) + "\n---\n\n# Body\n",
            encoding="utf-8",
        )
        tool = SkillTool(custom_skill_dirs=[str(skill_dir)], auto_load=False)
        agent = MagicMock()
        agent.model.context_window = 15_000
        tool._agent = agent
        prompt = tool.get_system_prompt()
        self.assertIn("wide", prompt)
        self.assertIn("Do not carry skills across turns unless re-mentioned", prompt)
        self.assertLessEqual(approx_catalog_tokens(prompt), 300)
        self.assertNotIn("# Body", prompt)
        self.assertNotIn("Z" * 40, prompt)


if __name__ == "__main__":
    os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")
    unittest.main()
