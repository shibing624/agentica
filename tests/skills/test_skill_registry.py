# -*- coding: utf-8 -*-
"""Tests for agentica.skills — Skill dataclass + SkillRegistry."""
import tempfile
import unittest
from pathlib import Path

from agentica.skills.skill import Skill
from agentica.skills.skill_registry import SkillRegistry, get_skill_registry, reset_skill_registry


# ===========================================================================
# Skill dataclass tests
# ===========================================================================

class TestSkillFromSkillMd(unittest.TestCase):
    """Skill.from_skill_md parses YAML frontmatter from SKILL.md."""

    def _write_skill_md(self, tmpdir: str, content: str) -> Path:
        p = Path(tmpdir) / "SKILL.md"
        p.write_text(content, encoding="utf-8")
        return p

    def test_valid_skill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_skill_md(tmpdir, """---
name: Test Skill
description: A test skill
license: MIT
trigger: /test
argument-hint: "<file>"
requires:
  - shell
  - python
allowed-tools:
  - shell
user-invocable: true
is-hidden: false
---

# Test Skill

Instructions here.
""")
            skill = Skill.from_skill_md(path, location="project")
            self.assertIsNotNone(skill)
            self.assertEqual(skill.name, "Test Skill")
            self.assertEqual(skill.description, "A test skill")
            self.assertEqual(skill.license, "MIT")
            self.assertEqual(skill.trigger, "/test")
            self.assertEqual(skill.argument_hint, "<file>")
            self.assertEqual(skill.requires, ["shell", "python"])
            self.assertEqual(skill.allowed_tools, ["shell"])
            self.assertTrue(skill.user_invocable)
            self.assertFalse(skill.is_hidden)
            self.assertEqual(skill.location, "project")
            self.assertIn("Instructions here.", skill.content)

    def test_missing_name_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_skill_md(tmpdir, """---
description: no name
---
body""")
            skill = Skill.from_skill_md(path)
            self.assertIsNone(skill)

    def test_missing_description_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_skill_md(tmpdir, """---
name: NoDesc
---
body""")
            skill = Skill.from_skill_md(path)
            self.assertIsNone(skill)

    def test_no_frontmatter_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_skill_md(tmpdir, "Just plain markdown")
            skill = Skill.from_skill_md(path)
            self.assertIsNone(skill)

    def test_nonexistent_file_returns_none(self):
        skill = Skill.from_skill_md(Path("/nonexistent/SKILL.md"))
        self.assertIsNone(skill)

    def test_minimal_skill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_skill_md(tmpdir, """---
name: Minimal
description: Minimal skill
---
Do stuff.""")
            skill = Skill.from_skill_md(path)
            self.assertIsNotNone(skill)
            self.assertEqual(skill.name, "Minimal")
            self.assertIsNone(skill.trigger)
            self.assertEqual(skill.requires, [])


class TestSkillMethods(unittest.TestCase):
    """Skill instance methods."""

    def _make_skill(self, **kwargs):
        content = kwargs.pop("content", None)
        defaults = dict(
            name="Test", description="Desc",
            path=Path("/tmp/test"),
        )
        defaults.update(kwargs)
        skill = Skill(**defaults)
        if content is not None:
            skill.content = content
        return skill

    def test_matches_trigger(self):
        skill = self._make_skill(trigger="/commit")
        self.assertTrue(skill.matches_trigger("/commit fix bug"))
        self.assertTrue(skill.matches_trigger("/commit"))
        self.assertFalse(skill.matches_trigger("/other"))
        self.assertFalse(skill.matches_trigger("commit"))

    def test_matches_trigger_no_trigger(self):
        skill = self._make_skill(trigger=None)
        self.assertFalse(skill.matches_trigger("/commit"))

    def test_get_prompt(self):
        skill = self._make_skill(name="MySkill", content="Instructions")
        prompt = skill.get_prompt()
        self.assertIn("Loading: MySkill", prompt)
        self.assertIn("Base directory:", prompt)
        self.assertIn("Instructions", prompt)

    def test_to_xml(self):
        skill = self._make_skill(name="MySkill", description="Does stuff", location="user")
        xml = skill.to_xml()
        self.assertIn("<skill>", xml)
        self.assertIn("<name>MySkill</name>", xml)
        self.assertIn("<description>Does stuff</description>", xml)
        self.assertIn("<location>user</location>", xml)

    def test_to_dict(self):
        skill = self._make_skill(name="MySkill", trigger="/my")
        d = skill.to_dict()
        self.assertEqual(d["name"], "MySkill")
        self.assertEqual(d["trigger"], "/my")
        self.assertIn("path", d)

    def test_repr(self):
        skill = self._make_skill(name="MySkill", location="project")
        self.assertIn("MySkill", repr(skill))

    def test_str(self):
        skill = self._make_skill(name="MySkill", description="Does stuff")
        self.assertIn("MySkill", str(skill))
        self.assertIn("Does stuff", str(skill))


# ===========================================================================
# SkillRegistry tests
# ===========================================================================

class TestSkillRegistry(unittest.TestCase):
    """SkillRegistry registration, lookup, and priority."""

    def setUp(self):
        self.registry = SkillRegistry()

    def _make_skill(self, name, location="project", trigger=None,
                    user_invocable=True, is_hidden=False):
        skill = Skill(
            name=name, description=f"Desc for {name}",
            path=Path("/tmp"),
            location=location, trigger=trigger,
            user_invocable=user_invocable, is_hidden=is_hidden,
        )
        skill.content = "body"
        return skill

    def test_register_and_get(self):
        skill = self._make_skill("A")
        self.assertTrue(self.registry.register(skill))
        self.assertIs(self.registry.get("A"), skill)

    def test_exists(self):
        skill = self._make_skill("A")
        self.registry.register(skill)
        self.assertTrue(self.registry.exists("A"))
        self.assertFalse(self.registry.exists("B"))

    def test_priority_project_over_user(self):
        proj = self._make_skill("A", location="project")
        user = self._make_skill("A", location="user")
        self.assertTrue(self.registry.register(proj))
        self.assertFalse(self.registry.register(user), "User should not override project")
        self.assertEqual(self.registry.get("A").location, "project")

    def test_priority_user_overrides_managed(self):
        managed = self._make_skill("A", location="managed")
        user = self._make_skill("A", location="user")
        self.assertTrue(self.registry.register(managed))
        self.assertTrue(self.registry.register(user), "User should override managed")
        self.assertEqual(self.registry.get("A").location, "user")

    def test_list_all(self):
        self.registry.register(self._make_skill("A"))
        self.registry.register(self._make_skill("B"))
        self.assertEqual(len(self.registry.list_all()), 2)

    def test_list_by_location(self):
        self.registry.register(self._make_skill("A", location="project"))
        self.registry.register(self._make_skill("B", location="user"))
        self.assertEqual(len(self.registry.list_by_location("project")), 1)
        self.assertEqual(len(self.registry.list_by_location("user")), 1)

    def test_match_trigger(self):
        self.registry.register(self._make_skill("Commit", trigger="/commit"))
        self.registry.register(self._make_skill("Review", trigger="/review"))
        matched = self.registry.match_trigger("/commit fix bug")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.name, "Commit")

    def test_match_trigger_respects_user_invocable(self):
        self.registry.register(self._make_skill(
            "Hidden", trigger="/hidden", user_invocable=False
        ))
        self.assertIsNone(self.registry.match_trigger("/hidden"))

    def test_match_trigger_no_match(self):
        self.registry.register(self._make_skill("A", trigger="/commit"))
        self.assertIsNone(self.registry.match_trigger("/unknown"))

    def test_list_triggers(self):
        self.registry.register(self._make_skill("A", trigger="/a"))
        self.registry.register(self._make_skill("B", trigger="/b", is_hidden=True))
        self.registry.register(self._make_skill("C", trigger="/c", user_invocable=False))
        triggers = self.registry.list_triggers()
        self.assertIn("/a", triggers)
        self.assertNotIn("/b", triggers)
        self.assertNotIn("/c", triggers)

    def test_remove(self):
        self.registry.register(self._make_skill("A"))
        self.assertTrue(self.registry.remove("A"))
        self.assertFalse(self.registry.exists("A"))
        self.assertFalse(self.registry.remove("A"))

    def test_clear(self):
        self.registry.register(self._make_skill("A"))
        self.registry.register(self._make_skill("B"))
        self.registry.clear()
        self.assertEqual(len(self.registry), 0)

    def test_generate_skills_prompt_empty(self):
        self.assertEqual(self.registry.generate_skills_prompt(), "")

    def test_generate_skills_prompt(self):
        self.registry.register(self._make_skill("A"))
        prompt = self.registry.generate_skills_prompt()
        self.assertIn("<available_skills>", prompt)
        self.assertIn("<skill>", prompt)
        self.assertIn("A", prompt)

    def test_generate_skills_prompt_budget(self):
        for i in range(100):
            self.registry.register(self._make_skill(f"Skill_{i}"))
        prompt = self.registry.generate_skills_prompt(char_budget=200)
        self.assertLessEqual(len(prompt), 500)  # Budget + wrapper

    def test_get_skill_instruction(self):
        self.registry.register(self._make_skill("A", trigger="/a"))
        instr = self.registry.get_skill_instruction()
        self.assertIn("Agent Skills", instr)
        self.assertIn("A", instr)
        self.assertIn("/a", instr)

    def test_get_skills_summary(self):
        self.registry.register(self._make_skill("A"))
        summary = self.registry.get_skills_summary()
        self.assertIn("Available Skills", summary)
        self.assertIn("A", summary)

    def test_dunder_methods(self):
        self.registry.register(self._make_skill("A"))
        self.assertEqual(len(self.registry), 1)
        self.assertIn("A", self.registry)
        self.assertNotIn("B", self.registry)
        items = list(self.registry)
        self.assertEqual(len(items), 1)

    def test_get_skill_by_trigger(self):
        self.registry.register(self._make_skill("A", trigger="/a"))
        skill = self.registry.get_skill_by_trigger("/a")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "A")
        self.assertIsNone(self.registry.get_skill_by_trigger("/none"))


class TestGlobalSkillRegistry(unittest.TestCase):
    """Global singleton get_skill_registry / reset_skill_registry."""

    def test_singleton(self):
        reset_skill_registry()
        r1 = get_skill_registry()
        r2 = get_skill_registry()
        self.assertIs(r1, r2)

    def test_reset(self):
        reset_skill_registry()
        r1 = get_skill_registry()
        reset_skill_registry()
        r2 = get_skill_registry()
        self.assertIsNot(r1, r2)


if __name__ == "__main__":
    unittest.main()
