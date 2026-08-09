# -*- coding: utf-8 -*-
"""Tests for Skill lazy loading and standard frontmatter discovery."""
import tempfile
import unittest
import importlib
import os
from pathlib import Path
from unittest.mock import patch

import agentica.config as agentica_config
import agentica.workspace as workspace_module
from agentica.skills.skill import Skill
from agentica.skills.skill_loader import SkillLoader
from agentica.skills.skill_registry import SkillRegistry


class TestSkillLazyLoading(unittest.TestCase):
    """Skill content should be lazy-loaded from SKILL.md."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.skill_dir = Path(self.tmpdir) / "test-skill"
        self.skill_dir.mkdir()
        (self.skill_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: A test skill\n---\n# Full content here\nDetailed instructions...",
            encoding="utf-8",
        )

    def test_from_skill_md_does_not_load_content(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        self.assertIsNotNone(skill)
        # Content should not be loaded yet
        self.assertFalse(skill._content_loaded)

    def test_content_access_triggers_load(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        self.assertFalse(skill._content_loaded)
        content = skill.content
        self.assertTrue(skill._content_loaded)
        self.assertIn("Full content", content)
        self.assertIn("Detailed instructions", content)

    def test_second_access_returns_cached(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        content1 = skill.content
        content2 = skill.content
        self.assertEqual(content1, content2)

    def test_content_setter(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        skill.content = "Custom content"
        self.assertTrue(skill._content_loaded)
        self.assertEqual(skill.content, "Custom content")

    def test_to_dict_includes_content(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        d = skill.to_dict()
        self.assertIn("content", d)
        self.assertIn("Full content", d["content"])

    def test_get_prompt_uses_lazy_content(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        prompt = skill.get_prompt()
        self.assertIn("Loading: test", prompt)
        self.assertIn("Full content", prompt)

    def test_invalidate_content_reloads_on_next_access(self):
        skill = Skill.from_skill_md(self.skill_dir / "SKILL.md")
        # First load
        self.assertIn("Full content", skill.content)
        self.assertTrue(skill._content_loaded)

        # Modify file on disk
        (self.skill_dir / "SKILL.md").write_text(
            "---\nname: test\ndescription: A test skill\n---\n# Updated content\nNew instructions here.",
            encoding="utf-8",
        )

        # Still cached
        self.assertIn("Full content", skill.content)

        # Invalidate and re-access
        skill.invalidate_content()
        self.assertFalse(skill._content_loaded)
        self.assertIn("Updated content", skill.content)
        self.assertIn("New instructions", skill.content)
        self.assertTrue(skill._content_loaded)


class TestConfigurableMemoryBudget(unittest.TestCase):
    """Workspace should consume the configurable AGENTS context budget."""

    def test_memory_character_budget_can_be_configured(self):
        with patch.dict(os.environ, {"AGENTICA_MAX_MEMORY_CHARACTER_COUNT": "12345"}, clear=False):
            config = importlib.reload(agentica_config)
            workspace = importlib.reload(workspace_module)
            self.assertEqual(config.AGENTICA_MAX_MEMORY_CHARACTER_COUNT, 12345)
            self.assertEqual(workspace.Workspace.MAX_MEMORY_CHARACTER_COUNT, 12345)

        importlib.reload(agentica_config)
        importlib.reload(workspace_module)


class TestSkillDescriptionIsTheDiscoveryField(unittest.TestCase):
    """Agent Skills discover via description; when_to_use is not a field."""

    def test_legacy_when_to_use_frontmatter_is_ignored(self):
        tmpdir = tempfile.mkdtemp()
        skill_dir = Path(tmpdir) / "search-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: arxiv-search\n"
            "description: Search papers. Use when the user mentions arxiv or academic search.\n"
            "when_to_use: arxiv, papers\n"
            "when-to-use: ignored-too\n"
            "---\n# Instructions",
            encoding="utf-8",
        )
        skill = Skill.from_skill_md(skill_dir / "SKILL.md")
        self.assertIsNotNone(skill)
        self.assertNotIn("when_to_use", skill.__dataclass_fields__)
        self.assertNotIn("when_to_use", skill.to_dict(include_content=False))
        self.assertNotIn("<when_to_use>", skill.to_xml())
        self.assertIn("arxiv", skill.description)

    def test_match_trigger_requires_explicit_trigger(self):
        registry = SkillRegistry()
        registry.register(Skill(
            name="paper-search",
            description="Search papers. Use for arxiv and academic search.",
            path=Path("/tmp"),
            user_invocable=True,
        ))
        self.assertIsNone(registry.match_trigger("search for arxiv papers"))

        registry.register(Skill(
            name="commit",
            description="Git commit helper",
            path=Path("/tmp"),
            trigger="/commit",
            user_invocable=True,
        ))
        matched = registry.match_trigger("/commit fix bug")
        self.assertIsNotNone(matched)
        self.assertEqual(matched.name, "commit")
        self.assertIsNone(registry.match_trigger("hello world"))


class TestSkillLoaderManagedDirs(unittest.TestCase):
    """Managed skill dirs can point directly at a single external skill directory."""

    def test_search_paths_deduplicate_same_user_skill_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            user_skill_dir = Path(tmpdir) / ".agentica" / "skills"
            user_skill_dir.mkdir(parents=True)

            with patch("agentica.skills.skill_loader.AGENTICA_SKILL_DIR", str(user_skill_dir)):
                loader = SkillLoader(project_root=Path(tmpdir))
                loader.home_dir = Path(tmpdir)
                paths = loader.get_search_paths()

            normalized = [path.resolve() for path, _location in paths]
            assert normalized.count(user_skill_dir.resolve()) == 1

    def test_load_all_accepts_direct_skill_dir_from_managed_paths(self):
        tmpdir = tempfile.mkdtemp()
        skill_dir = Path(tmpdir) / "learn-from-experience"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: learn-from-experience\ndescription: Learns from corrections\n---\n# Body",
            encoding="utf-8",
        )

        with patch("agentica.skills.skill_loader.AGENTICA_EXTRA_SKILL_PATHS", [str(skill_dir)]):
            registry = SkillLoader(project_root=Path(tmpdir)).load_all(SkillRegistry())

        skill = registry.get("learn-from-experience")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "learn-from-experience")

    def test_extra_skill_path_env_supports_multiple_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_value = (
                f"{tmpdir}/skills/learn-from-experience"
                f"{os.pathsep}"
                f"{tmpdir}/shared/skills"
            )
            with patch.dict(os.environ, {"AGENTICA_EXTRA_SKILL_PATH": env_value}, clear=False):
                config = importlib.reload(agentica_config)
                self.assertEqual(
                    config.AGENTICA_EXTRA_SKILL_PATHS,
                    [
                        f"{tmpdir}/skills/learn-from-experience",
                        f"{tmpdir}/shared/skills",
                    ],
                )


if __name__ == "__main__":
    unittest.main()
