# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unit tests for SkillTool

Tests cover:
1. SkillTool initialization with empty skill directories
2. list_skills() with no skills available
3. get_skill_info() with non-existent skill
4. get_system_prompt() with empty registry
5. Custom skill directory handling
"""
import os
import tempfile
import shutil
import unittest
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentica.tools.skill_tool import SkillTool
from agentica.skills.skill_registry import reset_skill_registry


class TestSkillToolEmpty(unittest.TestCase):
    """Test SkillTool behavior when skill directories are empty."""

    def setUp(self):
        """Reset skill registry before each test."""
        reset_skill_registry()

    def tearDown(self):
        """Clean up after each test."""
        reset_skill_registry()

    def test_init_with_empty_dirs(self):
        """Test SkillTool can be created even when skill directories are empty."""
        # Should not raise any exception
        skill_tool = SkillTool()
        self.assertIsNotNone(skill_tool)
        self.assertEqual(skill_tool.name, "skill_tool")

    def test_init_with_auto_load_false(self):
        """Test SkillTool with auto_load=False."""
        skill_tool = SkillTool(auto_load=False)
        self.assertIsNotNone(skill_tool)
        # Registry should be empty
        self.assertEqual(len(skill_tool.registry), 0)

    def test_list_skills_empty(self):
        """Test list_skills() returns appropriate message when no skills available."""
        skill_tool = SkillTool(auto_load=False)
        result = skill_tool.list_skills()
        
        self.assertIn("No skills available", result)
        self.assertIn(".claude/skills", result)
        self.assertIn(".agentica/skills", result)

    def test_get_skill_info_nonexistent(self):
        """Test get_skill_info() with non-existent skill returns error message."""
        skill_tool = SkillTool(auto_load=False)
        result = skill_tool.get_skill_info("nonexistent-skill")
        
        self.assertIn("Error", result)
        self.assertIn("not found", result)

    def test_get_system_prompt_empty(self):
        """Test get_system_prompt() returns valid prompt even with no skills."""
        skill_tool = SkillTool(auto_load=False)
        prompt = skill_tool.get_system_prompt()
        
        self.assertIsNotNone(prompt)
        self.assertIn("Skills Tool", prompt)
        self.assertIn("No skills are currently available", prompt)

    def test_repr_empty(self):
        """Test __repr__ works with empty registry."""
        skill_tool = SkillTool(auto_load=False)
        repr_str = repr(skill_tool)
        
        self.assertIn("SkillTool", repr_str)
        self.assertIn("skills=0", repr_str)


class TestSkillToolWithCustomDir(unittest.TestCase):
    """Test SkillTool with custom skill directories."""

    def setUp(self):
        """Create a temporary skill directory."""
        reset_skill_registry()
        self.temp_dir = tempfile.mkdtemp()
        self.skill_dir = os.path.join(self.temp_dir, "test-skill")
        os.makedirs(self.skill_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        reset_skill_registry()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_custom_dir_without_skill_md(self):
        """Test custom skill directory without SKILL.md file."""
        # Directory exists but has no SKILL.md
        skill_tool = SkillTool(
            custom_skill_dirs=[self.skill_dir],
            auto_load=False
        )
        
        # Should not crash, just log warning
        self.assertIsNotNone(skill_tool)
        self.assertEqual(len(skill_tool.registry), 0)

    def test_custom_dir_with_valid_skill(self):
        """Test custom skill directory with valid SKILL.md file."""
        # Create a valid SKILL.md
        skill_md_content = """---
name: test-skill
description: A test skill for unit testing
---

# Test Skill

This is a test skill for unit testing purposes.

## Usage

Use this skill for testing.
"""
        skill_md_path = os.path.join(self.skill_dir, "SKILL.md")
        with open(skill_md_path, "w") as f:
            f.write(skill_md_content)

        skill_tool = SkillTool(
            custom_skill_dirs=[self.skill_dir],
            auto_load=False
        )

        # Should have loaded the skill
        self.assertEqual(len(skill_tool.registry), 1)
        
        # Test get_skill_info
        result = skill_tool.get_skill_info("test-skill")
        self.assertIn("test-skill", result)
        self.assertIn("Description", result)
        
        # Test get_system_prompt includes skill instructions
        prompt = skill_tool.get_system_prompt()
        self.assertIn("test-skill", prompt)
        self.assertIn("instructions", prompt.lower())

    def test_custom_dir_nonexistent(self):
        """Test custom skill directory that doesn't exist."""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent")
        
        # Should not crash
        skill_tool = SkillTool(
            custom_skill_dirs=[nonexistent_dir],
            auto_load=False
        )
        
        self.assertIsNotNone(skill_tool)
        self.assertEqual(len(skill_tool.registry), 0)

    def test_add_skill_dir_at_runtime(self):
        """Test adding skill directory at runtime."""
        skill_tool = SkillTool(auto_load=False)
        self.assertEqual(len(skill_tool.registry), 0)

        # Create a valid SKILL.md
        skill_md_content = """---
name: runtime-skill
description: A skill added at runtime
---

# Runtime Skill

Added dynamically.
"""
        skill_md_path = os.path.join(self.skill_dir, "SKILL.md")
        with open(skill_md_path, "w") as f:
            f.write(skill_md_content)

        # Add skill at runtime
        skill = skill_tool.add_skill_dir(self.skill_dir)
        
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "runtime-skill")
        self.assertEqual(len(skill_tool.registry), 1)


class TestSkillToolIntegration(unittest.TestCase):
    """Integration tests for SkillTool with DeepAgent."""

    def setUp(self):
        """Reset skill registry."""
        reset_skill_registry()

    def tearDown(self):
        """Clean up."""
        reset_skill_registry()

    def test_skill_tool_with_deep_agent_empty_skills(self):
        """Test SkillTool works with DeepAgent when no skills are available."""
        # Create SkillTool with no skills
        skill_tool = SkillTool(auto_load=False)
        
        # This should not crash even with empty skills
        # Note: We don't actually run the agent, just verify initialization
        self.assertIsNotNone(skill_tool)
        self.assertEqual(len(skill_tool.registry), 0)
        
        # Verify system prompt is valid
        prompt = skill_tool.get_system_prompt()
        self.assertIsNotNone(prompt)


if __name__ == "__main__":
    unittest.main()
