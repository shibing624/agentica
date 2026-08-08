# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""

import logging
import os
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cost_tracker import CostTracker
from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)
from agentica.cli import commands as cli_commands
from agentica.cli import setup as cli_setup
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.memory.session_log import SessionLog



class TestToolIcons(unittest.TestCase):
    """Test cases for TOOL_ICONS configuration."""

    def test_tool_icons_exists(self):
        """Test TOOL_ICONS dictionary exists."""
        self.assertIsInstance(TOOL_ICONS, dict)

    def test_default_icon_exists(self):
        """Test default icon exists."""
        self.assertIn("default", TOOL_ICONS)

    def test_common_icons_exist(self):
        """Test common tool icons exist."""
        expected_icons = ["read_file", "write_file", "apply_patch", "execute", "web_search"]
        for icon in expected_icons:
            self.assertIn(icon, TOOL_ICONS)

    def test_icons_are_strings(self):
        """Test all icons are strings."""
        for key, value in TOOL_ICONS.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, str)


class TestToolRegistry(unittest.TestCase):
    """Test cases for TOOL_REGISTRY configuration."""

    def test_tool_registry_exists(self):
        """Test TOOL_REGISTRY dictionary exists."""
        self.assertIsInstance(TOOL_REGISTRY, dict)

    def test_registry_format(self):
        """Test registry entries have correct format."""
        for tool_name, (module_name, class_name, category, description) in TOOL_REGISTRY.items():
            self.assertIsInstance(tool_name, str)
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(class_name, str)
            self.assertIsInstance(category, str)
            self.assertIsInstance(description, str)

    def test_common_tools_registered(self):
        """Test common tools are registered."""
        expected_tools = ["arxiv", "duckduckgo", "wikipedia"]
        for tool in expected_tools:
            self.assertIn(tool, TOOL_REGISTRY)


class TestToolRegistryIntegrity(unittest.TestCase):
    """Test cases for tool registry integrity."""

    def test_all_tools_have_valid_module_names(self):
        """Test all tools have valid module names."""
        for tool_name, (module_name, class_name, category, description) in TOOL_REGISTRY.items():
            # Module name should not be empty
            self.assertTrue(len(module_name) > 0, f"Empty module name for {tool_name}")
            # Class name should not be empty
            self.assertTrue(len(class_name) > 0, f"Empty class name for {tool_name}")
            # Class name should be PascalCase (start with uppercase)
            self.assertTrue(class_name[0].isupper(), f"Class name {class_name} should start with uppercase")

    def test_no_duplicate_tools(self):
        """Test no duplicate tool names in registry."""
        tool_names = list(TOOL_REGISTRY.keys())
        self.assertEqual(len(tool_names), len(set(tool_names)))


if __name__ == "__main__":
    unittest.main()
