# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.cli import (
    TOOL_ICONS,
    TOOL_REGISTRY,
)


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
        expected_icons = ["read_file", "write_file", "execute", "web_search"]
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
        for tool_name, (module_name, class_name) in TOOL_REGISTRY.items():
            self.assertIsInstance(tool_name, str)
            self.assertIsInstance(module_name, str)
            self.assertIsInstance(class_name, str)

    def test_common_tools_registered(self):
        """Test common tools are registered."""
        expected_tools = ["arxiv", "duckduckgo", "wikipedia"]
        for tool in expected_tools:
            self.assertIn(tool, TOOL_REGISTRY)


class TestCLIHelpers(unittest.TestCase):
    """Test cases for CLI helper functions."""

    def test_tool_icon_lookup(self):
        """Test looking up tool icons."""
        # Test existing icon
        icon = TOOL_ICONS.get("read_file", TOOL_ICONS["default"])
        self.assertIsNotNone(icon)

        # Test default fallback
        icon = TOOL_ICONS.get("nonexistent_tool", TOOL_ICONS["default"])
        self.assertEqual(icon, TOOL_ICONS["default"])


class TestCLIImports(unittest.TestCase):
    """Test cases for CLI module imports."""

    def test_can_import_cli_module(self):
        """Test CLI module can be imported."""
        try:
            from agentica import cli
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import cli module: {e}")

    def test_can_import_deep_agent(self):
        """Test DeepAgent can be imported from CLI."""
        try:
            from agentica import DeepAgent
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import DeepAgent: {e}")


class TestCLIConfiguration(unittest.TestCase):
    """Test cases for CLI configuration."""

    def test_history_file_path(self):
        """Test history file path is set."""
        from agentica.cli import history_file
        self.assertIsInstance(history_file, str)
        self.assertTrue(history_file.endswith("cli_history.txt"))


class TestToolRegistryIntegrity(unittest.TestCase):
    """Test cases for tool registry integrity."""

    def test_all_tools_have_valid_module_names(self):
        """Test all tools have valid module names."""
        for tool_name, (module_name, class_name) in TOOL_REGISTRY.items():
            # Module name should not be empty
            self.assertTrue(len(module_name) > 0, f"Empty module name for {tool_name}")
            # Class name should not be empty
            self.assertTrue(len(class_name) > 0, f"Empty class name for {tool_name}")
            # Class name should be PascalCase (start with uppercase)
            self.assertTrue(
                class_name[0].isupper(),
                f"Class name {class_name} should start with uppercase"
            )

    def test_no_duplicate_tools(self):
        """Test no duplicate tool names in registry."""
        tool_names = list(TOOL_REGISTRY.keys())
        self.assertEqual(len(tool_names), len(set(tool_names)))


if __name__ == "__main__":
    unittest.main()
