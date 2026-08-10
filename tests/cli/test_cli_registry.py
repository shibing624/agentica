# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for CLI module.
"""

import logging
import os
import re
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


class TestHelpRendering(unittest.TestCase):
    """`/help` renders command names verbatim, brackets included.

    A key like "/model [p/m]" parses as a rich style tag, so an unescaped
    render drops the placeholder without any error.
    """

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _render_help(self, skills_registry=None) -> str:
        from rich.console import Console
        from agentica.cli.display import help_header

        buffer = StringIO()
        console = Console(file=buffer, width=120, highlight=False)
        with patch.object(help_header, "get_console", lambda: console):
            help_header.show_help(skills_registry=skills_registry)
        return self.ANSI_RE.sub("", buffer.getvalue())

    def test_bracketed_placeholders_survive(self):
        out = self._render_help()

        for cmd in ("/model [p/m]", "/resume [target]", "/fork [n|uuid]", "/debug [on|off]"):
            self.assertIn(cmd, out)

    def test_columns_stay_aligned(self):
        """Escaping must not shift the description column."""
        lines = self._render_help().splitlines()

        def desc_column(prefix, desc):
            line = next(ln for ln in lines if ln.strip().startswith(prefix))
            return line.index(desc)

        self.assertEqual(
            desc_column("/model [", "Show or switch model"),
            desc_column("/config", "Show current configuration"),
        )

    def test_skill_description_with_brackets_is_not_swallowed(self):
        skill = Mock()
        skill.description = "Handles [urgent] tickets"
        registry = MagicMock()
        registry.__len__.return_value = 1
        registry.auto_commands.return_value = {"/triage": skill}

        out = self._render_help(skills_registry=registry)

        self.assertIn("Handles [urgent] tickets", out)


if __name__ == "__main__":
    unittest.main()
