# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the opencode-style subagent loader
(``agentica/subagent_loader.py``).

The loader scans ``.agentica/agents/*.md`` files (YAML frontmatter + Markdown
body) and registers each as a custom subagent. It must be fail-soft: a single
malformed file is skipped with a warning and never blocks startup. These tests
isolate the loader on a tmp directory by patching ``get_search_paths`` /
``_resolve_target_dir`` and clean the module-global custom registry afterwards.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentica.subagent_loader as loader
from agentica.subagent import (
    get_custom_subagent_configs,
    unregister_custom_subagent,
)


class _LoaderTestCase(unittest.TestCase):
    """Common harness: tmp agent dir + custom-registry cleanup."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.agents_dir = Path(self._tmp.name) / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self._registered: list = []
        self._search_patch = patch(
            "agentica.subagent_loader.get_search_paths",
            return_value=[str(self.agents_dir)],
        )
        self._search_patch.start()

    def tearDown(self):
        self._search_patch.stop()
        for name in self._registered:
            unregister_custom_subagent(name)
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> Path:
        path = self.agents_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        return path

    def _track(self, name: str) -> None:
        if name not in self._registered:
            self._registered.append(name)


class TestParseValidFrontmatter(_LoaderTestCase):
    def test_parse_valid_frontmatter(self):
        """A well-formed file is registered and listed."""
        self._write(
            "reviewer",
            "---\n"
            "description: Reviews code for quality and bugs\n"
            "allowed_tools: [read_file, ls, glob, grep]\n"
            "tool_call_limit: 10\n"
            "---\n"
            "You are a code review expert.\n",
        )
        count = loader.load_all_agents()
        self.assertEqual(count, 1)
        configs = get_custom_subagent_configs()
        self.assertIn("reviewer", configs)
        self._track("reviewer")
        cfg = configs["reviewer"]
        self.assertEqual(cfg.description, "Reviews code for quality and bugs")
        self.assertEqual(set(cfg.allowed_tools), {"read_file", "ls", "glob", "grep"})

        defined = loader.list_defined_agents()
        names = [d["name"] for d in defined]
        self.assertIn("reviewer", names)
        reviewer = next(d for d in defined if d["name"] == "reviewer")
        self.assertEqual(reviewer["tool_call_limit"], 10)


class TestSkipMalformed(_LoaderTestCase):
    def test_missing_description_skipped(self):
        """A file without a description is skipped (fail-soft, no raise)."""
        self._write(
            "nodesc",
            "---\nallowed_tools: [read_file]\n---\nbody text\n",
        )
        count = loader.load_all_agents()
        self.assertEqual(count, 0)
        self.assertNotIn("nodesc", get_custom_subagent_configs())

    def test_empty_body_skipped(self):
        """A file with no body content is skipped (fail-soft, no raise)."""
        self._write(
            "nobody",
            "---\ndescription: foo\n---\n",
        )
        # Whitespace-only body is equally skipped.
        self._write(
            "nobodyws",
            "---\ndescription: foo\n---\n   \n",
        )
        count = loader.load_all_agents()
        self.assertEqual(count, 0)
        self.assertNotIn("nobody", get_custom_subagent_configs())
        self.assertNotIn("nobodyws", get_custom_subagent_configs())

    def test_bad_yaml_skipped(self):
        """Invalid YAML frontmatter is skipped without raising."""
        self._write(
            "badyaml",
            "---\ndescription: [unclosed\n---\nbody text\n",
        )
        count = loader.load_all_agents()
        self.assertEqual(count, 0)
        self.assertNotIn("badyaml", get_custom_subagent_configs())

    def test_no_frontmatter_skipped(self):
        """A file without any frontmatter block is skipped without raising."""
        self._write("nofm", "Just some prose, no frontmatter at all.\n")
        count = loader.load_all_agents()
        self.assertEqual(count, 0)
        self.assertNotIn("nofm", get_custom_subagent_configs())


class TestCreateAgentFile(_LoaderTestCase):
    def test_create_agent_file_writes_and_registers(self):
        """create_agent_file writes a valid .md and registers the subagent."""
        tmp_write = Path(self._tmp.name) / "write_target"
        tmp_write.mkdir(parents=True, exist_ok=True)
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=tmp_write,
        ):
            path = loader.create_agent_file(
                name="summarizer",
                description="Summarizes documents",
                system_prompt="You summarize documents faithfully.",
                allowed_tools=["read_file", "grep"],
                tool_call_limit=5,
            )
        self._track("summarizer")

        self.assertTrue(os.path.isfile(path))
        text = open(path, encoding="utf-8").read()
        self.assertIn("description: Summarizes documents", text)
        self.assertIn("read_file", text)
        self.assertIn("You summarize documents faithfully.", text)
        # Registered in the live registry.
        configs = get_custom_subagent_configs()
        self.assertIn("summarizer", configs)
        self.assertEqual(
            set(configs["summarizer"].allowed_tools), {"read_file", "grep"}
        )

    def test_create_agent_file_rejects_bad_name(self):
        """Path separators and traversal in the name are rejected."""
        with self.assertRaises(ValueError):
            loader.create_agent_file(
                name="evil/../x",
                description="d",
                system_prompt="p",
            )
        with self.assertRaises(ValueError):
            loader.create_agent_file(
                name="a/b",
                description="d",
                system_prompt="p",
            )
        # Empty / whitespace names are rejected too.
        with self.assertRaises(ValueError):
            loader.create_agent_file(name="  ", description="d", system_prompt="p")


class TestRemoveAgentFile(_LoaderTestCase):
    def test_remove_agent_file(self):
        """remove_agent_file deletes the file and unregisters the subagent."""
        # Write into the patched search-path dir so remove_agent_file finds it.
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=self.agents_dir,
        ):
            path = loader.create_agent_file(
                name="removeme",
                description="temporary agent",
                system_prompt="body",
            )
        self.assertTrue(os.path.isfile(path))
        self.assertIn("removeme", get_custom_subagent_configs())
        self._track("removeme")

        removed = loader.remove_agent_file("removeme")
        self.assertTrue(removed)
        self.assertFalse(os.path.isfile(path))
        self.assertNotIn("removeme", get_custom_subagent_configs())
        # Already unregistered; drop from cleanup list so tearDown doesn't re-unregister.
        if "removeme" in self._registered:
            self._registered.remove("removeme")

    def test_remove_agent_file_missing_returns_false(self):
        self.assertFalse(loader.remove_agent_file("does_not_exist_xyz"))


class TestOuterGuard(unittest.TestCase):
    """load_all_agents must never propagate errors to the CLI startup."""

    def test_load_all_agents_outer_guard(self):
        """An exploding get_search_paths still yields an int, not an exception."""
        with patch(
            "agentica.subagent_loader.get_search_paths",
            side_effect=RuntimeError("boom"),
        ):
            result = loader.load_all_agents()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
