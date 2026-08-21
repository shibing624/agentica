# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the opencode-style subagent loader
(``agentica/subagent_loader.py``).

The loader scans package, user, and project ``agents/*.md`` files. User-authored
files fail softly, while malformed package defaults fail loudly.
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
)


class _LoaderTestCase(unittest.TestCase):
    """Common harness: tmp agent dir + custom-registry cleanup."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.agents_dir = Path(self._tmp.name) / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self._search_patch = patch(
            "agentica.subagent_loader.get_search_locations",
            return_value=[loader.AgentSearchLocation(self.agents_dir, "project")],
        )
        self._search_patch.start()

    def tearDown(self):
        self._search_patch.stop()
        loader.load_all_agents()
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> Path:
        path = self.agents_dir / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        return path


class TestParseValidFrontmatter(_LoaderTestCase):
    def test_parse_valid_frontmatter(self):
        """A well-formed file is registered and listed."""
        self._write(
            "summarizer",
            "---\n"
            "description: Summarizes source files\n"
            "allowed_tools: [read_file, glob, grep]\n"
            "tool_call_limit: 10\n"
            "---\n"
            "You are a source summarization expert.\n",
        )
        count = loader.load_all_agents()
        self.assertEqual(count, 1)
        configs = get_custom_subagent_configs()
        self.assertIn("summarizer", configs)
        cfg = configs["summarizer"]
        self.assertEqual(cfg.description, "Summarizes source files")
        self.assertEqual(set(cfg.allowed_tools), {"read_file", "glob", "grep"})

        defined = loader.list_defined_agents()
        names = [d["id"] for d in defined]
        self.assertIn("summarizer", names)
        summarizer = next(d for d in defined if d["id"] == "summarizer")
        self.assertEqual(summarizer["tool_call_limit"], 10)

    def test_parse_all_runtime_fields(self):
        path = self._write(
            "specialist",
            "---\n"
            "name: Specialist Agent\n"
            "description: Does focused work\n"
            "allowed_tools: [read_file]\n"
            "denied_tools: [task]\n"
            "execute_policy: read_only\n"
            "model_tier: main\n"
            "tool_call_limit: 7\n"
            "max_turns: 42\n"
            "timeout: 90\n"
            "can_spawn_subagents: true\n"
            "inherit_context: true\n"
            "inherit_workspace: true\n"
            "inherit_knowledge: true\n"
            "---\n"
            "Do the focused task.\n",
        )

        descriptor = loader._parse_agent_file(path, source="project")

        self.assertEqual(descriptor["name"], "Specialist Agent")
        self.assertEqual(descriptor["model_tier"], "main")
        self.assertEqual(descriptor["max_turns"], 42)
        self.assertEqual(descriptor["timeout"], 90)
        self.assertTrue(descriptor["can_spawn_subagents"])
        self.assertTrue(descriptor["inherit_context"])
        self.assertTrue(descriptor["inherit_workspace"])
        self.assertTrue(descriptor["inherit_knowledge"])

    def test_empty_allowed_tools_means_no_tools(self):
        path = self._write(
            "isolated",
            "---\ndescription: No tools\nallowed_tools: []\ndenied_tools: []\n---\nPrompt.\n",
        )

        descriptor = loader._parse_agent_file(path)

        self.assertEqual(descriptor["allowed_tools"], [])
        self.assertEqual(descriptor["denied_tools"], [])

    def test_project_definition_overrides_package_definition(self):
        package_dir = Path(self._tmp.name) / "package_agents"
        package_dir.mkdir()
        package_file = package_dir / "explore.md"
        package_file.write_text(
            "---\ndescription: package version\n---\nPackage prompt.\n",
            encoding="utf-8",
        )
        self._write(
            "explore",
            "---\ndescription: project version\n---\nProject prompt.\n",
        )
        with patch(
            "agentica.subagent_loader.get_search_locations",
            return_value=[
                loader.AgentSearchLocation(self.agents_dir, "project"),
                loader.AgentSearchLocation(package_dir, "package", strict=True),
            ],
        ):
            loader.load_all_agents()

        config = get_custom_subagent_configs()["explore"]
        self.assertEqual(config.description, "project version")
        self.assertEqual(config.source, "project")
        self.assertEqual(config.system_prompt, "Project prompt.")


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
        with patch("agentica.subagent_loader.logger.warning") as warning:
            count = loader.load_all_agents()
        self.assertEqual(count, 0)
        self.assertNotIn("nobody", get_custom_subagent_configs())
        self.assertNotIn("nobodyws", get_custom_subagent_configs())
        messages = "\n".join(str(call.args[0]) for call in warning.call_args_list)
        self.assertIn("empty Markdown body", messages)

    def test_malformed_frontmatter_reported_separately(self):
        self._write(
            "malformed",
            "---\ndescription: foo\nPrompt without closing delimiter\n",
        )

        with patch("agentica.subagent_loader.logger.warning") as warning:
            count = loader.load_all_agents()

        self.assertEqual(count, 0)
        self.assertIn("malformed YAML frontmatter block", warning.call_args.args[0])

    def test_uppercase_file_stem_skipped(self):
        self._write(
            "Code",
            "---\ndescription: uppercase\n---\nPrompt.\n",
        )

        with patch("agentica.subagent_loader.logger.warning") as warning:
            count = loader.load_all_agents()

        self.assertEqual(count, 0)
        self.assertIn("file name must be lowercase", warning.call_args.args[0])

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
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=self.agents_dir,
        ):
            path = loader.create_agent_file(
                name="summarizer",
                description="Summarizes documents",
                system_prompt="You summarize documents faithfully.",
                allowed_tools=["read_file", "grep"],
                tool_call_limit=5,
            )
        self.assertTrue(os.path.isfile(path))
        text = open(path, encoding="utf-8").read()
        self.assertIn("description: Summarizes documents", text)
        self.assertIn("read_file", text)
        self.assertIn("You summarize documents faithfully.", text)
        # Registered in the live registry.
        configs = get_custom_subagent_configs()
        self.assertIn("summarizer", configs)
        self.assertEqual(set(configs["summarizer"].allowed_tools), {"read_file", "grep"})

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
        with self.assertRaises(ValueError):
            loader.create_agent_file(name="valid!", description="d", system_prompt="p")

    def test_create_agent_file_preserves_empty_tool_list(self):
        """An explicit empty list means no tools, not inherit all parent tools."""
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=self.agents_dir,
        ):
            path = loader.create_agent_file(
                name="isolated",
                description="No tools",
                system_prompt="Answer without tools.",
                allowed_tools=[],
                denied_tools=[],
            )

        text = Path(path).read_text(encoding="utf-8")
        self.assertIn("allowed_tools: []", text)
        self.assertIn("denied_tools: []", text)
        config = get_custom_subagent_configs()["isolated"]
        self.assertEqual(config.allowed_tools, [])
        self.assertEqual(config.denied_tools, [])

    def test_create_agent_file_writes_all_runtime_fields(self):
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=self.agents_dir,
        ):
            path = loader.create_agent_file(
                name="specialist",
                display_name="Specialist Agent",
                description="Does focused work",
                system_prompt="Do the focused task.",
                allowed_tools=["read_file", "task"],
                denied_tools=[],
                tool_call_limit=7,
                model_tier="main",
                execute_policy="read_only",
                max_turns=42,
                timeout=90,
                can_spawn_subagents=True,
                inherit_context=True,
                inherit_workspace=True,
                inherit_knowledge=True,
            )

        self.assertTrue(Path(path).is_file())
        config = get_custom_subagent_configs()["specialist"]
        self.assertEqual(config.name, "Specialist Agent")
        self.assertEqual(config.model_tier, "main")
        self.assertEqual(config.execute_policy, "read_only")
        self.assertEqual(config.max_turns, 42)
        self.assertEqual(config.timeout, 90)
        self.assertTrue(config.can_spawn_subagents)
        self.assertTrue(config.inherit_context)
        self.assertTrue(config.inherit_workspace)
        self.assertTrue(config.inherit_knowledge)

    def test_create_agent_file_validates_runtime_fields_before_writing(self):
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=self.agents_dir,
        ):
            with self.assertRaisesRegex(ValueError, "model_tier"):
                loader.create_agent_file(
                    name="invalid",
                    description="d",
                    system_prompt="p",
                    model_tier="cheap",
                )

        self.assertFalse((self.agents_dir / "invalid.md").exists())

    def test_create_normalizes_filename_and_remove_is_case_insensitive(self):
        with patch(
            "agentica.subagent_loader._resolve_target_dir",
            return_value=self.agents_dir,
        ):
            path = loader.create_agent_file(
                name="Summarizer",
                description="Summarizes documents",
                system_prompt="Summarize.",
            )

        self.assertEqual(Path(path).name, "summarizer.md")
        self.assertTrue(loader.remove_agent_file("SUMMARIZER"))
        self.assertFalse(Path(path).exists())

    def test_create_agent_file_rejects_unknown_scope(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            loader.create_agent_file(
                name="summarizer",
                description="d",
                system_prompt="p",
                scope="global",
            )


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
        removed = loader.remove_agent_file("removeme")
        self.assertTrue(removed)
        self.assertFalse(os.path.isfile(path))
        self.assertNotIn("removeme", get_custom_subagent_configs())

    def test_remove_agent_file_missing_returns_false(self):
        self.assertFalse(loader.remove_agent_file("does_not_exist_xyz"))


class TestPackageFailures(unittest.TestCase):
    def test_missing_package_directory_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-agents"
            with patch(
                "agentica.subagent_loader.get_search_locations",
                return_value=[loader.AgentSearchLocation(missing, "package", strict=True)],
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Reinstall agentica"):
                    loader.load_all_agents()

    def test_unexpected_discovery_error_propagates(self):
        with patch(
            "agentica.subagent_loader.get_search_locations",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                loader.load_all_agents()

    def test_malformed_package_definition_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp)
            (package_dir / "broken.md").write_text(
                "---\nname: Broken\n---\nPrompt.\n",
                encoding="utf-8",
            )
            with patch(
                "agentica.subagent_loader.get_search_locations",
                return_value=[loader.AgentSearchLocation(package_dir, "package", strict=True)],
            ):
                with self.assertRaisesRegex(ValueError, "missing description"):
                    loader.load_all_agents()

    def test_packaged_defaults_exclude_review(self):
        loader.load_all_agents()
        configs = loader.list_defined_agents()
        package_ids = {descriptor["id"] for descriptor in configs if descriptor["source"] == "package"}
        self.assertEqual(package_ids, {"explore", "research", "code"})


if __name__ == "__main__":
    unittest.main()
