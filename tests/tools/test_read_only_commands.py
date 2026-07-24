# -*- coding: utf-8 -*-
"""Read-only shell policy for subagents.

Two layers are covered:
1. ``is_read_only_command`` — the command classifier itself, including the
   bypass routes (compound commands, substitution, redirection).
2. ``SubagentRegistry._select_child_tools`` — that the ``read_only`` policy
   actually wraps ``execute`` and leaves the parent's Function untouched.

All tests mock the OpenAI key — no real LLM calls.
"""
import asyncio
import unittest

from agentica.subagent import (
    DEFAULT_SUBAGENT_CONFIGS,
    SubagentConfig,
    SubagentRegistry,
    SubagentType,
)
from agentica.tools.buildin_tools import BuiltinExecuteTool
from agentica.tools.safety import is_read_only_command


class TestIsReadOnlyCommand(unittest.TestCase):
    def assert_allowed(self, command: str):
        allowed, reason = is_read_only_command(command)
        self.assertTrue(allowed, f"{command!r} should be read-only, got: {reason}")

    def assert_refused(self, command: str):
        allowed, _ = is_read_only_command(command)
        self.assertFalse(allowed, f"{command!r} should be refused")

    def test_git_inspection_commands_allowed(self):
        for command in [
            "git diff",
            "git diff HEAD~1 -- agentica/",
            "git log --oneline -20",
            "git show abc123",
            "git status",
            "git blame agentica/subagent.py",
            "git -C /tmp/repo status",
        ]:
            self.assert_allowed(command)

    def test_git_mutating_subcommands_refused(self):
        for command in [
            "git commit -m msg",
            "git push --force",
            "git checkout -b feature",
            "git reset --hard",
            "git stash",
            "git fetch origin",
        ]:
            self.assert_refused(command)

    def test_test_runners_allowed(self):
        for command in [
            "pytest tests/ -v",
            "python -m pytest tests/ -k foo",
            "npm test",
            "npm run lint",
            "cargo test",
            "go vet ./...",
            "mypy agentica/",
        ]:
            self.assert_allowed(command)

    def test_runner_write_variants_refused(self):
        for command in ["ruff check --fix .", "cargo fmt", "eslint --fix src/"]:
            self.assert_refused(command)

    def test_case_insensitive_flag_is_not_treated_as_in_place_edit(self):
        """`-i` means "ignore case" to git, not "in place"."""
        for command in [
            "git grep -i TODO",
            "git log -i --grep=fix",
            "git diff -i",
        ]:
            self.assert_allowed(command)

    def test_package_installs_refused(self):
        for command in ["npm install", "npm publish", "yarn", "pip install requests"]:
            self.assert_refused(command)

    def test_arbitrary_interpreters_refused(self):
        for command in [
            "python script.py",
            "python3 -c 'import shutil'",
            "bash deploy.sh",
            "sh -c ls",
        ]:
            self.assert_refused(command)

    def test_compound_command_checks_every_segment(self):
        """A benign first token must not smuggle a mutating second segment."""
        self.assert_allowed("ls -la && git status")
        self.assert_refused("git log && rm -rf build")
        self.assert_refused("echo ok; git reset --hard")
        self.assert_refused("git diff | tee out.txt")

    def test_command_substitution_refused(self):
        self.assert_refused("echo $(rm -rf x)")
        self.assert_refused("echo `whoami`")

    def test_redirection_refused_but_devnull_allowed(self):
        self.assert_refused("git diff > patch.txt")
        self.assert_refused("git log >> log.txt")
        self.assert_allowed("pytest tests/ 2>&1")
        self.assert_allowed("git log > /dev/null")

    def test_empty_command_refused(self):
        self.assert_refused("")
        self.assert_refused("   ")


class TestSubagentExecutePolicy(unittest.TestCase):
    def setUp(self):
        self.registry = SubagentRegistry()
        self.tool = BuiltinExecuteTool(work_dir="/tmp")

    def _child_execute(self, policy: str):
        config = SubagentConfig(
            type=SubagentType.CUSTOM,
            name="t",
            description="d",
            system_prompt="p",
            allowed_tools=["execute"],
            execute_policy=policy,
        )
        child_tools = self.registry._select_child_tools([self.tool], config)
        self.assertEqual(len(child_tools), 1)
        return child_tools[0].functions["execute"]

    def test_read_only_policy_refuses_mutating_command(self):
        execute_fn = self._child_execute("read_only")
        result = asyncio.run(execute_fn.entrypoint(command="git commit -m x"))
        self.assertIn("Refused", result)
        self.assertIn("read-only", result)

    def test_read_only_policy_runs_inspection_command(self):
        execute_fn = self._child_execute("read_only")
        result = asyncio.run(execute_fn.entrypoint(command="echo hello"))
        self.assertIn("hello", result)

    def test_read_only_policy_does_not_mutate_parent_function(self):
        """The parent agent shares this Function object — it must stay unguarded."""
        self._child_execute("read_only")
        parent_fn = self.tool.functions["execute"]
        self.assertFalse(parent_fn.is_read_only)
        result = asyncio.run(parent_fn.entrypoint(command="echo parent-unrestricted"))
        self.assertIn("parent-unrestricted", result)

    def test_inherit_policy_leaves_execute_untouched(self):
        execute_fn = self._child_execute("inherit")
        self.assertIs(execute_fn, self.tool.functions["execute"])

    def test_read_only_execute_is_marked_and_described(self):
        execute_fn = self._child_execute("read_only")
        self.assertTrue(execute_fn.is_read_only)
        self.assertFalse(execute_fn.is_destructive)
        self.assertIn("READ-ONLY", execute_fn.description)

    def test_builtin_subagents_expose_read_only_execute(self):
        for subagent_type in (
            SubagentType.EXPLORE,
            SubagentType.RESEARCH,
            SubagentType.CODE,
            SubagentType.REVIEW,
        ):
            config = DEFAULT_SUBAGENT_CONFIGS[subagent_type]
            self.assertEqual(config.execute_policy, "read_only", subagent_type)
            self.assertIn("execute", config.allowed_tools, subagent_type)
            self.assertNotIn("execute", config.denied_tools, subagent_type)


class TestSubagentLoaderExecutePolicy(unittest.TestCase):
    """Markdown-defined subagents must be able to opt into the read-only shell."""

    def _parse(self, frontmatter_extra: str):
        import tempfile
        from pathlib import Path

        from agentica.subagent_loader import _parse_agent_file

        body = (
            "---\n"
            "description: d\n"
            f"{frontmatter_extra}"
            "---\n\n"
            "You are a test agent.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checker.md"
            path.write_text(body, encoding="utf-8")
            return _parse_agent_file(path)

    def test_execute_policy_defaults_to_inherit(self):
        self.assertEqual(self._parse("")["execute_policy"], "inherit")

    def test_execute_policy_read_only_is_parsed(self):
        descriptor = self._parse("execute_policy: read_only\n")
        self.assertEqual(descriptor["execute_policy"], "read_only")

    def test_invalid_execute_policy_falls_back_to_inherit(self):
        descriptor = self._parse("execute_policy: bogus\n")
        self.assertEqual(descriptor["execute_policy"], "inherit")


if __name__ == "__main__":
    unittest.main()
