# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the unified 3-tier tool permission model
(agentica.agent.permissions), and its wiring through Agent/DeepAgent.

No tier hides write or network tools: every tier exposes the full schema;
the runner parks before execute instead.
"""
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agentica.agent import Agent
from agentica.agent.config import SandboxConfig, ToolConfig
from agentica.agent.permissions import (
    PERMISSION_MODES,
    sandbox_should_be_enabled,
    validate_permission_mode,
)


class TestPermissionHelpers(unittest.TestCase):
    def test_validate_permission_mode_accepts_known_modes(self):
        for mode in PERMISSION_MODES:
            validate_permission_mode(mode)  # must not raise

    def test_validate_permission_mode_rejects_unknown(self):
        with self.assertRaises(ValueError):
            validate_permission_mode("yolo")

    def test_validate_permission_mode_rejects_the_retired_ask(self):
        """The tier was renamed; a saved "ask" must fail loud, not mean something new."""
        with self.assertRaises(ValueError):
            validate_permission_mode("ask")

    def test_sandbox_should_be_enabled(self):
        self.assertTrue(sandbox_should_be_enabled("read-only"))
        self.assertTrue(sandbox_should_be_enabled("auto"))
        self.assertFalse(sandbox_should_be_enabled("allow-all"))


class TestToolConfigPermissionMode(unittest.TestCase):
    def test_default_is_allow_all(self):
        self.assertEqual(ToolConfig().permission_mode, "allow-all")

    def test_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            ToolConfig(permission_mode="strict")


class TestAgentPermissionMode(unittest.TestCase):
    """Plain Agent() must default to today's unrestricted behavior."""

    def test_default_agent_has_allow_all_and_concrete_sandbox_config(self):
        agent = Agent()
        self.assertEqual(agent.tool_config.permission_mode, "allow-all")
        self.assertIsInstance(agent.sandbox_config, SandboxConfig)
        self.assertFalse(agent.sandbox_config.enabled)

    def test_explicit_sandbox_config_is_not_overridden(self):
        custom = SandboxConfig(enabled=True, blocked_commands=["rm -rf /"])
        agent = Agent(sandbox_config=custom)
        self.assertIs(agent.sandbox_config, custom)

    def test_read_only_mode_exposes_write_tools(self):
        agent = Agent(tool_config=ToolConfig(permission_mode="read-only"))
        self.assertTrue(agent._is_tool_enabled("read_file"))
        self.assertTrue(agent._is_tool_enabled("write_file"))
        self.assertTrue(agent._is_tool_enabled("execute"))
        self.assertTrue(agent._is_tool_enabled("apply_patch"))

    def test_read_only_mode_keeps_memory_tools(self):
        """No tier hides writes: save_memory stays visible."""
        agent = Agent(tool_config=ToolConfig(permission_mode="read-only"))
        self.assertTrue(agent._is_tool_enabled("search_memory"))
        self.assertTrue(agent._is_tool_enabled("save_memory"))

    def test_auto_mode_enables_sandbox_and_allows_all_tools(self):
        agent = Agent(tool_config=ToolConfig(permission_mode="auto"))
        self.assertTrue(agent.sandbox_config.enabled)
        self.assertTrue(agent._is_tool_enabled("write_file"))
        self.assertTrue(agent._is_tool_enabled("apply_patch"))

    def test_auto_mode_seeds_writable_dirs_with_work_dir(self):
        """Regression: SandboxConfig only enforces work_dir as a fallback when
        writable_dirs is non-empty (see BuiltinFileTool._validate_write_path).
        An enabled sandbox with an empty writable_dirs list is a silent no-op."""
        agent = Agent(work_dir="/tmp/some_work_dir", tool_config=ToolConfig(permission_mode="auto"))
        self.assertIn("/tmp/some_work_dir", agent.sandbox_config.writable_dirs)

    def test_set_permission_mode_switches_at_runtime(self):
        agent = Agent()  # allow-all
        agent.set_permission_mode("read-only")
        self.assertEqual(agent.tool_config.permission_mode, "read-only")
        self.assertTrue(agent.sandbox_config.enabled)
        self.assertTrue(agent._is_tool_enabled("write_file"))

        agent.set_permission_mode("allow-all")
        self.assertFalse(agent.sandbox_config.enabled)
        self.assertTrue(agent._is_tool_enabled("write_file"))

    def test_set_permission_mode_rejects_unknown_mode(self):
        agent = Agent()
        with self.assertRaises(ValueError):
            agent.set_permission_mode("full")

    def test_query_level_enabled_tools_overrides_permission_mode(self):
        """RunConfig.enabled_tools (query-level) still wins over permission_mode."""
        agent = Agent(tool_config=ToolConfig(permission_mode="read-only"))
        agent._enabled_tools = ["write_file"]
        self.assertTrue(agent._is_tool_enabled("write_file"))
        self.assertFalse(agent._is_tool_enabled("read_file"))

    def test_approve_defaults_none_and_clone_copies_callback(self):
        async def cb(fc):
            return "allow"

        agent = Agent(approve=cb)
        self.assertIs(agent.approve, cb)
        clone = agent.clone()
        self.assertIs(clone.approve, cb)
        self.assertIsNone(Agent().approve)


class TestDeepAgentPermissionMode(unittest.TestCase):
    def _build(self, **kwargs):
        from agentica.agent.deep import DeepAgent

        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch("agentica.agent.base.Agent._merge_tool_system_prompts"):
            return DeepAgent(model=MagicMock(), workspace=tmpdir, **kwargs)

    def test_default_permission_mode_is_allow_all(self):
        agent = self._build()
        self.assertEqual(agent.tool_config.permission_mode, "allow-all")
        self.assertFalse(agent.sandbox_config.enabled)

    def test_read_only_mode_exposes_write_tools_at_construction(self):
        agent = self._build(permission_mode="read-only")
        self.assertEqual(agent.tool_config.permission_mode, "read-only")
        self.assertTrue(agent._is_tool_enabled("write_file"))
        self.assertTrue(agent._is_tool_enabled("apply_patch"))
        self.assertTrue(agent._is_tool_enabled("read_file"))

    def test_auto_mode_enables_sandbox(self):
        agent = self._build(permission_mode="auto")
        self.assertTrue(agent.sandbox_config.enabled)

    def test_auto_mode_seeds_writable_dirs_with_work_dir(self):
        with tempfile.TemporaryDirectory() as work_dir, patch(
            "agentica.agent.base.Agent._load_mcp_tools"
        ), patch("agentica.agent.base.Agent._merge_tool_system_prompts"):
            from agentica.agent.deep import DeepAgent

            agent = DeepAgent(model=MagicMock(), work_dir=work_dir, workspace=work_dir, permission_mode="auto")
        self.assertIn(work_dir, agent.sandbox_config.writable_dirs)

    def test_invalid_permission_mode_raises(self):
        with self.assertRaises(ValueError):
            self._build(permission_mode="full")

    def test_set_permission_mode_mutates_shared_sandbox_instance(self):
        """The SandboxConfig handed to builtin file tools at construction is
        the SAME object as agent.sandbox_config, so runtime mode switches
        propagate to already-built tools without rebuilding the Agent."""
        agent = self._build(permission_mode="allow-all", include_execute=False)
        from agentica.tools.builtin import BuiltinFileTool

        file_tool = next(t for t in agent.tools if isinstance(t, BuiltinFileTool))
        self.assertIs(file_tool._sandbox_config, agent.sandbox_config)

        agent.set_permission_mode("auto")
        self.assertTrue(file_tool._sandbox_config.enabled)
        self.assertNotIn("request_path_access", file_tool.functions)

        agent.set_permission_mode("allow-all")
        self.assertNotIn("request_path_access", file_tool.functions)

    def test_request_path_access_removed_from_schema(self):
        allow = self._build()
        ask = self._build(permission_mode="read-only")
        auto = self._build(permission_mode="auto")
        from agentica.tools.builtin import BuiltinFileTool

        def names(agent):
            tool = next(t for t in agent.tools if isinstance(t, BuiltinFileTool))
            return set(tool.functions)

        self.assertNotIn("request_path_access", names(allow))
        self.assertNotIn("request_path_access", names(ask))
        self.assertNotIn("request_path_access", names(auto))


if __name__ == "__main__":
    unittest.main()
