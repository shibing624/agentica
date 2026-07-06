# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for the unified 3-tier tool permission model
(agentica.agent.permissions), and its wiring through Agent/DeepAgent.
"""
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agentica.agent import Agent
from agentica.agent.config import SandboxConfig, ToolConfig
from agentica.agent.permissions import (
    PERMISSION_MODES,
    READ_ONLY_TOOLS,
    read_only_whitelist,
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

    def test_read_only_whitelist_for_ask(self):
        self.assertEqual(set(read_only_whitelist("ask")), READ_ONLY_TOOLS)

    def test_read_only_whitelist_none_for_auto_and_allow_all(self):
        self.assertIsNone(read_only_whitelist("auto"))
        self.assertIsNone(read_only_whitelist("allow-all"))

    def test_sandbox_should_be_enabled(self):
        self.assertTrue(sandbox_should_be_enabled("ask"))
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

    def test_ask_mode_hides_write_tools(self):
        agent = Agent(tool_config=ToolConfig(permission_mode="ask"))
        self.assertTrue(agent._is_tool_enabled("read_file"))
        self.assertFalse(agent._is_tool_enabled("write_file"))

    def test_auto_mode_enables_sandbox_and_allows_all_tools(self):
        agent = Agent(tool_config=ToolConfig(permission_mode="auto"))
        self.assertTrue(agent.sandbox_config.enabled)
        self.assertTrue(agent._is_tool_enabled("write_file"))

    def test_auto_mode_seeds_writable_dirs_with_work_dir(self):
        """Regression: SandboxConfig only enforces work_dir as a fallback when
        writable_dirs is non-empty (see BuiltinFileTool._validate_write_path).
        An enabled sandbox with an empty writable_dirs list is a silent no-op."""
        agent = Agent(work_dir="/tmp/some_work_dir", tool_config=ToolConfig(permission_mode="auto"))
        self.assertIn("/tmp/some_work_dir", agent.sandbox_config.writable_dirs)

    def test_set_permission_mode_switches_at_runtime(self):
        agent = Agent()  # allow-all
        agent.set_permission_mode("ask")
        self.assertEqual(agent.tool_config.permission_mode, "ask")
        self.assertTrue(agent.sandbox_config.enabled)
        self.assertFalse(agent._is_tool_enabled("write_file"))

        agent.set_permission_mode("allow-all")
        self.assertFalse(agent.sandbox_config.enabled)
        self.assertTrue(agent._is_tool_enabled("write_file"))

    def test_set_permission_mode_rejects_unknown_mode(self):
        agent = Agent()
        with self.assertRaises(ValueError):
            agent.set_permission_mode("full")

    def test_query_level_enabled_tools_overrides_permission_mode(self):
        """RunConfig.enabled_tools (query-level) still wins over permission_mode."""
        agent = Agent(tool_config=ToolConfig(permission_mode="ask"))
        agent._enabled_tools = ["write_file"]
        self.assertTrue(agent._is_tool_enabled("write_file"))
        self.assertFalse(agent._is_tool_enabled("read_file"))


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

    def test_ask_mode_rejects_write_tools_at_construction(self):
        agent = self._build(permission_mode="ask")
        self.assertEqual(agent.tool_config.permission_mode, "ask")
        self.assertFalse(agent._is_tool_enabled("write_file"))
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
        from agentica.tools.buildin_tools import BuiltinFileTool

        file_tool = next(t for t in agent.tools if isinstance(t, BuiltinFileTool))
        self.assertIs(file_tool._sandbox_config, agent.sandbox_config)

        agent.set_permission_mode("auto")
        self.assertTrue(file_tool._sandbox_config.enabled)


if __name__ == "__main__":
    unittest.main()
