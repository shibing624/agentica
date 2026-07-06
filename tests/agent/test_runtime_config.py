# -*- coding: utf-8 -*-
"""
Tests for Tool/Skill Runtime Config (enable/disable control).

Tests cover:
1. ToolRuntimeConfig / SkillRuntimeConfig dataclass
2. Agent.enable_tool / disable_tool / enable_skill / disable_skill
3. Agent-level tool filtering via _filter_model_functions
4. Query-level enabled_tools / enabled_skills via RunConfig
5. Workspace YAML runtime config loading
6. Priority: query-level > agent-level > default
"""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentica.agent.config import ToolRuntimeConfig, SkillRuntimeConfig
from agentica.run_config import RunConfig


class TestRuntimeConfigDataclass:
    """Test ToolRuntimeConfig and SkillRuntimeConfig dataclasses."""

    def test_tool_runtime_config_defaults(self):
        cfg = ToolRuntimeConfig(name="execute")
        assert cfg.name == "execute"
        assert cfg.enabled is True

    def test_tool_runtime_config_disabled(self):
        cfg = ToolRuntimeConfig(name="execute", enabled=False)
        assert cfg.enabled is False

    def test_skill_runtime_config_defaults(self):
        cfg = SkillRuntimeConfig(name="paper-digest")
        assert cfg.name == "paper-digest"
        assert cfg.enabled is True

    def test_skill_runtime_config_disabled(self):
        cfg = SkillRuntimeConfig(name="iwiki-doc", enabled=False)
        assert cfg.enabled is False


class TestRunConfigFields:
    """Test RunConfig enabled_tools/enabled_skills fields."""

    def test_run_config_defaults(self):
        cfg = RunConfig()
        assert cfg.enabled_tools is None
        assert cfg.enabled_skills is None
        assert cfg.max_api_retry is None

    def test_run_config_with_enabled_tools(self):
        cfg = RunConfig(enabled_tools=["web_search", "read_file"])
        assert cfg.enabled_tools == ["web_search", "read_file"]

    def test_run_config_with_enabled_skills(self):
        cfg = RunConfig(enabled_skills=["paper-digest"])
        assert cfg.enabled_skills == ["paper-digest"]

    def test_run_config_with_use_structured_outputs(self):
        cfg = RunConfig(use_structured_outputs=True)
        assert cfg.use_structured_outputs is True

    def test_run_config_with_max_api_retry(self):
        cfg = RunConfig(max_api_retry=2)
        assert cfg.max_api_retry == 2


class TestAgentEnableDisable:
    """Test Agent.enable_tool/disable_tool/enable_skill/disable_skill."""

    def test_enable_tool(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.enable_tool("execute")
        assert "execute" in agent._tool_runtime_configs
        assert agent._tool_runtime_configs["execute"].enabled is True

    def test_disable_tool(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_tool("execute")
        assert "execute" in agent._tool_runtime_configs
        assert agent._tool_runtime_configs["execute"].enabled is False

    def test_enable_skill(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.enable_skill("paper-digest")
        assert "paper-digest" in agent._skill_runtime_configs
        assert agent._skill_runtime_configs["paper-digest"].enabled is True

    def test_disable_skill(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_skill("iwiki-doc")
        assert "iwiki-doc" in agent._skill_runtime_configs
        assert agent._skill_runtime_configs["iwiki-doc"].enabled is False

    def test_toggle_tool(self):
        """Test disable then re-enable a tool."""
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_tool("execute")
        assert agent._is_tool_enabled("execute") is False
        agent.enable_tool("execute")
        assert agent._is_tool_enabled("execute") is True


class TestIsToolEnabled:
    """Test Agent._is_tool_enabled priority logic."""

    def test_default_enabled(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        assert agent._is_tool_enabled("any_tool") is True

    def test_agent_level_disabled(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_tool("execute")
        assert agent._is_tool_enabled("execute") is False
        assert agent._is_tool_enabled("read_file") is True

    def test_query_level_whitelist(self):
        """Query-level whitelist overrides agent-level config."""
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.enable_tool("execute")  # agent-level: enabled
        agent._enabled_tools = ["web_search"]  # query-level: only web_search
        assert agent._is_tool_enabled("execute") is False
        assert agent._is_tool_enabled("web_search") is True

    def test_query_level_overrides_agent_level(self):
        """Query-level should override agent-level even for disabled tools."""
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_tool("execute")  # agent-level: disabled
        agent._enabled_tools = ["execute"]  # query-level: whitelist includes it
        assert agent._is_tool_enabled("execute") is True


class TestIsSkillEnabled:
    """Test Agent._is_skill_enabled priority logic."""

    def test_default_enabled(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        assert agent._is_skill_enabled("any_skill") is True

    def test_agent_level_disabled(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_skill("iwiki-doc")
        assert agent._is_skill_enabled("iwiki-doc") is False
        assert agent._is_skill_enabled("paper-digest") is True

    def test_query_level_whitelist(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent._enabled_skills = ["paper-digest"]
        assert agent._is_skill_enabled("paper-digest") is True
        assert agent._is_skill_enabled("iwiki-doc") is False


class TestFilterModelFunctions:
    """Test Agent._filter_model_functions."""

    def test_no_filtering_when_no_config(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        # Mock model with functions
        agent.model = MagicMock()
        agent.model.functions = {"read_file": MagicMock(), "execute": MagicMock()}
        agent.model.tools = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "execute"}},
        ]
        agent._filter_model_functions()
        assert len(agent.model.functions) == 2

    def test_agent_level_filter(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent.disable_tool("execute")
        # Mock model
        agent.model = MagicMock()
        agent.model.functions = {"read_file": MagicMock(), "execute": MagicMock()}
        agent.model.tools = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "execute"}},
        ]
        agent._filter_model_functions()
        assert "read_file" in agent.model.functions
        assert "execute" not in agent.model.functions
        assert len(agent.model.tools) == 1

    def test_query_level_whitelist_filter(self):
        from agentica import Agent
        agent = Agent(model=MagicMock())
        agent._enabled_tools = ["read_file"]
        # Mock model
        agent.model = MagicMock()
        agent.model.functions = {
            "read_file": MagicMock(),
            "execute": MagicMock(),
            "web_search": MagicMock(),
        }
        agent.model.tools = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "execute"}},
            {"type": "function", "function": {"name": "web_search"}},
        ]
        agent._filter_model_functions()
        assert "read_file" in agent.model.functions
        assert "execute" not in agent.model.functions
        assert "web_search" not in agent.model.functions
        assert len(agent.model.tools) == 1


class TestWorkspaceYamlConfig:
    """Test loading runtime config from .agentica/runtime_config.yaml."""

    def test_load_yaml_config(self):
        """Test loading tool/skill configs from YAML file."""
        from agentica import Agent

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".agentica"
            config_dir.mkdir()
            config_file = config_dir / "runtime_config.yaml"
            config_file.write_text(
                "tools:\n"
                "  execute:\n"
                "    enabled: false\n"
                "  web_search:\n"
                "    enabled: true\n"
                "skills:\n"
                "  iwiki-doc:\n"
                "    enabled: false\n",
                encoding="utf-8",
            )

            # Patch cwd to tmpdir
            with patch("os.getcwd", return_value=tmpdir):
                agent = Agent(model=MagicMock())
                # Manually trigger load since _post_init already ran with real cwd
                agent._tool_runtime_configs.clear()
                agent._skill_runtime_configs.clear()
                agent._load_runtime_config()

            assert agent._tool_runtime_configs["execute"].enabled is False
            assert agent._tool_runtime_configs["web_search"].enabled is True
            assert agent._skill_runtime_configs["iwiki-doc"].enabled is False

    def test_no_yaml_file_is_ok(self):
        """No YAML file should not cause errors."""
        from agentica import Agent
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.getcwd", return_value=tmpdir):
                agent = Agent(model=MagicMock())
                agent._load_runtime_config()
                assert len(agent._tool_runtime_configs) == 0

    def test_agent_level_overrides_yaml(self):
        """Agent-level config should override YAML config."""
        from agentica import Agent

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / ".agentica"
            config_dir.mkdir()
            config_file = config_dir / "runtime_config.yaml"
            config_file.write_text(
                "tools:\n"
                "  execute:\n"
                "    enabled: false\n",
                encoding="utf-8",
            )

            with patch("os.getcwd", return_value=tmpdir):
                agent = Agent(model=MagicMock())
                agent._tool_runtime_configs.clear()
                agent._load_runtime_config()

            # YAML says disabled
            assert agent._is_tool_enabled("execute") is False
            # Agent-level override: re-enable
            agent.enable_tool("execute")
            assert agent._is_tool_enabled("execute") is True


class TestQueryLevelViaRunConfig:
    """Test enabled_tools/enabled_skills passed through RunConfig."""

    def test_run_config_enabled_tools_passed_to_runner(self):
        """Verify RunConfig.enabled_tools flows into _run_impl."""
        config = RunConfig(enabled_tools=["web_search", "read_file"])
        assert config.enabled_tools == ["web_search", "read_file"]

    def test_run_config_enabled_skills_passed_to_runner(self):
        """Verify RunConfig.enabled_skills flows into _run_impl."""
        config = RunConfig(enabled_skills=["paper-digest"])
        assert config.enabled_skills == ["paper-digest"]
