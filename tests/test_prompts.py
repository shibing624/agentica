# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for PromptBuilder and prompt modules.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentica.agent import Agent
from agentica.agent.config import PromptConfig
from agentica.model.openai import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.tools.base import Tool
from agentica.tools.skill_tool import SkillTool


def _mock_resp(content="OK"):
    resp = MagicMock()
    resp.content = content
    resp.parsed = None
    resp.audio = None
    resp.reasoning_content = None
    resp.created_at = None
    return resp


# ===========================================================================
# TestPromptBuilder
# ===========================================================================


class TestPromptBuilder:
    """Tests for PromptBuilder and modular prompt assembly."""

    def test_builder_default_modules(self):
        """PromptBuilder should assemble default modules."""
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_builder_build_includes_soul(self):
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt(enable_soul=True)
        # Soul module should contribute some content
        assert len(result) > 100  # Non-trivial prompt

    def test_builder_selective_modules(self):
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt(
            enable_soul=True,
            enable_heartbeat=False,
            enable_tools_guide=False,
            enable_self_verification=False,
        )
        assert isinstance(result, str)

    def test_builder_with_identity(self):
        from agentica.prompts.builder import PromptBuilder
        result = PromptBuilder.build_system_prompt(identity="You are a coding assistant")
        assert "coding assistant" in result


# ===========================================================================
# TestPromptModules
# ===========================================================================


class TestPromptModules:
    """Tests for individual prompt modules."""

    def test_load_prompt_reads_md(self):
        from agentica.prompts.base.utils import load_prompt
        # Soul prompt should exist as soul.md
        content = load_prompt("soul.md")
        assert isinstance(content, str)
        assert len(content) > 0

    def test_soul_module_content(self):
        from agentica.prompts.base.soul import get_soul_prompt
        content = get_soul_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_tools_module_content(self):
        from agentica.prompts.base.tools import get_tools_prompt
        content = get_tools_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_tools_module_with_active_tools(self):
        from agentica.prompts.base.tools import get_tools_prompt
        content = get_tools_prompt(active_tools=["read_file", "edit_file", "execute"])
        assert isinstance(content, str)
        assert "read_file" in content
        assert "execute" in content

    def test_tools_module_with_descriptions(self):
        from agentica.prompts.base.tools import get_tools_prompt
        descs = {"read_file": "Read a file from disk", "execute": "Run shell commands"}
        content = get_tools_prompt(
            active_tools=["read_file", "execute"],
            tool_descriptions=descs,
        )
        assert "Read a file from disk" in content
        assert "Run shell commands" in content

    def test_builder_with_tool_descriptions(self):
        from agentica.prompts.builder import PromptBuilder
        descs = {"read_file": "Read file content"}
        result = PromptBuilder.build_system_prompt(
            active_tools=["read_file"],
            tool_descriptions=descs,
        )
        assert "Read file content" in result

    def test_heartbeat_module_content(self):
        from agentica.prompts.base.heartbeat import get_heartbeat_prompt
        content = get_heartbeat_prompt()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_self_verification_module_content(self):
        from agentica.prompts.base.self_verification import get_self_verification_prompt
        content = get_self_verification_prompt()
        assert isinstance(content, str)
        assert len(content) > 0


# ===========================================================================
# TestGetSystemMessage
# ===========================================================================


class TestGetSystemMessage:
    """Tests for Agent.get_system_message() prompt construction."""

    @pytest.mark.asyncio
    async def test_system_message_includes_instructions(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            instructions=["Be concise", "Be accurate"],
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "Be concise" in msg.content
        assert "Be accurate" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_includes_datetime(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            prompt_config=PromptConfig(add_datetime_to_instructions=True),
        )
        msg = await agent.get_system_message()
        assert msg is not None
        # Should include date/time info
        content = msg.content
        assert "UTC" in content or "20" in content  # Year prefix

    @pytest.mark.asyncio
    async def test_system_message_with_system_prompt(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            prompt_config=PromptConfig(system_prompt="You are a helpful assistant."),
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "helpful assistant" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_callable_system_prompt(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            prompt_config=PromptConfig(system_prompt=lambda agent=None: "Dynamic system prompt."),
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "Dynamic system prompt" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_with_description(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            description="A test assistant",
        )
        msg = await agent.get_system_message()
        assert msg is not None
        assert "test assistant" in msg.content

    @pytest.mark.asyncio
    async def test_system_message_with_agentic_prompt(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            prompt_config=PromptConfig(enable_agentic_prompt=True),
        )
        msg = await agent.get_system_message()
        assert msg is not None
        # Agentic prompt adds more content from PromptBuilder
        assert len(msg.content) > 100

    @pytest.mark.asyncio
    async def test_system_message_separates_tool_policy_and_session_guidance(self):
        class FakeTool(Tool):
            def get_system_prompt(self):
                return "STATIC TOOL POLICY"

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[FakeTool(name="fake"), SkillTool(auto_load=False)],
        )
        # Agent clones stateful tools per-instance — patch the agent's clone,
        # not the original, so the prompt-merge step sees the dynamic guidance.
        agent_skill_tool = next(t for t in agent.tools if isinstance(t, SkillTool))
        agent_skill_tool.get_system_prompt = lambda: "# Skills\n\nDYNAMIC SKILL GUIDANCE"
        agent.refresh_tool_system_prompts()
        msg = await agent.get_system_message()

        assert msg is not None
        assert "## Tool Usage Guide" in msg.content
        assert "STATIC TOOL POLICY" in msg.content
        assert "## Session Guidance" in msg.content
        assert "DYNAMIC SKILL GUIDANCE" in msg.content
        assert msg.content.index("STATIC TOOL POLICY") < msg.content.index("DYNAMIC SKILL GUIDANCE")

    @pytest.mark.asyncio
    async def test_agentic_prompt_keeps_skill_guidance_dynamic(self):
        class FakeTool(Tool):
            def get_system_prompt(self):
                return "STATIC TOOL POLICY"

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[FakeTool(name="fake"), SkillTool(auto_load=False)],
            prompt_config=PromptConfig(enable_agentic_prompt=True),
        )
        # Patch the agent's per-instance SkillTool clone (see isolation contract).
        agent_skill_tool = next(t for t in agent.tools if isinstance(t, SkillTool))
        agent_skill_tool.get_system_prompt = lambda: "# Skills\n\nDYNAMIC SKILL GUIDANCE"
        agent.refresh_tool_system_prompts()
        msg = await agent.get_system_message()

        assert msg is not None
        assert "## Tool Usage Guide" in msg.content
        assert "STATIC TOOL POLICY" in msg.content
        assert "## Session Guidance" in msg.content
        assert "DYNAMIC SKILL GUIDANCE" in msg.content
        assert msg.content.index("STATIC TOOL POLICY") < msg.content.index("DYNAMIC SKILL GUIDANCE")

    @pytest.mark.asyncio
    async def test_default_prompt_labels_dynamic_blocks(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        )
        agent.add_session_guidance("# Skills\n\nDYNAMIC SKILL GUIDANCE")
        agent.get_workspace_context_prompt = AsyncMock(return_value="# Workspace Rules\n- rule")
        agent.get_workspace_memory_prompt = AsyncMock(return_value="### Memory Rule\nPrefer concise responses.")

        msg = await agent.get_system_message()

        assert msg is not None
        # New contract: lightweight markdown comment markers instead of
        # CDATA-wrapped XML. The names stay the same so existing greps work.
        assert "<!-- workspace_context -->" in msg.content
        assert "<!-- /workspace_context -->" in msg.content
        assert "<![CDATA[" not in msg.content
        assert "# Workspace Rules" in msg.content
        assert "<!-- session_guidance -->" in msg.content
        assert "DYNAMIC SKILL GUIDANCE" in msg.content
        assert "<!-- workspace_memory -->" in msg.content
        assert "Prefer concise responses." in msg.content

    @pytest.mark.asyncio
    async def test_agentic_prompt_labels_dynamic_blocks(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            prompt_config=PromptConfig(enable_agentic_prompt=True),
        )
        agent.add_session_guidance("# Skills\n\nDYNAMIC SKILL GUIDANCE")
        agent.get_workspace_context_prompt = AsyncMock(return_value="# Workspace Rules\n- rule")
        agent.get_workspace_memory_prompt = AsyncMock(return_value="### Memory Rule\nPrefer concise responses.")

        msg = await agent.get_system_message()

        assert msg is not None
        assert "<!-- workspace_context -->" in msg.content
        assert "<![CDATA[" not in msg.content
        assert "# Workspace Rules" in msg.content
        assert "<!-- session_guidance -->" in msg.content
        assert "DYNAMIC SKILL GUIDANCE" in msg.content
        assert "<!-- workspace_memory -->" in msg.content
        assert "Prefer concise responses." in msg.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
