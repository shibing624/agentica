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

    def test_tools_module_keeps_call_batching_guidance(self):
        """Batching independent calls has no tool-docstring equivalent."""
        from agentica.prompts.base.tools import get_tools_prompt
        content = get_tools_prompt()
        assert "one message" in content
        assert "run them in order" in content

    def test_tools_module_requires_exact_path_grounding(self):
        from agentica.prompts.base.tools import get_tools_prompt
        content = get_tools_prompt()
        assert "Do not invent file paths" in content
        assert "does not invent siblings" in content
        assert "hedge-probe" in content
        assert "Before calling any path-taking tool" not in content
        assert "*** Update File" in content
        assert "python rewriter" in content
        assert "`config.py`" in content
        assert "parallel `read_file`" in content
        assert "not read-patch-read-patch" in content
        assert "Read the current file" not in content

    def test_tools_module_allows_execute_pipelines(self):
        from agentica.prompts.base.tools import get_tools_prompt
        content = get_tools_prompt()
        assert "grep (not grep/rg)" not in content
        assert "(not find" not in content
        assert "not cat" not in content
        assert "| rg" in content or "| head" in content
        assert "do not force every probe into one script" in content
        assert "parallel_safe=True" in content
        assert "Prefer one long call" not in content
        assert "apply_patch" in content
        assert "find . -type f" not in content
        assert "xargs ls" not in content

    def test_heartbeat_module_content(self):
        from agentica.prompts.base.heartbeat import get_heartbeat_prompt
        content = get_heartbeat_prompt()
        assert isinstance(content, str)
        assert len(content) > 0
        assert "verify your work" in content


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
    async def test_tools_guide_skipped_without_file_tools(self):
        """An agent with unrelated tools must not be told to prefer read_file."""
        class WeatherLikeTool(Tool):
            def __init__(self):
                super().__init__(name="weather")
                self.register(self.get_weather)

            def get_weather(self, city: str) -> str:
                """Get the weather."""
                return "sunny"

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[WeatherLikeTool()],
            prompt_config=PromptConfig(enable_agentic_prompt=True),
        )
        msg = await agent.get_system_message()

        assert msg is not None
        for phantom in ("read_file", "apply_patch", "glob"):
            assert phantom not in msg.content

    @pytest.mark.asyncio
    async def test_tools_guide_present_with_file_tools(self):
        class FileLikeTool(Tool):
            def __init__(self):
                super().__init__(name="files")
                self.register(self.read_file)

            def read_file(self, file_path: str) -> str:
                """Read a file."""
                return ""

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[FileLikeTool()],
            prompt_config=PromptConfig(enable_agentic_prompt=True),
        )
        msg = await agent.get_system_message()

        assert msg is not None
        assert "# Using Your Tools" in msg.content
        assert "do not force every probe into one script" in msg.content
        assert "parallel_safe=True" in msg.content
        assert "<<'EOF'" in msg.content
        assert "2>/dev/null" in msg.content
        assert "apply_patch" in msg.content
        assert "not `;`" not in msg.content
        assert "swift" not in msg.content

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


class TestGitStateStaysOutOfThePrompt:
    """Git state is a tool call away and belongs there.

    Provider prompt caching is a byte-exact prefix match, and the system message
    sits in the prefix of *every* later cache breakpoint — so a per-turn
    ``git status`` re-prices not just the cached system prefix but the whole
    conversation behind it. Freezing it instead would only trade that for a
    stale file list. Both are worse than one `git` call when the agent
    actually needs the answer."""

    @pytest.mark.asyncio
    async def test_no_git_state_in_the_system_prompt(self, tmp_path):
        import subprocess

        from agentica.workspace import Workspace

        try:
            subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("git is unavailable")
        # Workspace.exists() is gated on users/; without it the prompt would
        # skip the whole workspace zone and the assertions below prove nothing.
        workspace = Workspace(tmp_path)
        workspace.initialize()
        workspace.user_agent_md_path().write_text("# Rules\n- be brief\n")
        (tmp_path / "dirty.py").write_text("x = 1\n")

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            workspace=workspace,
        )
        await workspace.freeze_snapshots()
        content = (await agent.get_system_message()).content

        assert "be brief" in content, "workspace context never reached the prompt"
        for leaked in ("## Git Status", "Git branch", "Uncommitted changes", "dirty.py"):
            assert leaked not in content


class TestSessionSnapshotsKeepThePromptByteStable:
    """Everything the system prompt reads from live state is a session snapshot.

    Provider prompt caching matches a byte-exact prefix, and the system message
    sits in the prefix of every later cache breakpoint — so a section that
    changes mid-session re-prices the whole conversation behind it, not just
    itself. Experiences and the skills catalogue are both written *by the agent
    itself* during a session (capture hooks, skill upgrade), which is exactly
    how that used to happen with nobody asking for it."""

    @pytest.mark.asyncio
    async def test_experiences_are_frozen_and_name_their_index(self, tmp_path):
        from agentica.experience.compiler import CompiledCard
        from agentica.workspace import Workspace

        workspace = Workspace(tmp_path)
        workspace.initialize()
        store = workspace.get_compiled_experience_store()
        await store.write(CompiledCard(
            title="session_start_lesson",
            content="Verify the path before writing.",
            experience_type="correction",
        ))

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            workspace=workspace,
            enable_experience_capture=True,
        )
        await workspace.freeze_snapshots()
        before = (await agent.get_system_message()).content

        # What the capture hooks do mid-session: write a new card.
        await store.write(CompiledCard(
            title="mid_session_lesson",
            content="Learned after the prompt was frozen.",
            experience_type="correction",
        ))
        after = (await agent.get_system_message()).content

        assert before == after, "a new experience card rewrote the cached prompt"
        assert "session_start_lesson" in before
        assert "mid_session_lesson" not in after
        assert str(workspace.experience_index_path) in before, (
            "the snapshot must name the index so the current set stays reachable"
        )

    @pytest.mark.asyncio
    async def test_memory_index_is_frozen_and_names_its_path(self, tmp_path):
        from agentica.workspace import Workspace

        workspace = Workspace(tmp_path)
        workspace.initialize()
        await workspace.write_memory_entry(
            title="Tea shop",
            content="BODY_MUST_NOT_APPEAR",
            memory_type="user",
            description="user runs a wechat tea shop",
        )
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            workspace=workspace,
            enable_long_term_memory=True,
        )
        await workspace.freeze_snapshots()
        before = (await agent.get_system_message()).content

        await workspace.write_memory_entry(
            title="Mid-session fact",
            content="learned later",
            memory_type="user",
            description="should not rewrite the prompt",
        )
        after = (await agent.get_system_message()).content

        assert before == after
        assert "Tea shop" in before
        assert "user runs a wechat tea shop" in before
        assert "BODY_MUST_NOT_APPEAR" not in before
        assert "Mid-session fact" not in after
        assert str(workspace.memory_index_path().resolve()) in before

    @pytest.mark.asyncio
    async def test_skill_catalogue_survives_a_mid_session_refresh(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            tools=[SkillTool(auto_load=False)],
        )
        # Patch the agent's per-instance SkillTool clone (see isolation contract).
        skill_tool = next(t for t in agent.tools if isinstance(t, SkillTool))
        skill_tool.get_system_prompt = lambda: "# Skills\n\nORIGINAL CATALOGUE"
        agent.refresh_tool_system_prompts()
        agent.freeze_session_guidance()
        before = (await agent.get_system_message()).content

        # What the skill upgrade hook does mid-session, unattended.
        skill_tool.get_system_prompt = lambda: "# Skills\n\nUPGRADED CATALOGUE"
        agent.refresh_tool_system_prompts()
        after = (await agent.get_system_message()).content

        assert before == after, "a background skill upgrade rewrote the stable prefix"
        assert "ORIGINAL CATALOGUE" in before
        assert "UPGRADED CATALOGUE" not in after

    @pytest.mark.asyncio
    async def test_unfrozen_agent_still_renders_live_guidance(self):
        """No Runner (direct SDK call) must not mean an empty skills block."""
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        )
        agent.add_session_guidance("# Skills\n\nLIVE GUIDANCE")

        content = (await agent.get_system_message()).content

        assert "LIVE GUIDANCE" in content

    def test_freeze_is_idempotent_and_clones_start_unfrozen(self):
        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        )
        agent.add_session_guidance("FIRST")
        agent.freeze_session_guidance()
        agent.add_session_guidance("SECOND")
        agent.freeze_session_guidance()

        assert agent._get_session_guidance_block() == "FIRST"
        # A subagent / swarm clone is a new session and must freeze its own.
        assert agent.clone()._session_guidance_snapshot is None


class TestKnowledgeRetrievalDoesNotBlockTheLoop:
    """``Knowledge.search`` is synchronous down to the embedding HTTP call, the
    vector store and the reranker. On the per-turn prompt path it must not run
    inline, or every turn with ``add_references`` freezes unrelated concurrent
    work for the whole retrieval round trip."""

    class _BlockingKnowledge:
        def __init__(self):
            self.thread_name = None

        def search(self, query, num_documents=None, **kwargs):
            import threading
            import time as _time

            self.thread_name = threading.current_thread().name
            _time.sleep(0.1)
            return []

    def _agent(self, knowledge):
        from agentica.agent.config import ToolConfig

        agent = Agent(
            name="A",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
            knowledge=knowledge,
            tool_config=ToolConfig(add_references=True, search_knowledge=False),
        )
        return agent

    @pytest.mark.asyncio
    async def test_retrieval_runs_off_the_event_loop(self):
        import asyncio
        import threading

        knowledge = self._BlockingKnowledge()
        agent = self._agent(knowledge)
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        spinner = asyncio.create_task(ticker())
        await agent.get_user_message(message="what is agentica")
        spinner.cancel()

        assert knowledge.thread_name is not None, "knowledge.search was never called"
        assert knowledge.thread_name != threading.current_thread().name, (
            "knowledge.search ran on the event loop thread"
        )
        assert ticks > 5, f"event loop only advanced {ticks} times during retrieval"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
