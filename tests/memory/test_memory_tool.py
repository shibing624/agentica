# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for BuiltinMemoryTool, MemoryExtractHooks, and memory config
"""
import asyncio
import json
import os
import sys
import tempfile
import shutil
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.workspace import Workspace
from agentica.tools.buildin_tools import BuiltinMemoryTool
from agentica.tools.builtin.task_state_tools import BuiltinMemoryTool as CanonicalBuiltinMemoryTool
from agentica.hooks import MemoryExtractHooks, ConversationArchiveHooks


def test_memory_tool_legacy_export_points_to_canonical_class():
    assert BuiltinMemoryTool is CanonicalBuiltinMemoryTool


class TestBuiltinMemoryTool:
    """Test BuiltinMemoryTool save_memory and search_memory."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Workspace(self.temp_dir)
        self.workspace.initialize()
        self.tool = BuiltinMemoryTool()
        self.tool.set_workspace(self.workspace)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test BuiltinMemoryTool initialization."""
        tool = BuiltinMemoryTool()
        assert tool.name == "builtin_memory_tool"
        assert "save_memory" in tool.functions
        assert "search_memory" in tool.functions

    def test_get_system_prompt(self):
        """Test that system prompt contains memory instructions."""
        prompt = self.tool.get_system_prompt()
        assert prompt is not None
        assert "save_memory" in prompt
        assert "search_memory" in prompt
        assert "user" in prompt
        assert "feedback" in prompt
        assert "project" in prompt
        assert "reference" in prompt

    def test_save_memory(self):
        """Test saving a memory entry."""
        result = asyncio.run(self.tool.save_memory(
            title="user_role",
            content="User is a Python developer",
            memory_type="user",
        ))
        assert "Memory saved" in result
        assert "user_role" in result

        # Verify file was created
        memory_dir = self.workspace._get_user_memory_dir()
        files = list(memory_dir.glob("*.md"))
        assert len(files) == 1
        assert "user_user_role" in files[0].name

    def test_save_memory_invalid_type(self):
        """Test saving with invalid memory type."""
        with pytest.raises(ValueError, match="invalid"):
            asyncio.run(self.tool.save_memory(
                title="test",
                content="test content",
                memory_type="invalid",
            ))

    def test_save_memory_empty_title(self):
        """Test saving with empty title."""
        with pytest.raises(ValueError):
            asyncio.run(self.tool.save_memory(
                title="",
                content="test content",
            ))

    def test_save_memory_empty_content(self):
        """Test saving with empty content."""
        with pytest.raises(ValueError):
            asyncio.run(self.tool.save_memory(
                title="test",
                content="",
            ))

    def test_save_memory_no_workspace(self):
        """Test saving without workspace configured."""
        tool = BuiltinMemoryTool()
        with pytest.raises(RuntimeError, match="No workspace"):
            asyncio.run(tool.save_memory(
                title="test",
                content="test content",
            ))

    def test_search_memory(self):
        """Test searching memories."""
        # Save some memories first
        asyncio.run(self.tool.save_memory(
            title="python_preference",
            content="User prefers Python for AI development",
            memory_type="user",
        ))
        asyncio.run(self.tool.save_memory(
            title="testing_framework",
            content="User prefers pytest over unittest",
            memory_type="feedback",
        ))

        # Search
        result = self.tool.search_memory("python")
        parsed = json.loads(result)
        assert len(parsed) >= 1

    def test_search_memory_no_results(self):
        """Test searching with no matching results."""
        result = self.tool.search_memory("nonexistent_term_xyz")
        assert "No memories found" in result

    def test_search_memory_no_fallback_on_miss(self):
        """No fallback to recent memories: a true miss returns a no-match message."""
        for i in range(12):
            asyncio.run(self.tool.save_memory(
                title=f"memory_{i:02d}",
                content=f"Persistent user memory {i:02d}",
                memory_type="user",
            ))

        result = self.tool.search_memory("你好世界")
        assert "No memories found" in result

    def test_search_memory_includes_verified_candidates_and_conversations(self):
        """Verified memories, candidates, and recent conversations share one result pool."""
        asyncio.run(self.tool.save_memory(
            title="verified_canteen",
            content="Verified memory: canteen advertising uses Tencent marketing account setup.",
            memory_type="project",
        ))
        asyncio.run(self.workspace.write_memory_entry(
            title="candidate_canteen",
            content="Candidate memory: canteen advertising budget and audience context.",
            memory_type="project",
            description="canteen advertising Tencent marketing",
            source="auto_extract",
        ))
        asyncio.run(self.workspace.archive_conversation(
            [
                {"role": "user", "content": "老板在聊 canteen advertising 和 Tencent marketing"},
                {"role": "assistant", "content": "建议先明确品类、预算和投放目标。"},
            ],
            session_id="run-1",
        ))

        result = self.tool.search_memory("canteen advertising Tencent marketing", limit=10)
        parsed = json.loads(result)

        sources = {item["source"] for item in parsed}
        assert {"memory", "memory_candidate", "conversation"}.issubset(sources)
        assert all("score" in item for item in parsed)

    def test_search_memory_limits_conversations_to_recent_seven_days(self):
        """Conversation search should ignore older daily archive files by default."""
        conv_dir = self.workspace._get_user_conversations_dir()
        conv_dir.mkdir(parents=True, exist_ok=True)
        old_day = date.today() - timedelta(days=8)
        recent_day = date.today() - timedelta(days=1)
        (conv_dir / f"{old_day.isoformat()}.md").write_text(
            "---\n\n### old\n\n**user**: old canteen advertising note\n",
            encoding="utf-8",
        )
        (conv_dir / f"{recent_day.isoformat()}.md").write_text(
            "---\n\n### recent\n\n**user**: recent canteen advertising note\n",
            encoding="utf-8",
        )

        result = self.tool.search_memory("canteen advertising", limit=10)
        parsed = json.loads(result)

        contents = "\n".join(item["content"] for item in parsed)
        assert "recent canteen advertising note" in contents
        assert "old canteen advertising note" not in contents

    def test_search_memory_respects_total_character_limit(self):
        """Tool output should be capped so memory recall cannot explode context."""
        for i in range(5):
            asyncio.run(self.tool.save_memory(
                title=f"long_note_{i}",
                content=f"canteen advertising note {i}: " + ("x" * 500),
                memory_type="project",
            ))

        result = self.tool.search_memory("canteen advertising", limit=10, max_chars=600)
        parsed = json.loads(result)

        total_chars = sum(len(item["content"]) for item in parsed)
        assert total_chars <= 600
        assert any(item.get("truncated") for item in parsed)

    def test_search_memory_no_workspace(self):
        """Test searching without workspace configured."""
        tool = BuiltinMemoryTool()
        with pytest.raises(RuntimeError, match="No workspace"):
            tool.search_memory("test")

    def test_memory_index_updated(self):
        """Test that MEMORY.md index is updated after save."""
        asyncio.run(self.tool.save_memory(
            title="deploy_target",
            content="Deploy to AWS Lambda",
            memory_type="project",
        ))

        index_path = self.workspace._get_user_memory_md()
        assert index_path.exists()
        content = index_path.read_text(encoding="utf-8")
        assert "deploy_target" in content

    def test_save_multiple_memories(self):
        """Test saving multiple memories."""
        types = ["user", "feedback", "project", "reference"]
        for i, mem_type in enumerate(types):
            asyncio.run(self.tool.save_memory(
                title=f"memory_{i}",
                content=f"Content for memory {i}",
                memory_type=mem_type,
            ))

        memory_dir = self.workspace._get_user_memory_dir()
        files = list(memory_dir.glob("*.md"))
        assert len(files) == 4

    def test_search_memory_chinese(self):
        """Test searching with Chinese query (CJK bigram matching)."""
        asyncio.run(self.tool.save_memory(
            title="deploy_target",
            content="部署目标是阿里云的函数计算服务",
            memory_type="project",
        ))
        asyncio.run(self.tool.save_memory(
            title="user_lang",
            content="用户偏好使用中文交流",
            memory_type="user",
        ))

        # Chinese query should match via character bigrams
        result = self.tool.search_memory("部署")
        parsed = json.loads(result)
        assert len(parsed) >= 1
        # The deploy memory should match
        assert any("部署" in r["content"] for r in parsed)

    def test_search_memory_mixed_lang(self):
        """Test searching with mixed Chinese+English query."""
        asyncio.run(self.tool.save_memory(
            title="framework_choice",
            content="项目使用FastAPI框架开发REST API",
            memory_type="project",
        ))

        # Mixed query: English word "FastAPI" + Chinese context
        result = self.tool.search_memory("FastAPI框架")
        parsed = json.loads(result)
        assert len(parsed) >= 1
        assert any("FastAPI" in r["content"] for r in parsed)


class TestMemoryExtractHooks:
    """Test MemoryExtractHooks auto-extraction logic."""

    def test_init(self):
        """Test  initialization."""
        hooks = MemoryExtractHooks()
        assert isinstance(hooks._tool_calls, dict)

    def test_skips_when_save_memory_called(self):
        """Even at flush time, skip extraction when save_memory was called in the batch."""
        hooks = MemoryExtractHooks(every_n_turns=1, min_seconds_between=0, background=False)

        agent = MagicMock()
        agent.agent_id = "test_agent"
        agent.session_id = "sess_save"
        agent.run_input = "x" * 250
        agent.workspace = MagicMock()
        agent.workspace.user_id = "u1"
        agent.auxiliary_model = None
        agent.model = MagicMock()
        agent.model.response = AsyncMock()

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_tool_end(agent=agent, tool_name="save_memory"))
        asyncio.run(hooks.on_agent_end(agent=agent, output="y" * 250))

        # Even though every_n_turns=1 forces a flush, save_memory in the
        # batch suppresses the LLM call.
        agent.model.response.assert_not_called()

    def test_skips_when_no_workspace(self):
        """Test that extraction is skipped when no workspace."""
        hooks = MemoryExtractHooks()

        agent = MagicMock()
        agent.agent_id = "test_agent"
        agent.run_input = "test input"
        agent.workspace = None

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_agent_end(agent=agent, output="test output"))
        # Should not crash

    def test_tracks_tool_calls(self):
        """Test that tool calls are tracked correctly."""
        hooks = MemoryExtractHooks()

        agent = MagicMock()
        agent.agent_id = "test_agent"
        agent.run_input = "test"

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_tool_end(agent=agent, tool_name="read_file"))
        asyncio.run(hooks.on_tool_end(agent=agent, tool_name="save_memory"))

        assert "save_memory" in hooks._tool_calls["test_agent"]

    def test_skips_short_conversations(self):
        """Test that very short conversations are skipped."""
        hooks = MemoryExtractHooks()

        agent = MagicMock()
        agent.agent_id = "test_agent"
        agent.run_input = "hi"
        agent.workspace = MagicMock()
        agent.model = MagicMock()
        agent.model.response = AsyncMock()

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_agent_end(agent=agent, output="hello"))

        # Conversation < 200 chars, should skip extraction
        agent.model.response.assert_not_called()

    def test_default_buffers_without_calling_llm(self):
        """With default every_n_turns=10, a single turn just buffers, never fires LLM."""
        hooks = MemoryExtractHooks()
        agent = MagicMock()
        agent.agent_id = "test_agent_buf"
        agent.session_id = "sess_buf"
        agent.run_input = "x" * 150
        agent.workspace = MagicMock()
        agent.workspace.user_id = "u1"
        agent.auxiliary_model = None
        agent.model = MagicMock()
        agent.model.response = AsyncMock(return_value=MagicMock(content="[]"))

        async def scenario():
            await hooks.on_agent_start(agent=agent)
            await hooks.on_agent_end(agent=agent, output="y" * 600)
            key = ("u1", "sess_buf")
            assert key in hooks._buffers, "turn should be buffered for later flush"
            assert len(hooks._buffers[key]) == 1
            agent.model.response.assert_not_called()

        asyncio.run(scenario())

    def test_flushes_after_every_n_turns(self):
        """When every_n_turns turns accumulate, the LLM flush fires once."""
        # background=False so the assertion right after on_agent_end is
        # deterministic; production default is background=True.
        hooks = MemoryExtractHooks(every_n_turns=2, min_seconds_between=0, background=False)
        agent = MagicMock()
        agent.agent_id = "test_agent_flush"
        agent.resolve_auxiliary_model.side_effect = lambda task: agent.auxiliary_model or agent.model
        agent.session_id = "sess_flush"
        agent.run_input = "x" * 150
        agent.workspace = MagicMock()
        agent.workspace.user_id = "u1"
        agent.workspace.write_memory_entry = AsyncMock()
        agent.auxiliary_model = None
        agent.model = MagicMock()
        agent.model.response = AsyncMock(return_value=MagicMock(content="[]"))

        async def scenario():
            await hooks.on_agent_start(agent=agent)
            await hooks.on_agent_end(agent=agent, output="y" * 600)
            agent.model.response.assert_not_called()
            await hooks.on_agent_start(agent=agent)
            await hooks.on_agent_end(agent=agent, output="z" * 600)
            agent.model.response.assert_awaited_once()
            assert hooks._buffers.get(("u1", "sess_flush"), []) == []

        asyncio.run(scenario())

    def test_on_pre_compact_force_flushes(self):
        """on_pre_compact drains the buffer even when below every_n_turns."""
        hooks = MemoryExtractHooks(every_n_turns=99, min_seconds_between=0)
        agent = MagicMock()
        agent.agent_id = "test_agent_pre"
        agent.resolve_auxiliary_model.side_effect = lambda task: agent.auxiliary_model or agent.model
        agent.session_id = "sess_pre"
        agent.run_input = "x" * 150
        agent.workspace = MagicMock()
        agent.workspace.user_id = "u1"
        agent.workspace.write_memory_entry = AsyncMock()
        agent.auxiliary_model = None
        agent.model = MagicMock()
        agent.model.response = AsyncMock(return_value=MagicMock(content="[]"))

        async def scenario():
            await hooks.on_agent_start(agent=agent)
            await hooks.on_agent_end(agent=agent, output="y" * 600)
            agent.model.response.assert_not_called()
            await hooks.on_pre_compact(agent=agent)
            agent.model.response.assert_awaited_once()

        asyncio.run(scenario())

    def test_prefers_auxiliary_model_over_main_model(self):
        """Extraction should run on auxiliary_model when configured, not main model."""
        hooks = MemoryExtractHooks(every_n_turns=1, min_seconds_between=0, background=False)

        agent = MagicMock()
        agent.agent_id = "test_agent_auxiliary"
        agent.resolve_auxiliary_model.side_effect = lambda task: agent.auxiliary_model or agent.model
        agent.session_id = "sess_auxiliary"
        agent.run_input = "x" * 150
        agent.workspace = MagicMock()
        agent.workspace.user_id = "u1"
        agent.workspace.write_memory_entry = AsyncMock()
        agent.model = MagicMock()
        agent.model.response = AsyncMock(return_value=MagicMock(content="[]"))
        agent.auxiliary_model = MagicMock()
        agent.auxiliary_model.response = AsyncMock(return_value=MagicMock(content="[]"))

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_agent_end(agent=agent, output="y" * 600))

        agent.auxiliary_model.response.assert_awaited_once()
        agent.model.response.assert_not_called()

    def test_falls_back_to_main_model_when_no_auxiliary(self):
        """When auxiliary_model is None, extraction uses agent.model."""
        hooks = MemoryExtractHooks(every_n_turns=1, min_seconds_between=0, background=False)

        agent = MagicMock()
        agent.agent_id = "test_agent_fallback"
        agent.resolve_auxiliary_model.side_effect = lambda task: agent.auxiliary_model or agent.model
        agent.session_id = "sess_fb"
        agent.run_input = "x" * 150
        agent.workspace = MagicMock()
        agent.workspace.user_id = "u1"
        agent.workspace.write_memory_entry = AsyncMock()
        agent.auxiliary_model = None
        agent.model = MagicMock()
        agent.model.response = AsyncMock(return_value=MagicMock(content="[]"))

        asyncio.run(hooks.on_agent_start(agent=agent))
        asyncio.run(hooks.on_agent_end(agent=agent, output="y" * 600))

        agent.model.response.assert_awaited_once()

    def test_extract_and_save_passes_global_sync_flag(self):
        """Extracted user memories should pass global sync when enabled."""
        hooks = MemoryExtractHooks(sync_memories_to_global_agent_md=True)
        workspace = MagicMock()
        workspace.write_memory_entry = AsyncMock()
        model = MagicMock()
        model.response = AsyncMock(return_value=MagicMock(content=json.dumps([
            {
                "title": "user_role",
                "content": "User is a senior ML engineer.",
                "type": "user",
            }
        ])))

        asyncio.run(hooks._extract_and_save(model, workspace, "User: test\n\nAssistant: ok"))

        workspace.write_memory_entry.assert_awaited_once_with(
            title="user_role",
            content="User is a senior ML engineer.",
            memory_type="user",
            description="user_role",
            sync_to_global_agent_md=True,
            source="auto_extract",
        )


class TestWorkspaceMemoryConfig:
    """Test that auto_archive and auto_extract_memory are separate configs."""

    def test_config_defaults(self):
        """Test default values for workspace memory config."""
        from agentica.agent.config import WorkspaceMemoryConfig
        config = WorkspaceMemoryConfig()
        assert config.auto_archive is False
        assert config.auto_extract_memory is False
        assert config.sync_memories_to_global_agent_md is False
        assert config.load_workspace_context is True
        assert config.load_workspace_memory is True
        assert config.max_memory_entries == 5

    def test_archive_only_no_extract(self):
        """Test that auto_archive=True without auto_extract_memory only injects archive hooks."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.agent.config import WorkspaceMemoryConfig

        temp_dir = tempfile.mkdtemp()
        workspace = Workspace(temp_dir)
        workspace.initialize()

        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            workspace=workspace,
            enable_long_term_memory=True,
            long_term_memory_config=WorkspaceMemoryConfig(
                auto_archive=True,
                auto_extract_memory=False,
            ),
        )

        # Should have hooks
        assert agent._default_run_hooks is not None
        # Should only have ConversationArchiveHooks, not MemoryExtractHooks
        hooks_list = agent._default_run_hooks._hooks_list
        hook_types = [type(h).__name__ for h in hooks_list]
        assert "ConversationArchiveHooks" in hook_types
        assert "MemoryExtractHooks" not in hook_types

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_both_archive_and_extract(self):
        """Test that both flags inject both hooks."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.agent.config import WorkspaceMemoryConfig

        temp_dir = tempfile.mkdtemp()
        workspace = Workspace(temp_dir)
        workspace.initialize()

        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            workspace=workspace,
            enable_long_term_memory=True,
            long_term_memory_config=WorkspaceMemoryConfig(
                auto_archive=True,
                auto_extract_memory=True,
            ),
        )

        hooks_list = agent._default_run_hooks._hooks_list
        hook_types = [type(h).__name__ for h in hooks_list]
        assert "ConversationArchiveHooks" in hook_types
        assert "MemoryExtractHooks" in hook_types

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_hooks_when_both_false(self):
        """Test that no hooks are injected when both flags are false."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat

        temp_dir = tempfile.mkdtemp()
        workspace = Workspace(temp_dir)
        workspace.initialize()

        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            workspace=workspace,
        )

        assert agent._default_run_hooks is None

        shutil.rmtree(temp_dir, ignore_errors=True)


class TestAgentAutoMemoryRegistration:
    """Test that Agent registers BuiltinMemoryTool when long-term memory is enabled."""

    def test_register_memory_tool_when_memory_true(self):
        """Test BuiltinMemoryTool registration when long-term memory is enabled."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat

        temp_dir = tempfile.mkdtemp()
        workspace = Workspace(temp_dir)
        workspace.initialize()

        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            workspace=workspace,
            enable_long_term_memory=True,
        )

        has_memory_tool = any(
            isinstance(t, BuiltinMemoryTool) for t in (agent.tools or [])
        )
        assert has_memory_tool, "BuiltinMemoryTool should be registered when long-term memory is enabled"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_memory_tool_when_memory_false(self):
        """Test that BuiltinMemoryTool is NOT registered by default."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat

        temp_dir = tempfile.mkdtemp()
        workspace = Workspace(temp_dir)
        workspace.initialize()

        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            workspace=workspace,
        )

        has_memory_tool = any(
            isinstance(t, BuiltinMemoryTool) for t in (agent.tools or [])
        )
        assert not has_memory_tool, "BuiltinMemoryTool should NOT be registered by default"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_duplicate_memory_tool(self):
        """Test that BuiltinMemoryTool is not duplicated if already provided."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat

        temp_dir = tempfile.mkdtemp()
        workspace = Workspace(temp_dir)
        workspace.initialize()

        manual_tool = BuiltinMemoryTool()
        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            workspace=workspace,
            enable_long_term_memory=True,
            tools=[manual_tool],
        )

        count = sum(1 for t in (agent.tools or []) if isinstance(t, BuiltinMemoryTool))
        assert count == 1, "Should not duplicate BuiltinMemoryTool"

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_no_memory_tool_without_workspace(self):
        """Test that BuiltinMemoryTool is NOT registered without workspace."""
        from agentica.agent.base import Agent
        from agentica.model.openai import OpenAIChat

        agent = Agent(
            model=OpenAIChat(api_key="fake_openai_key"),
            enable_long_term_memory=True,
        )

        has_memory_tool = any(
            isinstance(t, BuiltinMemoryTool)
            for t in (agent.tools or [])
        )
        assert not has_memory_tool, "BuiltinMemoryTool should NOT be registered without workspace"


class TestClearDailyMemory:
    """Test that clear_daily_memory only deletes date-pattern files."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.workspace = Workspace(self.temp_dir)
        self.workspace.initialize()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_only_deletes_date_files(self):
        """Test that clear_daily_memory preserves typed entry files."""
        memory_dir = self.workspace._get_user_memory_dir()

        # Create date-pattern files
        (memory_dir / "2025-01-01.md").write_text("old memory", encoding="utf-8")
        (memory_dir / "2025-01-02.md").write_text("older memory", encoding="utf-8")
        (memory_dir / "2025-12-31.md").write_text("recent memory", encoding="utf-8")

        # Create typed entry files (should NOT be deleted)
        (memory_dir / "user_role.md").write_text("User is a developer", encoding="utf-8")
        (memory_dir / "project_deploy.md").write_text("Deploy to AWS", encoding="utf-8")

        # Keep only 1 most recent date file
        self.workspace.clear_daily_memory(keep_days=1)

        remaining = sorted(f.name for f in memory_dir.glob("*.md"))
        # Should keep: 2025-12-31.md (most recent date), user_role.md, project_deploy.md
        assert "2025-12-31.md" in remaining
        assert "user_role.md" in remaining
        assert "project_deploy.md" in remaining
        # Should delete: 2025-01-01.md, 2025-01-02.md
        assert "2025-01-01.md" not in remaining
        assert "2025-01-02.md" not in remaining


class TestRelevanceScoring:
    """Test the shared compute_relevance_score helper."""

    def test_english_word_match(self):
        score = Workspace.compute_relevance_score("python developer", "python developer at google")
        assert score > 0.5

    def test_no_match(self):
        score = Workspace.compute_relevance_score("rust", "python developer at google")
        assert score == 0.0

    def test_cjk_bigram_match(self):
        score = Workspace.compute_relevance_score("部署目标", "部署目标是阿里云")
        assert score > 0.0

    def test_empty_query(self):
        score = Workspace.compute_relevance_score("", "some content")
        assert score == 0.0
