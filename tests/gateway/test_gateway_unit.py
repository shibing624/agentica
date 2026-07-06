"""Unit tests for gateway services: ModelFactory, LRUCache, Router, ChannelManager, ResponseFormatter, Settings.

Requires the [gateway] extras:
    pip install agentica[gateway]
"""
import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# Gateway tests require fastapi + lark-oapi etc. Skip cleanly if not installed.
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway] extras")


class TestAgentServiceApprovalMode:
    """set_session_approval_mode uses the unified ask/auto/allow-all vocabulary
    and mutates an already-cached Agent's permission mode in place."""

    def test_unknown_mode_falls_back_to_default(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        svc.set_session_approval_mode("s1", "full")
        assert svc.get_session_approval_mode("s1") == svc._DEFAULT_APPROVAL_MODE

    def test_valid_modes_are_persisted(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        for mode in ("ask", "auto", "allow-all"):
            svc.set_session_approval_mode("s1", mode)
            assert svc.get_session_approval_mode("s1") == mode

    def test_switches_cached_agent_permission_mode_in_place(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        agent = MagicMock()
        svc._cache.put("s1", agent)

        svc.set_session_approval_mode("s1", "ask")

        agent.set_permission_mode.assert_called_once_with("ask")

    def test_run_config_no_longer_carries_enabled_tools(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.run_context import RunSource

        svc = AgentService(workspace_path=str(tmp_path))
        svc.set_session_approval_mode("s1", "ask")
        run_config = svc._run_config_for_session("s1", RunSource.gateway)
        assert run_config.enabled_tools is None


class TestAgentServiceRunSource:
    """AgentService passes gateway/cron run source into RunConfig."""

    def test_chat_uses_gateway_source_by_default(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.run_context import RunSource

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        agent = MagicMock()
        agent.run = AsyncMock(return_value=MagicMock(content="ok", tools=[]))
        svc._get_agent = AsyncMock(return_value=agent)

        asyncio.run(svc.chat("hello", session_id="s1", user_id="u1"))

        config = agent.run.call_args.kwargs["config"]
        assert config.source == RunSource.gateway

    def test_chat_accepts_cron_source_override(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.run_context import RunSource

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        agent = MagicMock()
        agent.run = AsyncMock(return_value=MagicMock(content="ok", tools=[]))
        svc._get_agent = AsyncMock(return_value=agent)

        asyncio.run(svc.chat("hello", session_id="s1", user_id="u1", source=RunSource.cron))

        config = agent.run.call_args.kwargs["config"]
        assert config.source == RunSource.cron


class TestAgentServiceCronUsesAuxiliaryModel:
    """Scheduled (cron) sessions default to the cheaper auxiliary model as
    their main model when one is configured; interactive chat sessions and
    cron sessions without an auxiliary model configured are unaffected."""

    def test_cron_session_uses_auxiliary_model_when_configured(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.config import settings

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        with patch.object(settings, "auxiliary_model_provider", "zhipuai"), \
             patch.object(settings, "auxiliary_model_name", "glm-4.7-flash"), \
             patch("agentica.gateway.services.agent_service.create_model") as mock_create:
            mock_create.return_value = MagicMock()
            svc._build_agent("scheduled_job1")

        mock_create.assert_called_once()
        assert mock_create.call_args.args == ("zhipuai", "glm-4.7-flash")

    def test_cron_session_falls_back_to_main_model_without_auxiliary(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.config import settings

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        with patch.object(settings, "auxiliary_model_name", ""), \
             patch("agentica.gateway.services.agent_service.create_model") as mock_create:
            mock_create.return_value = MagicMock()
            svc._build_agent("scheduled_job1")

        assert mock_create.call_args.args == (svc.model_provider, svc.model_name)

    def test_interactive_session_ignores_auxiliary_model_shortcut(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.config import settings

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        with patch.object(settings, "auxiliary_model_provider", "zhipuai"), \
             patch.object(settings, "auxiliary_model_name", "glm-4.7-flash"), \
             patch("agentica.gateway.services.agent_service.create_model") as mock_create:
            mock_create.return_value = MagicMock()
            svc._build_agent("chat123")

        assert mock_create.call_args_list[0].args == (svc.model_provider, svc.model_name)


class TestAgentServiceRunCron:
    """run_cron() builds an independent, uncached Agent per job execution and
    is excluded from the chat sidebar (list_sessions())."""

    def test_run_cron_never_touches_the_interactive_agent_cache(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.run_context import RunSource

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        agent = MagicMock()
        agent.run = AsyncMock(return_value=MagicMock(content="done", tools=[]))
        svc._build_agent = MagicMock(return_value=agent)

        result = asyncio.run(svc.run_cron("do the thing", job_id="job1", user_id="u1"))

        assert result.content == "done"
        assert result.session_id == "scheduled_job1"
        config = agent.run.call_args.kwargs["config"]
        assert config.source == RunSource.cron
        # Never cached: a second run must build a fresh Agent again.
        assert svc._cache.get("scheduled_job1") is None
        asyncio.run(svc.run_cron("do it again", job_id="job1", user_id="u1"))
        assert svc._build_agent.call_count == 2

    def test_run_cron_rejects_concurrent_runs_of_the_same_job(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        busy_lock = MagicMock()
        busy_lock.locked.return_value = True
        svc._get_session_lock = MagicMock(return_value=busy_lock)

        with pytest.raises(RuntimeError, match="already has an active run"):
            asyncio.run(svc.run_cron("x", job_id="job1"))

    def test_list_sessions_excludes_scheduled_job_sessions(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.memory.session_log import SessionLog

        with patch.object(SessionLog, "list_sessions", return_value=[
            {"session_id": "scheduled_job1", "path": "x", "last_timestamp": 1},
            {"session_id": "chat123", "path": "y", "last_timestamp": 2},
        ]), patch.object(SessionLog, "session_preview", return_value={"first_user": "hi", "user_count": 1}):
            svc = AgentService(workspace_path=str(tmp_path))
            sessions = svc.list_sessions()

        assert [s["session_id"] for s in sessions] == ["chat123"]


# ============== TestSettings ==============

class TestSettings:
    """Test Settings configuration class."""

    def test_from_env_defaults(self):
        """Settings.from_env() with no env vars uses sensible defaults."""
        from agentica.gateway.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            with patch("agentica.gateway.config.apply_global_config", return_value={}):
                s = Settings.from_env()
        assert s.host == "0.0.0.0"
        assert s.port == 8789
        assert s.debug is False
        assert s.gateway_token is None
        assert s.model_provider == "deepseek"
        assert s.model_name == "deepseek-v4-flash"
        assert s.model_thinking == ""
        assert s.model_reasoning_effort == ""

    def test_from_env_custom(self):
        """Settings.from_env() reads custom env vars (profile mocked empty)."""
        from agentica.gateway.config import Settings
        env = {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "DEBUG": "true",
            "GATEWAY_TOKEN": "secret123",
            "AGENTICA_MODEL_PROVIDER": "openai",
            "AGENTICA_MODEL_NAME": "gpt-4o",
            "AGENTICA_MODEL_THINKING": "enabled",
            "AGENTICA_REASONING_EFFORT": "max",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("agentica.gateway.config.apply_global_config", return_value={}):
                s = Settings.from_env()
        assert s.host == "127.0.0.1"
        assert s.port == 9000
        assert s.debug is True
        assert s.gateway_token == "secret123"
        assert s.model_provider == "openai"
        assert s.model_name == "gpt-4o"
        assert s.model_thinking == "enabled"
        assert s.model_reasoning_effort == "max"

    def test_mutable_model_fields(self):
        """model_provider, model_name, model_thinking should be mutable."""
        from agentica.gateway.config import Settings
        s = Settings()
        s.model_provider = "anthropic"
        s.model_name = "claude-3.5"
        s.model_thinking = "auto"
        s.model_reasoning_effort = "high"
        assert s.model_provider == "anthropic"
        assert s.model_name == "claude-3.5"
        assert s.model_thinking == "auto"
        assert s.model_reasoning_effort == "high"

    def test_base_dir_mutable(self):
        """base_dir should be settable as both str and Path."""
        from pathlib import Path
        from agentica.gateway.config import Settings
        s = Settings()
        s.base_dir = "/tmp/test_dir"
        assert s.base_dir == Path("/tmp/test_dir")
        s.base_dir = Path("/tmp/other")
        assert s.base_dir == Path("/tmp/other")

    def test_upload_allowed_ext_set(self):
        """upload_allowed_ext_set parses comma-separated extensions."""
        from agentica.gateway.config import Settings
        s = Settings(upload_allowed_extensions=".py,.js,.ts")
        ext_set = s.upload_allowed_ext_set
        assert ext_set == {".py", ".js", ".ts"}


# ============== TestUploadFile ==============

class TestUploadFile:
    """Upload endpoint: streamed size limit + workspace path containment."""

    def _make_upload(self, content, filename):
        import io
        from starlette.datastructures import UploadFile
        return UploadFile(io.BytesIO(content), filename=filename)

    def _settings(self, tmp_path, max_mb=1):
        s = MagicMock()
        s.upload_allowed_ext_set = {".txt"}
        s.upload_max_size_mb = max_mb
        s.workspace_path = tmp_path
        return s

    def test_oversized_upload_rejected_with_mb_unit(self, tmp_path):
        from fastapi import HTTPException
        from agentica.gateway.routes import chat
        up = self._make_upload(b"x" * (2 * 1024 * 1024), "big.txt")
        with patch.object(chat, "settings", self._settings(tmp_path, max_mb=1)):
            with pytest.raises(HTTPException) as ei:
                asyncio.run(chat.upload_file(file=up, target_dir=""))
        assert ei.value.status_code == 413
        # Error message must use MB consistently (the old KB-vs-MB bug).
        assert "1MB" in ei.value.detail and "KB" not in ei.value.detail

    def test_valid_upload_written_to_workspace(self, tmp_path):
        from agentica.gateway.routes import chat
        up = self._make_upload(b"hello", "note.txt")
        with patch.object(chat, "settings", self._settings(tmp_path)):
            result = asyncio.run(chat.upload_file(file=up, target_dir=""))
        assert result["status"] == "ok"
        assert (tmp_path / "note.txt").read_bytes() == b"hello"

    def test_target_dir_outside_workspace_rejected(self, tmp_path):
        from fastapi import HTTPException
        from agentica.gateway.routes import chat
        up = self._make_upload(b"hi", "x.txt")
        outside = str(tmp_path.parent)  # parent dir escapes the workspace root
        with patch.object(chat, "settings", self._settings(tmp_path)):
            with pytest.raises(HTTPException) as ei:
                asyncio.run(chat.upload_file(file=up, target_dir=outside))
        assert ei.value.status_code == 400


# ============== TestLRUAgentCache ==============

class TestLRUAgentCache:
    """Test LRU cache for agent instances."""

    def test_put_and_get(self):
        from agentica.gateway.services.agent_service import LRUAgentCache
        cache = LRUAgentCache(max_size=3)
        mock_agent = MagicMock()
        cache.put("s1", mock_agent)
        assert cache.get("s1") is mock_agent
        assert cache.get("nonexistent") is None

    def test_eviction(self):
        """Oldest entry is evicted when cache exceeds max_size."""
        from agentica.gateway.services.agent_service import LRUAgentCache
        cache = LRUAgentCache(max_size=2)
        cache.put("s1", MagicMock())
        cache.put("s2", MagicMock())
        cache.put("s3", MagicMock())
        assert cache.get("s1") is None  # evicted
        assert cache.get("s2") is not None
        assert cache.get("s3") is not None

    def test_access_refreshes_lru_order(self):
        """Accessing an entry moves it to the end (most recent)."""
        from agentica.gateway.services.agent_service import LRUAgentCache
        cache = LRUAgentCache(max_size=2)
        cache.put("s1", MagicMock())
        cache.put("s2", MagicMock())
        cache.get("s1")  # refresh s1
        cache.put("s3", MagicMock())  # should evict s2, not s1
        assert cache.get("s1") is not None
        assert cache.get("s2") is None

    def test_delete(self):
        from agentica.gateway.services.agent_service import LRUAgentCache
        cache = LRUAgentCache(max_size=5)
        cache.put("s1", MagicMock())
        assert cache.delete("s1") is True
        assert cache.delete("s1") is False
        assert cache.get("s1") is None

    def test_clear(self):
        from agentica.gateway.services.agent_service import LRUAgentCache
        cache = LRUAgentCache(max_size=5)
        cache.put("s1", MagicMock())
        cache.put("s2", MagicMock())
        cache.clear()
        assert len(cache) == 0

    def test_keys(self):
        from agentica.gateway.services.agent_service import LRUAgentCache
        cache = LRUAgentCache(max_size=5)
        cache.put("s1", MagicMock())
        cache.put("s2", MagicMock())
        assert set(cache.keys()) == {"s1", "s2"}


# ============== TestMessageRouter ==============

class TestMessageRouter:
    """Test message routing rules and priority."""

    def _make_message(self, channel="feishu", channel_id="chat1", sender_id="user1"):
        from agentica.gateway.channels.base import ChannelType, Message
        return Message(
            channel=ChannelType(channel),
            channel_id=channel_id,
            sender_id=sender_id,
            sender_name="Test User",
            content="hello",
            message_id="msg1",
        )

    def test_default_route(self):
        """No rules → default agent."""
        from agentica.gateway.services.router import MessageRouter
        router = MessageRouter(default_agent="main")
        msg = self._make_message()
        assert router.route(msg) == "main"

    def test_sender_match(self):
        """Exact sender_id match routes to specific agent."""
        from agentica.gateway.services.router import MessageRouter, RoutingRule
        from agentica.gateway.channels.base import ChannelType
        router = MessageRouter(default_agent="main")
        router.add_rule(RoutingRule(agent_id="vip_agent", sender_id="user1"))
        msg = self._make_message(sender_id="user1")
        assert router.route(msg) == "vip_agent"

    def test_channel_match(self):
        """Channel type match."""
        from agentica.gateway.services.router import MessageRouter, RoutingRule
        from agentica.gateway.channels.base import ChannelType
        router = MessageRouter(default_agent="main")
        router.add_rule(RoutingRule(agent_id="tg_agent", channel=ChannelType.TELEGRAM))
        msg = self._make_message(channel="telegram")
        assert router.route(msg) == "tg_agent"

    def test_priority_ordering(self):
        """Higher priority rules are checked first."""
        from agentica.gateway.services.router import MessageRouter, RoutingRule
        from agentica.gateway.channels.base import ChannelType
        router = MessageRouter(default_agent="main")
        router.add_rule(RoutingRule(agent_id="low", channel=ChannelType.FEISHU, priority=1))
        router.add_rule(RoutingRule(agent_id="high", channel=ChannelType.FEISHU, priority=10))
        msg = self._make_message(channel="feishu")
        assert router.route(msg) == "high"

    def test_no_match_falls_to_default(self):
        """Non-matching rules fall through to default."""
        from agentica.gateway.services.router import MessageRouter, RoutingRule
        from agentica.gateway.channels.base import ChannelType
        router = MessageRouter(default_agent="main")
        router.add_rule(RoutingRule(agent_id="tg_agent", channel=ChannelType.TELEGRAM))
        msg = self._make_message(channel="feishu")
        assert router.route(msg) == "main"

    def test_session_id_format(self):
        """Session ID has deterministic format."""
        from agentica.gateway.services.router import MessageRouter
        router = MessageRouter()
        msg = self._make_message(channel="feishu", channel_id="chat123")
        sid = router.get_session_id(msg, "agent1")
        assert sid == "agent:agent1:feishu:chat123"

    def test_remove_rule(self):
        """Removing a rule by agent_id."""
        from agentica.gateway.services.router import MessageRouter, RoutingRule
        from agentica.gateway.channels.base import ChannelType
        router = MessageRouter(default_agent="main")
        router.add_rule(RoutingRule(agent_id="x", channel=ChannelType.FEISHU))
        router.remove_rule("x")
        assert len(router.rules) == 0

    def test_list_rules(self):
        """list_rules returns serialized dicts."""
        from agentica.gateway.services.router import MessageRouter, RoutingRule
        from agentica.gateway.channels.base import ChannelType
        router = MessageRouter()
        router.add_rule(RoutingRule(agent_id="a", channel=ChannelType.FEISHU, priority=5))
        rules = router.list_rules()
        assert len(rules) == 1
        assert rules[0]["agent_id"] == "a"
        assert rules[0]["channel"] == "feishu"
        assert rules[0]["priority"] == 5


# ============== TestChannelManager ==============

class TestChannelManager:
    """Test channel manager lifecycle and dispatch."""

    def _make_channel(self, channel_type_str="feishu", connected=True):
        from agentica.gateway.channels.base import ChannelType
        from unittest.mock import AsyncMock
        ch = MagicMock()
        ch.channel_type = ChannelType(channel_type_str)
        ch.is_connected = connected
        ch.send = AsyncMock(return_value=True)
        ch.connect = AsyncMock()
        ch.disconnect = AsyncMock()
        ch.set_handler = MagicMock()
        return ch

    def test_register_and_list(self):
        from agentica.gateway.services.channel_manager import ChannelManager
        mgr = ChannelManager()
        ch = self._make_channel("feishu")
        mgr.register(ch)
        assert mgr.list_channels() == ["feishu"]
        ch.set_handler.assert_called_once()

    def test_get_status(self):
        from agentica.gateway.services.channel_manager import ChannelManager
        mgr = ChannelManager()
        ch = self._make_channel("telegram", connected=True)
        mgr.register(ch)
        status = mgr.get_status()
        assert "telegram" in status
        assert status["telegram"]["connected"] is True

    def test_send_unknown_channel_type(self):
        from agentica.gateway.services.channel_manager import ChannelManager
        mgr = ChannelManager()
        result = asyncio.run(mgr.send("nonexistent", "chat1", "hello"))
        assert result is False

    def test_send_channel_not_registered(self):
        from agentica.gateway.services.channel_manager import ChannelManager
        from agentica.gateway.channels.base import ChannelType
        mgr = ChannelManager()
        result = asyncio.run(mgr.send(ChannelType.FEISHU, "chat1", "hello"))
        assert result is False

    def test_get_channel(self):
        from agentica.gateway.services.channel_manager import ChannelManager
        from agentica.gateway.channels.base import ChannelType
        mgr = ChannelManager()
        ch = self._make_channel("feishu")
        mgr.register(ch)
        assert mgr.get_channel(ChannelType.FEISHU) is ch
        assert mgr.get_channel(ChannelType.TELEGRAM) is None


# ============== TestResponseFormatter ==============

class TestResponseFormatter:
    """Test response formatting utilities."""

    def test_format_edit_file(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        result = format_tool_call_args("edit_file", {
            "file_path": "test.py",
            "old_string": "line1\nline2\nline3",
            "new_string": "line1\nnewline\nline3\nline4",
        })
        # "line1\nline2\nline3" → 2 newlines + 1 = 3
        assert result["_diff_del"] == 3
        # "line1\nnewline\nline3\nline4" → 3 newlines + 1 = 4
        assert result["_diff_add"] == 4
        assert result["file_path"] == "test.py"

    def test_format_write_file(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        result = format_tool_call_args("write_file", {
            "file_path": "new.py",
            "content": "import os\nimport sys\n",
        })
        assert result["_lines"] == 3  # 2 newlines + 1
        assert result["file_path"] == "new.py"

    def test_format_generic_truncation(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        long_str = "x" * 200
        result = format_tool_call_args("some_tool", {"query": long_str, "limit": 10})
        assert result["query"].endswith("...")
        assert len(result["query"]) == 103  # 100 + "..."
        assert result["limit"] == 10

    def test_format_tool_result_normal(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        name, result_str, is_task = format_tool_result({
            "tool_name": "read_file",
            "content": "file contents here",
        })
        assert name == "read_file"
        assert result_str == "file contents here"
        assert is_task is False

    def test_format_tool_result_empty(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        name, result_str, is_task = format_tool_result({
            "tool_name": "ls",
            "content": "",
        })
        assert result_str == "(no output)"

    def test_format_tool_result_error(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        name, result_str, is_task = format_tool_result({
            "tool_name": "execute",
            "content": "permission denied",
            "tool_call_error": True,
        })
        assert result_str.startswith("Error: ")

    def test_format_tool_result_task_meta(self):
        import json
        from agentica.gateway.services.response_formatter import format_tool_result
        task_content = json.dumps({"success": True, "tool_count": 3, "tool_calls_summary": ["a", "b"]})
        name, result_str, is_task = format_tool_result({
            "tool_name": "task",
            "content": task_content,
        })
        assert is_task is True
        parsed = json.loads(result_str)
        assert parsed["_task_meta"] is True
        assert parsed["tool_count"] == 3

    def test_extract_metrics_none(self):
        from agentica.gateway.services.response_formatter import extract_metrics
        assert extract_metrics(None) is None

    def test_extract_metrics_with_data(self):
        from agentica.gateway.services.response_formatter import extract_metrics
        agent = MagicMock()
        agent.run_response.metrics = {"input_tokens": [100], "output_tokens": [50]}
        result = extract_metrics(agent)
        assert result["input_tokens"] == [100]

    def test_multi_edit_file(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        result = format_tool_call_args("multi_edit_file", {
            "file_path": "test.py",
            "edits": [
                {"old_string": "a\nb", "new_string": "c"},
                {"old_string": "d", "new_string": "e\nf\ng"},
            ],
        })
        assert result["_edit_count"] == 2
        assert result["_diff_del"] == 3  # "a\nb" = 2+1, "d" = 0+1
        assert result["_diff_add"] == 4  # "c" = 0+1, "e\nf\ng" = 2+1


# ============== TestModelFactory ==============

class TestModelFactory:
    """Test model factory provider dispatch."""

    def test_unknown_provider_raises(self):
        from agentica.gateway.services.model_factory import create_model
        with pytest.raises(ValueError, match="Unknown model_provider"):
            create_model("nonexistent_provider", "some-model")

    def test_openai_provider(self):
        from agentica.gateway.services.model_factory import create_model
        model = create_model("openai", "gpt-4o-mini")
        assert model.__class__.__name__ == "OpenAIChat"

    def test_kimi_provider(self):
        from agentica.gateway.services.model_factory import create_model
        model = create_model("kimi", "moonshot-v1")
        assert model.__class__.__name__ == "KimiChat"

    def test_openai_compat_provider(self):
        """Providers in PROVIDER_FACTORIES (e.g. deepseek) should be created via the factory dispatch."""
        from agentica.gateway.services.model_factory import create_model
        from agentica import PROVIDER_FACTORIES
        provider_name = next(iter(PROVIDER_FACTORIES))
        model = create_model(provider_name, "test-model")
        assert model is not None

    def test_deepseek_provider_uses_v4_flash_thinking_defaults(self):
        """No thinking/reasoning args -> plain model with no reasoning_effort / extra_body.

        Thinking is opt-in only via the ``thinking`` / ``reasoning_effort`` args.
        Empty args -> plain model so user-side ``extra_body`` won't conflict.
        """
        from agentica.gateway.services.model_factory import create_model

        model = create_model("deepseek", "deepseek-v4-flash")

        assert model.id == "deepseek-v4-flash"
        assert model.base_url == "https://api.deepseek.com"
        assert model.context_window == 1_000_000
        assert model.reasoning_effort is None
        assert model.extra_body is None

    def test_deepseek_provider_respects_gateway_reasoning_effort(self):
        """reasoning_effort arg should override DeepSeek provider defaults."""
        from agentica.gateway.services.model_factory import create_model

        model = create_model("deepseek", "deepseek-v4-flash", reasoning_effort="max")

        assert model.reasoning_effort == "max"

    def test_cron_tools_returns_list(self):
        from agentica.gateway.services.model_factory import get_cron_tools
        tools = get_cron_tools()
        assert isinstance(tools, list)

    def test_cron_instructions_non_empty(self):
        from agentica.gateway.services.model_factory import get_cron_instructions
        instructions = get_cron_instructions()
        assert "cronjob" in instructions

    def test_self_manage_tools_returns_list(self):
        from agentica.gateway.services.model_factory import get_self_manage_tools
        tools = get_self_manage_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_self_manage_instructions_non_empty(self):
        from agentica.gateway.services.model_factory import get_self_manage_instructions
        instructions = get_self_manage_instructions()
        assert "self_manage" in instructions


# ============== TestChannelBase ==============

class TestChannelBase:
    """Test Channel base class shared utilities."""

    def test_split_text_normal(self):
        from agentica.gateway.channels.base import Channel
        chunks = Channel.split_text("abcdefgh", 3)
        assert chunks == ["abc", "def", "gh"]

    def test_split_text_empty(self):
        from agentica.gateway.channels.base import Channel
        assert Channel.split_text("", 100) == [""]

    def test_split_text_short(self):
        from agentica.gateway.channels.base import Channel
        assert Channel.split_text("hi", 100) == ["hi"]

    def test_check_allowlist_empty_allows_all(self):
        from agentica.gateway.channels.base import Channel
        # Create a concrete subclass for testing
        class _TestChannel(Channel):
            @property
            def channel_type(self):
                from agentica.gateway.channels.base import ChannelType
                return ChannelType.WEB
            async def connect(self): return True
            async def disconnect(self): pass
            async def send(self, channel_id, content, **kw): return True

        ch = _TestChannel(allowed_users=[])
        assert ch.check_allowlist("anyone") is True

    def test_check_allowlist_filters(self):
        from agentica.gateway.channels.base import Channel, ChannelType
        class _TestChannel(Channel):
            @property
            def channel_type(self):
                return ChannelType.WEB
            async def connect(self): return True
            async def disconnect(self): pass
            async def send(self, channel_id, content, **kw): return True

        ch = _TestChannel(allowed_users=["user1", "user2"])
        assert ch.check_allowlist("user1") is True
        assert ch.check_allowlist("user3") is False
