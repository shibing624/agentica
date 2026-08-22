"""Unit tests for gateway services: ModelFactory, LRUCache, Router, ChannelManager, ResponseFormatter, Settings.

Requires the [gateway] extras:
    pip install agentica[gateway]
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

# Gateway tests require fastapi + lark-oapi etc. Skip cleanly if not installed.
pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway] extras")


class TestAgentServiceApprovalMode:
    """set_session_approval_mode uses the unified ask/auto/allow-all vocabulary
    and mutates an already-cached Agent's permission mode in place."""

    def test_unknown_mode_falls_back_to_default(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        assert svc._DEFAULT_APPROVAL_MODE == "auto"
        svc.set_session_approval_mode("s1", "full")
        assert svc.get_session_approval_mode("s1") == "auto"

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
        svc._cache.put(svc._sk("s1"), agent)

        svc.set_session_approval_mode("s1", "ask")

        agent.set_permission_mode.assert_called_once_with("ask")

    def test_run_config_no_longer_carries_enabled_tools(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.run_context import RunSource

        svc = AgentService(workspace_path=str(tmp_path))
        svc.set_session_approval_mode("s1", "ask")
        run_config = svc._run_config_for_session("s1", RunSource.gateway)
        assert run_config.enabled_tools is None


class TestAgentServiceSteer:
    """Web interrupt is Agent.steer on the cached session agent."""

    def _svc(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        return AgentService(workspace_path=str(tmp_path))

    def test_steer_session_forwards_to_cached_agent(self, tmp_path):
        svc = self._svc(tmp_path)
        agent = MagicMock()
        agent.user_id = svc._owner(None)
        agent.steer.return_value = True
        svc._cache.put(svc._sk("s1"), agent)
        assert svc.steer_session("s1", "don't rewrite tests") is True
        agent.steer.assert_called_once_with("don't rewrite tests")

    def test_steer_session_missing_agent_is_false(self, tmp_path):
        svc = self._svc(tmp_path)
        assert svc.steer_session("missing", "hello") is False

    def test_steer_session_wrong_owner_is_false(self, tmp_path):
        svc = self._svc(tmp_path)
        agent = MagicMock()
        agent.user_id = "alice"
        agent.steer.return_value = True
        svc._cache.put(svc._sk("s1", "alice"), agent)
        assert svc.steer_session("s1", "x", owner="bob") is False
        agent.steer.assert_not_called()

    def test_take_undelivered_steer_returns_texts(self, tmp_path):
        svc = self._svc(tmp_path)
        agent = MagicMock()
        agent.user_id = svc._owner(None)
        agent.pop_undelivered_steer.return_value = [("late", False), ("also", True)]
        svc._cache.put(svc._sk("s1"), agent)
        assert svc.take_undelivered_steer("s1") == ["late", "also"]
        agent.pop_undelivered_steer.assert_called_once()


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


class TestAgentServiceChatMedia:
    """chat() routes inbound media: base-capable payloads attach to the run,
    fallback descriptions become text parts, and notes reach the reply."""

    def _svc_with_agent(self, tmp_path, agent):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        svc._get_agent = AsyncMock(return_value=agent)
        return svc

    def test_base_capable_image_attaches_to_run(self, tmp_path):
        from agentica.gateway.channels.base import InboundMedia

        agent = MagicMock()
        agent.model = SimpleNamespace(id="gpt-4o", supports_images=True)
        agent.run = AsyncMock(return_value=MagicMock(content="看到了", tools=[]))
        svc = self._svc_with_agent(tmp_path, agent)

        result = asyncio.run(svc.chat(
            "这是什么", session_id="s1", user_id="u1",
            media=[InboundMedia(kind="image", data=b"\xff\xd8\xff\xe0xx")],
        ))

        images = agent.run.call_args.kwargs["images"]
        assert len(images) == 1
        assert images[0]["url"].startswith("data:image/jpeg;base64,")
        assert agent.run.call_args.kwargs["audio"] is None
        assert agent.run.call_args.args[0] == "这是什么"
        assert result.media_notes == []

    def test_fallback_voice_transcript_appended_and_noted(self, tmp_path, monkeypatch):
        from agentica.gateway.channels.base import InboundMedia
        from agentica.gateway.services import media_understanding as mu

        class _FakeModel:
            async def response(self, messages):
                return SimpleNamespace(content="你好，世界")

        monkeypatch.setattr(mu, "get_setting", lambda key, default=None: {
            "model_provider": "openai",
            "model_name": "gemini-3.6-flash",
            "api_key": "k",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        } if key == "media_model" else default)
        monkeypatch.setattr(
            "agentica.gateway.services.agent_service.media_understanding",
            mu.MediaUnderstandingService(create_model_fn=lambda *a, **kw: _FakeModel()),
        )

        agent = MagicMock()
        agent.model = SimpleNamespace(id="deepseek-v4-flash", supports_images=False)
        agent.run = AsyncMock(return_value=MagicMock(content="好的", tools=[]))
        svc = self._svc_with_agent(tmp_path, agent)

        result = asyncio.run(svc.chat(
            "", session_id="s1", user_id="u1",
            media=[InboundMedia(kind="voice", data=b"RIFF" + b"\x00" * 40)],
        ))

        sent_message = agent.run.call_args.args[0]
        assert "[语音转写]\n你好，世界" in sent_message
        assert agent.run.call_args.kwargs["audio"] is None
        assert result.media_notes
        assert "gemini-3.6-flash" in result.media_notes[0]

    def test_no_media_keeps_plain_call(self, tmp_path):
        agent = MagicMock()
        agent.model = SimpleNamespace(id="deepseek-v4-flash", supports_images=False)
        agent.run = AsyncMock(return_value=MagicMock(content="ok", tools=[]))
        svc = self._svc_with_agent(tmp_path, agent)

        result = asyncio.run(svc.chat("hi", session_id="s1", user_id="u1"))

        assert agent.run.call_args.kwargs["images"] is None
        assert agent.run.call_args.kwargs["audio"] is None
        assert result.media_notes == []


class TestAgentServiceStreamToolDispatch:
    """Each tool callback must fire once, for its OWN tool.

    A parallel batch is announced up front and completed in call order, all
    events carrying the same cumulative ``tools`` list — so the stream handler
    must read ``chunk.tool_call`` (the subject of the event) instead of guessing
    by position.
    """

    @staticmethod
    def _parallel_batch_chunks():
        from agentica.run_response import RunEvent, RunResponse, ToolCallInfo

        calls = [
            ("c1", "read_file", {"file_path": "a.py"}, "ALPHA"),
            ("c2", "execute", {"command": "ls"}, "BETA"),
            ("c3", "grep", {"pattern": "x"}, "GAMMA"),
        ]
        tools = [{"tool_call_id": cid, "tool_name": n, "tool_args": a} for cid, n, a, _ in calls]
        chunks = [
            RunResponse(event=RunEvent.tool_call_started.value, tools=tools,
                        tool_call=ToolCallInfo(tool_call_id=cid, tool_name=n, tool_args=a))
            for cid, n, a, _ in calls
        ]
        for cid, n, a, content in calls:
            for t in tools:
                if t["tool_call_id"] == cid:
                    t["content"] = content
            chunks.append(RunResponse(
                event=RunEvent.tool_call_completed.value, tools=tools,
                tool_call=ToolCallInfo(tool_call_id=cid, tool_name=n, tool_args=a,
                                       content=content),
            ))
        chunks.append(RunResponse(event=RunEvent.run_response.value, content="done"))
        return chunks

    def _run(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        chunks = self._parallel_batch_chunks()

        async def fake_run_stream(*args, **kwargs):
            for c in chunks:
                yield c

        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        agent = MagicMock()
        agent.run_stream = fake_run_stream
        svc._get_agent = AsyncMock(return_value=agent)

        started, results = [], []

        async def on_tool_call(name, args, tool_call_id=""):
            started.append((name, tool_call_id))

        async def on_tool_result(name, result, extra=None):
            results.append((name, result))

        with patch(
            "agentica.gateway.services.agent_service.usage_payload",
            AsyncMock(return_value=None),
        ):
            result = asyncio.run(svc.chat_stream(
                "go", session_id="s1", user_id="u1",
                on_tool_call=on_tool_call, on_tool_result=on_tool_result,
            ))
        return started, results, result

    def test_each_tool_reports_once_in_call_order(self, tmp_path):
        started, results, chat_result = self._run(tmp_path)
        assert started == [("read_file", "c1"), ("execute", "c2"), ("grep", "c3")]
        assert [n for n, _ in results] == ["read_file", "execute", "grep"]
        assert chat_result.tool_calls == 3

    def test_each_result_belongs_to_its_own_tool(self, tmp_path):
        _, results, _ = self._run(tmp_path)
        assert results == [
            ("read_file", ""),
            ("execute", "BETA"),
            ("grep", ""),
        ]

    def test_write_file_completed_carries_unified_diff(self):
        from agentica.gateway.services.agent_service import dispatch_stream_chunk
        from agentica.run_response import RunEvent, RunResponse, ToolCallInfo

        extras = []

        async def on_tool_result(name, result, extra=None):
            extras.append(extra)

        chunk = RunResponse(
            event=RunEvent.tool_call_completed.value,
            tool_call=ToolCallInfo(
                tool_name="write_file",
                content="Created file, absolute path: hello.py",
                tool_display_meta={"files": [{
                    "path": "hello.py", "action": "add", "before": "", "after": "hi\n",
                }]},
            ),
        )
        asyncio.run(dispatch_stream_chunk(chunk, on_tool_result=on_tool_result))
        assert extras[0]["diff"].startswith("diff -- hello.py")
        assert "+hi" in extras[0]["diff"]


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

    def test_interactive_session_passes_responses_protocol(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.config import settings

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        with patch.object(settings, "model_provider", "openai"), \
             patch.object(settings, "model_name", "gpt-5.6-sol"), \
             patch.object(settings, "model_wire_api", "responses"), \
             patch.object(settings, "model_reasoning", "high"), \
             patch.object(settings, "model_reasoning_effort", ""), \
             patch.object(settings, "auxiliary_model_name", ""), \
             patch("agentica.gateway.services.agent_service.create_model") as mock_create:
            mock_create.return_value = MagicMock()
            svc._build_agent("chat123")

        kwargs = mock_create.call_args.kwargs
        assert kwargs["wire_api"] == "responses"
        assert kwargs["reasoning"] == "high"
        assert kwargs["reasoning_effort"] == ""


class TestAgentServiceNumHistoryTurns:
    """_build_agent must read num_history_turns from settings — the same
    single source of truth for both the interactive agent and any agent
    rebuilt after a model/profile switch invalidates the cache — never a
    separate hardcoded literal."""

    def test_build_agent_uses_settings_num_history_turns(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.config import settings

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        with patch.object(settings, "num_history_turns", 9), \
             patch("agentica.gateway.services.agent_service.create_model"), \
             patch("agentica.gateway.services.agent_service.DeepAgent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock(tools=[])
            svc._build_agent("chat123")

        assert mock_agent_cls.call_args.kwargs["num_history_turns"] == 9

    def test_rebuild_after_model_switch_stays_aligned_with_settings(self, tmp_path):
        """Simulates a model switch (settings.num_history_turns changed at
        runtime) followed by an agent rebuild — the rebuilt agent must pick
        up the new value, not a stale hardcoded one."""
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.config import settings

        svc = AgentService(workspace_path=str(tmp_path))
        svc._workspace = None
        with patch.object(settings, "num_history_turns", 3), \
             patch("agentica.gateway.services.agent_service.create_model"), \
             patch("agentica.gateway.services.agent_service.DeepAgent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock(tools=[])
            svc._build_agent("chat123")
        assert mock_agent_cls.call_args.kwargs["num_history_turns"] == 3

        with patch.object(settings, "num_history_turns", 12), \
             patch("agentica.gateway.services.agent_service.create_model"), \
             patch("agentica.gateway.services.agent_service.DeepAgent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock(tools=[])
            svc._build_agent("chat123")
        assert mock_agent_cls.call_args.kwargs["num_history_turns"] == 12


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
        from agentica.gateway.services import live_turn
        from agentica.memory.session_log import SessionLog

        live_turn.reset()
        with patch.object(SessionLog, "list_sessions", return_value=[
            {"session_id": "scheduled_job1", "path": "x", "last_timestamp": 1},
            {"session_id": "chat123", "path": "y", "last_timestamp": 2},
        ]), patch.object(SessionLog, "session_preview", return_value={"first_user": "hi", "user_count": 1}):
            svc = AgentService(workspace_path=str(tmp_path))
            sessions = svc.list_sessions()

        assert [s["session_id"] for s in sessions] == ["chat123"]
        assert sessions[0]["running"] is False

    def test_list_sessions_includes_every_project(self, tmp_path):
        """The web sidebar is grouped by work dir; listing only settings.base_dir
        dropped a just-finished chat after opening its trace."""
        from agentica.gateway.services.agent_service import AgentService
        from agentica.memory.session_log import SessionLog

        dir_a = tmp_path / "repo-a"
        dir_b = tmp_path / "repo-b"
        dir_a.mkdir()
        dir_b.mkdir()
        SessionLog("sess-a", work_dir=str(dir_a), user_id="default").append("user", "hello a")
        SessionLog("sess-b", work_dir=str(dir_b), user_id="default").append("user", "hello b")

        svc = AgentService(workspace_path=str(tmp_path))
        sessions = svc.list_sessions(owner="default")
        by_id = {s["session_id"]: s for s in sessions}
        assert set(by_id) >= {"sess-a", "sess-b"}
        assert by_id["sess-a"]["work_dir"] == str(dir_a)
        assert by_id["sess-b"]["work_dir"] == str(dir_b)

    def test_list_sessions_orders_by_creation_time(self, tmp_path):
        """Sidebar order = creation time (first request), newest on top; a
        later message must not reshuffle it."""
        from agentica.gateway.services.agent_service import AgentService
        from agentica.gateway.services import live_turn
        from agentica.memory.session_log import SessionLog

        live_turn.reset()
        with patch.object(SessionLog, "list_sessions", return_value=[
            {"session_id": "old", "path": "a",
             "first_timestamp": "2026-01-01T00:00:00.000Z", "last_timestamp": "2026-03-01T00:00:00.000Z"},
            {"session_id": "new", "path": "b",
             "first_timestamp": "2026-01-03T00:00:00.000Z", "last_timestamp": "2026-01-02T00:00:00.000Z"},
        ]), patch.object(SessionLog, "session_preview", return_value={"first_user": "hi", "user_count": 1}):
            svc = AgentService(workspace_path=str(tmp_path))
            sessions = svc.list_sessions()

        assert [s["session_id"] for s in sessions] == ["new", "old"]

    def test_list_sessions_real_files_keep_creation_order(self, tmp_path):
        """jsonl mtime follows the last write; web order must follow the first line."""
        from agentica.gateway.services.agent_service import AgentService
        from agentica.memory.session_log import SessionLog

        d = tmp_path / "repo"
        d.mkdir()
        SessionLog("older", work_dir=str(d), user_id="default").append(
            "user", "first", timestamp="2026-01-01T00:00:00.000Z",
        )
        SessionLog("newer", work_dir=str(d), user_id="default").append(
            "user", "second", timestamp="2026-01-02T00:00:00.000Z",
        )
        SessionLog("older", work_dir=str(d), user_id="default").append(
            "user", "later on older", timestamp="2026-01-03T00:00:00.000Z",
        )

        svc = AgentService(workspace_path=str(tmp_path))
        sessions = [
            s for s in svc.list_sessions(owner="default")
            if s["session_id"] in {"older", "newer"}
        ]
        assert [s["session_id"] for s in sessions] == ["newer", "older"]
        assert sessions[0]["first_timestamp"] == "2026-01-02T00:00:00.000Z"
        assert sessions[1]["first_timestamp"] == "2026-01-01T00:00:00.000Z"

    def test_session_log_for_finds_other_project_after_restart(self, tmp_path, monkeypatch):
        """View trace used settings.base_dir after a restart, so a chat that
        lived in another project 404'd even though the jsonl was on disk."""
        from agentica.gateway.config import settings
        from agentica.gateway.services.agent_service import AgentService
        from agentica.memory.session_log import SessionLog

        default_dir = tmp_path / "default-cwd"
        other = tmp_path / "other-repo"
        default_dir.mkdir()
        other.mkdir()
        log = SessionLog("s_trace1", work_dir=str(other), user_id="default")
        log.append("user", "hello from other")

        monkeypatch.setattr(settings, "base_dir", default_dir)
        svc = AgentService(workspace_path=str(tmp_path))
        assert svc._session_work_dirs == {}
        found = svc.session_log_for("s_trace1", owner="default")
        assert found.path.exists()
        assert found.path == log.path
        assert svc.get_session_work_dir("s_trace1") == str(other)


class TestAgentServiceOwnerPartition:
    """Cache, lock, live_turn and list_sessions are per-account, not bare session_id."""

    def test_delete_does_not_drop_another_owners_run(self, tmp_path):
        from agentica.gateway.services import live_turn
        from agentica.gateway.services.agent_service import AgentService

        live_turn.reset()
        live_turn.start("s1", owner="alice")
        svc = AgentService(workspace_path=str(tmp_path))
        alice_agent = MagicMock()
        alice_agent.user_id = "alice"
        svc._cache.put(svc._sk("s1", "alice"), alice_agent)
        lock = svc._get_session_lock("s1", "alice")

        assert svc.delete_session("s1", owner="bob") is False
        assert live_turn.active("s1", owner="alice") is not None
        assert svc._cache.contains(svc._sk("s1", "alice"))
        assert svc._session_locks.get(svc._sk("s1", "alice")) is lock
        live_turn.reset()

    def test_list_sessions_includes_live_run_before_jsonl(self, tmp_path):
        from agentica.gateway.services import live_turn
        from agentica.gateway.services.agent_service import AgentService

        live_turn.reset()
        live_turn.start("brand-new-live", owner="default")
        svc = AgentService(workspace_path=str(tmp_path))
        sessions = svc.list_sessions(owner="default")
        by_id = {s["session_id"]: s for s in sessions}
        assert "brand-new-live" in by_id
        assert by_id["brand-new-live"]["running"] is True
        live_turn.reset()

    def test_list_sessions_unread_when_activity_is_after_last_read(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.memory.session_log import SessionLog

        work = tmp_path / "repo"
        work.mkdir()
        log = SessionLog("s-unread", work_dir=str(work), user_id="default")
        log.append("user", "hello", timestamp="2026-01-01T00:00:00.000Z")
        log.set_last_read_at("2026-01-01T00:00:00.000Z")
        log.append("assistant", "world", timestamp="2026-01-01T00:01:00.000Z")

        svc = AgentService(workspace_path=str(tmp_path))
        by_id = {s["session_id"]: s for s in svc.list_sessions(owner="default")}
        assert by_id["s-unread"]["unread"] is True
        assert by_id["s-unread"]["running"] is False

        svc.mark_session_read("s-unread", owner="default")
        by_id = {s["session_id"]: s for s in svc.list_sessions(owner="default")}
        assert by_id["s-unread"]["unread"] is False

    def test_list_sessions_legacy_without_last_read_is_not_unread(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService
        from agentica.memory.session_log import SessionLog

        work = tmp_path / "repo"
        work.mkdir()
        SessionLog("s-old", work_dir=str(work), user_id="default").append(
            "user", "hello", timestamp="2026-01-01T00:00:00.000Z",
        )
        svc = AgentService(workspace_path=str(tmp_path))
        by_id = {s["session_id"]: s for s in svc.list_sessions(owner="default")}
        assert by_id["s-old"]["unread"] is False

    def test_same_session_id_locks_do_not_collide_across_owners(self, tmp_path):
        from agentica.gateway.services.agent_service import AgentService

        svc = AgentService(workspace_path=str(tmp_path))
        a = svc._get_session_lock("s1", "alice")
        b = svc._get_session_lock("s1", "bob")
        assert a is not b


# ============== TestSettings ==============

class TestSettings:
    """Test Settings configuration class."""

    def test_from_env_defaults(self):
        """Settings.from_env() with no env vars uses sensible defaults."""
        from agentica.gateway.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            with patch("agentica.gateway.config.apply_global_config", return_value={}):
                s = Settings.from_env()
        # Loopback, not 0.0.0.0: reaching this API is enough to run tools as
        # the user, so exposing it to the LAN has to be asked for.
        assert s.host == "127.0.0.1"
        assert s.port == 8881
        assert s.debug is False
        assert s.parent_pid == 0
        assert s.wechat_token_file.endswith("wxbot_token.json")
        assert s.wechat_allowed_users == []

    def test_peer_bridge_on_by_default_and_env_opts_out(self):
        """PEER_BRIDGE defaults to on; falsey values disable it."""
        from agentica.gateway.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            with patch("agentica.gateway.config.apply_global_config", return_value={}):
                assert Settings.from_env().peer_bridge_enabled is True
        for value in ("false", "0", "no", "off"):
            with patch.dict(os.environ, {"PEER_BRIDGE": value}, clear=True):
                with patch("agentica.gateway.config.apply_global_config", return_value={}):
                    assert Settings.from_env().peer_bridge_enabled is False, value

    def test_from_env_custom(self):
        """Settings.from_env() reads custom env vars (profile mocked empty)."""
        from agentica.gateway.config import Settings
        env = {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "DEBUG": "true",
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
        assert s.model_provider == "openai"
        assert s.model_name == "gpt-4o"
        assert s.model_thinking == "enabled"
        assert s.model_reasoning_effort == "max"

    def test_from_env_responses_profile_ignores_chat_effort_env(self):
        from agentica.gateway.config import Settings
        profile = {
            "model_provider": "openai",
            "model_name": "gpt-5.6-sol",
            "wire_api": "responses",
            "reasoning": "high",
        }
        with patch.dict(os.environ, {"AGENTICA_REASONING_EFFORT": "max"}, clear=False):
            with patch("agentica.gateway.config.apply_global_config", return_value=profile):
                s = Settings.from_env()
        assert s.model_wire_api == "responses"
        assert s.model_reasoning == "high"
        assert s.model_reasoning_effort == ""

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

    def test_pending_approval_pins_entry_against_eviction(self):
        """A LiveTurn with pending approvals is not LRU-evicted."""
        from agentica.agent.approvals import PendingApproval
        from agentica.gateway.services import live_turn
        from agentica.gateway.services.agent_service import AgentService, LRUAgentCache

        async def _run():
            live_turn.reset()
            turn = live_turn.start("s1", owner="alice")
            turn.approvals.wait(PendingApproval(
                tool_call_id="t1", name="execute", arguments={},
                question="q", preview="",
            ))
            cache = LRUAgentCache(max_size=1)
            pinned = MagicMock()
            cache.put(AgentService._sk("s1", "alice"), pinned)
            cache.put(AgentService._sk("s2", "alice"), MagicMock())
            assert cache.get(AgentService._sk("s1", "alice")) is pinned
            live_turn.reset()

        asyncio.run(_run())


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


class TestChannelCatalog:
    """Personal Assistant page payload: every entry, live status."""

    def test_web_always_on_and_wechat_listed(self):
        from agentica.gateway.channels.catalog import CATALOG, channel_overview
        data = channel_overview(
            None, host="127.0.0.1", port=8881,
            web_url="http://127.0.0.1:8881/chat",
        )
        assert data["web_url"] == "http://127.0.0.1:8881/chat"
        assert data["listen"]["loopback"] is True
        ids = [c["id"] for c in data["catalog"]]
        assert ids[0] == "web"
        assert "wechat" in ids and "qq" in ids
        assert len(ids) == len(CATALOG)
        web = data["catalog"][0]
        assert web["connected"] is True and web["configured"] is True
        wechat = next(c for c in data["catalog"] if c["id"] == "wechat")
        assert wechat["recommended"] is True
        assert wechat["configured"] is False
        assert wechat["env"][0]["name"] == "WECHAT_TOKEN_FILE"

    def test_registered_im_is_configured(self):
        from agentica.gateway.channels.base import ChannelType
        from agentica.gateway.channels.catalog import channel_overview
        from agentica.gateway.services.channel_manager import ChannelManager
        mgr = ChannelManager()
        ch = MagicMock()
        ch.channel_type = ChannelType.WECHAT
        ch.is_connected = True
        ch.set_handler = MagicMock()
        mgr.register(ch)
        data = channel_overview(
            mgr, host="0.0.0.0", port=8881,
            web_url="http://192.168.1.8:8881/chat",
        )
        wechat = next(c for c in data["catalog"] if c["id"] == "wechat")
        assert wechat["configured"] is True
        assert wechat["connected"] is True
        assert data["listen"]["loopback"] is False
        assert data["channels"] == ["wechat"]


# ============== TestResponseFormatter ==============

class TestResponseFormatter:
    """Test response formatting utilities."""

    def test_format_write_file_keeps_full_content(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        result = format_tool_call_args("write_file", {
            "file_path": "new.py",
            "content": "import os\nimport sys\n",
        })
        assert result["file_path"] == "new.py"
        assert result["content"] == "import os\nimport sys\n"

    def test_format_read_file_truncates_one_liner(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        long_str = "x" * 200
        result = format_tool_call_args("read_file", {"file_path": long_str, "offset": 0})
        assert result["file_path"].endswith("...")
        assert len(result["file_path"]) == 103
        assert result["offset"] == 0

    def test_format_execute_keeps_full_command(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        cmd = "python3 /tmp/bubble_sort.py --n 10000 --seed 1"
        result = format_tool_call_args("execute", {"command": cmd, "timeout": 300})
        assert result["command"] == cmd
        assert result["timeout"] == 300

    def test_format_tool_result_hides_read_file(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        name, result_str, extra = format_tool_result(
            ToolCallInfo(tool_name="read_file", content="file contents here")
        )
        assert name == "read_file"
        assert result_str == ""
        assert extra == {}

    def test_format_tool_result_includes_tool_call_id(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        _, _, extra = format_tool_result(
            ToolCallInfo(tool_name="execute", content="ok", tool_call_id="c9")
        )
        assert extra["tool_call_id"] == "c9"

    def test_format_tool_result_keeps_write_and_search(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        for name, body in (
            ("write_file", "Wrote 12 lines to foo.py"),
            ("apply_patch", "Successfully applied patch to foo.py"),
            ("write_todos", "Updated 3 todos"),
            ("web_search", "1. example.com — hello"),
            ("fetch_url", "<html>ok</html>"),
            ("save_memory", "Saved to memory"),
            ("search_memory", "- note about bubble sort"),
        ):
            _, result_str, extra = format_tool_result(
                ToolCallInfo(tool_name=name, content=body)
            )
            assert result_str == body, name
            assert extra == {}

    def test_format_tool_result_write_file_includes_unified_diff(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        name, result_str, extra = format_tool_result(ToolCallInfo(
            tool_name="write_file",
            content="Created file, absolute path: /tmp/a.py",
            tool_display_meta={"files": [{
                "path": "a.py", "action": "add", "before": "", "after": "print(1)\n",
            }]},
        ))
        assert name == "write_file"
        assert "Created file" in result_str
        assert extra["diff"].startswith("diff -- a.py")
        assert "+print(1)" in extra["diff"]

    def test_format_tool_result_apply_patch_multi_file_diff(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        _, _, extra = format_tool_result(ToolCallInfo(
            tool_name="apply_patch",
            content="Successfully applied patch to 2 files",
            tool_display_meta={"files": [
                {"path": "a.py", "action": "update", "before": "old\n", "after": "new\n"},
                {"path": "b.py", "action": "delete", "before": "gone\n", "after": None},
            ]},
        ))
        assert "diff -- a.py" in extra["diff"]
        assert "-old" in extra["diff"]
        assert "+new" in extra["diff"]
        assert "diff -- b.py" in extra["diff"]
        assert "-gone" in extra["diff"]

    def test_format_tool_result_execute_keeps_full_output(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        body = ("sorted ok\n" * 80)  # well past the old 500-char clip
        name, result_str, extra = format_tool_result(
            ToolCallInfo(tool_name="execute", content=body)
        )
        assert name == "execute"
        assert result_str == body
        assert extra == {}

    def test_format_tool_result_empty(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        name, result_str, extra = format_tool_result(
            ToolCallInfo(tool_name="execute", content="")
        )
        assert result_str == "(no output)"
        assert extra == {}

    def test_format_tool_result_error(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        name, result_str, extra = format_tool_result(
            ToolCallInfo(tool_name="execute", content="permission denied", is_error=True)
        )
        assert result_str.startswith("Error: ")
        assert extra == {}

    def test_format_tool_result_read_file_error_still_shown(self):
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        _, result_str, _ = format_tool_result(
            ToolCallInfo(tool_name="read_file", content="no such file", is_error=True)
        )
        assert "no such file" in result_str
        assert result_str.startswith("Error: ")

    def test_format_tool_result_task_keeps_detail(self):
        import json
        from agentica.gateway.services.response_formatter import format_tool_result
        from agentica.run_response import ToolCallInfo
        payload = {
            "success": True,
            "subagent_type": "explore",
            "subagent_name": "explore-1",
            "result": "Found foo in bar.py",
            "tool_calls_summary": [
                {"name": "read_file", "info": "bar.py"},
                {"name": "grep", "info": "foo"},
            ],
            "execution_time": 1.5,
            "tool_count": 2,
        }
        name, result_str, extra = format_tool_result(
            ToolCallInfo(tool_name="task", content=json.dumps(payload))
        )
        assert name == "task"
        assert extra == {}
        assert "Found foo in bar.py" in result_str
        assert "read_file bar.py" in result_str
        assert "grep foo" in result_str
        assert "explore-1" in result_str

    def test_extract_metrics_none(self):
        from agentica.gateway.services.response_formatter import extract_metrics
        assert extract_metrics(None) is None

    def test_extract_metrics_with_data(self):
        from agentica.gateway.services.response_formatter import extract_metrics
        agent = MagicMock()
        agent.run_response.metrics = {"input_tokens": [100], "output_tokens": [50]}
        result = extract_metrics(agent)
        assert result["input_tokens"] == [100]

    def test_apply_patch_keeps_full_patch(self):
        from agentica.gateway.services.response_formatter import format_tool_call_args
        patch = """*** Begin Patch
*** Update File: app.py
@@
-OLD = 1
+NEW = 1
*** End Patch"""
        result = format_tool_call_args("apply_patch", {"patch": patch})
        assert result["patch"] == patch


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

    def test_openai_responses_wire_api(self):
        from agentica.gateway.services.model_factory import create_model
        model = create_model(
            "openai",
            "gpt-5.6-sol",
            wire_api="responses",
            reasoning="high",
        )
        assert model.__class__.__name__ == "OpenAIResponses"
        assert model.reasoning == "high"

    def test_reasoning_does_not_implicitly_select_responses(self):
        from agentica.gateway.services.model_factory import create_model
        with pytest.raises(ValueError, match="requires wire_api='responses'"):
            create_model("openai", "gpt-5.6-sol", reasoning="high")

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


class TestGatewayStartupLogPath:
    def test_home_path_uses_tilde(self, tmp_path):
        from agentica.gateway.main import _display_home_path

        home = Path.home()
        assert _display_home_path(str(home / ".agentica/logs/20260814-65634.log")) == (
            "~/.agentica/logs/20260814-65634.log"
        )
        assert _display_home_path(str(tmp_path / "x.log")) == str(tmp_path / "x.log")


class TestCompactSession:
    def _svc(self, tmp_path, agent):
        from agentica.gateway.services.agent_service import AgentService
        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        svc._get_agent = AsyncMock(return_value=agent)
        return svc

    def test_empty_history_is_not_an_error_payload_ok_false(self, tmp_path):
        agent = MagicMock()
        agent.working_memory.messages = []
        svc = self._svc(tmp_path, agent)
        result = asyncio.run(svc.compact_session("s1"))
        assert result["ok"] is False
        assert "No messages" in result["error"]

    def test_local_compact_collapses_runs(self, tmp_path):
        messages = [MagicMock(), MagicMock(), MagicMock()]
        agent = MagicMock()
        agent.working_memory.messages = messages
        agent.model.supports_native_compaction = False
        agent.model.id = "gpt-4o-mini"
        agent._run_hooks = None
        agent._session_log = None
        agent.tool_config.compression_manager.auto_compact = AsyncMock(return_value=True)
        agent.run_response = None
        agent.model.usage = None
        # usage_payload walks working_memory + measure_context; stub the live
        # agent enough that compact's return path can call it.
        svc = self._svc(tmp_path, agent)
        with patch(
            "agentica.gateway.services.agent_service.usage_payload",
            AsyncMock(return_value={"context_tokens": 12}),
        ):
            result = asyncio.run(svc.compact_session("s1"))
        assert result["ok"] is True
        assert result["native"] is False
        assert result["messages_before"] == 3
        agent.working_memory.collapse_runs.assert_called_once()
        agent.tool_config.compression_manager.auto_compact.assert_awaited_once()

    def test_compact_refused_when_session_locked(self, tmp_path):
        agent = MagicMock()
        svc = self._svc(tmp_path, agent)
        busy_lock = MagicMock()
        busy_lock.locked.return_value = True
        svc._get_session_lock = MagicMock(return_value=busy_lock)
        with pytest.raises(RuntimeError, match="already has an active run"):
            asyncio.run(svc.compact_session("s1"))


class TestGoalEventPayload:
    def test_formats_token_progress(self):
        from agentica.gateway.services.agent_service import goal_event_payload
        d = goal_event_payload({
            "status": "active",
            "objective": "ship it",
            "tokens_used": 1234,
            "token_budget": 80_000,
            "turns_used": 2,
            "turn_budget": 15,
        })
        assert d["progress"] == "tokens 1,234/80,000"
        assert d["objective"] == "ship it"
        assert d["status"] == "active"
        assert "turns" not in d["progress"]
        assert "wall" not in d["progress"]

    def test_unlimited_budget_is_bare_count(self):
        from agentica.gateway.services.agent_service import goal_event_payload
        d = goal_event_payload({"tokens_used": 10, "turns_used": 1})
        assert d["progress"] == "tokens 10"


class TestWebRunGoalBudgets:
    def _svc(self, tmp_path, agent):
        from agentica.gateway.services.agent_service import AgentService
        svc = AgentService(workspace_path=str(tmp_path))
        svc._ensure_initialized = AsyncMock()
        svc._workspace = None
        svc._get_agent = AsyncMock(return_value=agent)
        return svc

    def _agent(self):
        agent = MagicMock()
        agent.working_memory.get_messages.return_value = []
        result = MagicMock()
        result.status = "complete"
        result.reason = "done"
        result.response_content = "ok"
        result.turns_used = 1
        agent.run_goal = AsyncMock(return_value=result)
        return agent

    def test_default_is_unlimited_tokens_only(self, tmp_path):
        agent = self._agent()
        svc = self._svc(tmp_path, agent)
        asyncio.run(svc.run_goal("hi", "s1"))
        kwargs = agent.run_goal.call_args.kwargs
        assert kwargs["token_budget"] == -1
        assert "turn_budget" not in kwargs
        assert "wall_clock_budget_sec" not in kwargs

    def test_positive_token_budget_is_forwarded(self, tmp_path):
        agent = self._agent()
        svc = self._svc(tmp_path, agent)
        asyncio.run(svc.run_goal("hi", "s1", token_budget=500_000))
        assert agent.run_goal.call_args.kwargs["token_budget"] == 500_000

    def test_run_goal_always_passes_stream_chunks(self, tmp_path):
        agent = self._agent()
        svc = self._svc(tmp_path, agent)
        asyncio.run(svc.run_goal("hi", "s1"))
        kwargs = agent.run_goal.call_args.kwargs
        assert callable(kwargs["stream_chunks"])
        assert kwargs["isolate"] is False
        assert "seed_messages" not in kwargs

