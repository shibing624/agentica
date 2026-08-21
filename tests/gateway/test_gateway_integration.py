"""Integration tests for gateway FastAPI endpoints using TestClient.

Tests that exercise the HTTP layer (routes, middleware, response format).
Agent calls are mocked at the AgentService level to avoid real LLM calls.
"""
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

# Guard: only run if fastapi + httpx are available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture()
def mock_app():
    """Create a TestClient with deps patched after lifespan initialization.

    The lifespan creates real services; we patch deps.agent_service
    with a mock after startup to intercept all agent calls.
    """
    from agentica.gateway.services.agent_service import ChatResult
    from agentica.gateway.main import app
    from agentica.gateway import deps

    mock_svc = MagicMock()
    mock_svc.chat = AsyncMock(return_value=ChatResult(
        content="Hello from agent",
        tool_calls=1,
        session_id="test-session",
        user_id="test-user",
        tools_used=["read_file"],
    ))
    mock_svc.list_sessions = MagicMock(return_value=[
        {"session_id": "s1", "name": "Chat 1", "preview": "hello", "user_count": 1, "last_timestamp": None, "size_bytes": 100, "archived": False},
        {"session_id": "s2", "name": "Chat 2", "preview": "world", "user_count": 2, "last_timestamp": None, "size_bytes": 200, "archived": True},
    ])
    mock_svc.delete_session = MagicMock(return_value=True)
    mock_svc.rename_session = MagicMock()
    mock_svc.archive_session = MagicMock()
    mock_svc.get_context_window = MagicMock(return_value=128000)
    mock_svc._ensure_initialized = AsyncMock()
    mock_svc.model_provider = "openai"
    mock_svc.model_name = "gpt-4o"
    mock_svc.max_tokens = 0
    mock_svc.temperature = 0.0
    mock_svc.top_p = 0.0
    mock_svc.model_wire_api = ""
    mock_svc.model_reasoning = ""
    mock_svc.model_reasoning_effort = ""
    mock_svc._invalidate_cache = AsyncMock()
    mock_svc.reload_profile = AsyncMock()
    mock_svc.has_active_runs = MagicMock(return_value=False)
    mock_svc.read_user_agents_md = AsyncMock(return_value={
        "content": "# User Instructions\n",
        "path": "/tmp/AGENTS.md",
        "empty_template": True,
        "auto_extract": True,
        "user_id": "default",
    })
    mock_svc.write_user_agents_md = AsyncMock(return_value={
        "content": "be brief\n",
        "path": "/tmp/AGENTS.md",
        "empty_template": False,
        "auto_extract": True,
        "user_id": "default",
    })
    mock_svc._cache = MagicMock()
    mock_svc._cache.keys = MagicMock(return_value=[])
    mock_svc._workspace = None

    with TestClient(app, raise_server_exceptions=False) as client:
        # Override deps AFTER lifespan has initialized
        original_svc = deps.agent_service
        deps.agent_service = mock_svc
        yield client, mock_svc
        deps.agent_service = original_svc


class TestHealthEndpoint:
    """Test /health and / endpoints."""

    def test_root(self, mock_app):
        client, _ = mock_app
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert "version" in data

    def test_health(self, mock_app):
        client, _ = mock_app
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestChatEndpoint:
    """Test /api/chat (non-streaming)."""

    def test_chat_success(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/chat", json={
            "message": "Hello",
            "session_id": "test-session",
            "user_id": "test-user",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "Hello from agent"
        assert data["tool_calls"] == 1
        mock_svc.chat.assert_awaited_once()

    def test_chat_missing_message(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/chat", json={})
        assert resp.status_code == 422  # validation error


class TestSteerEndpoint:
    """POST /api/chat/steer is the web interrupt; false is not an HTTP error."""

    def test_steer_accepted(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.steer_session = MagicMock(return_value=True)
        resp = client.post("/api/chat/steer", json={"session_id": "s1", "message": "stop rewriting"})
        assert resp.status_code == 200
        assert resp.json() == {"accepted": True}
        mock_svc.steer_session.assert_called_once()

    def test_steer_refused_is_200(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.steer_session = MagicMock(return_value=False)
        resp = client.post("/api/chat/steer", json={"session_id": "s1", "message": "too late"})
        assert resp.status_code == 200
        assert resp.json() == {"accepted": False}

    def test_take_undelivered(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.take_undelivered_steer = MagicMock(return_value=["late note"])
        resp = client.post("/api/chat/steer/take", json={"session_id": "s1"})
        assert resp.status_code == 200
        assert resp.json() == {"messages": ["late note"]}


class TestGoalEndpoint:
    """POST /api/goal: web standing-goal. Default token_budget=-1 (unlimited)."""

    def test_omitted_token_budget_is_unlimited(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.run_goal = AsyncMock(return_value={
            "status": "complete", "reason": "done", "content": "ok", "turns_used": 1,
        })
        with client.stream("POST", "/api/goal", json={
            "objective": "ship it", "session_id": "s1",
        }) as resp:
            assert resp.status_code == 200
            list(resp.iter_lines())
        assert mock_svc.run_goal.await_count == 1
        assert mock_svc.run_goal.call_args.kwargs["token_budget"] == -1

    def test_token_budget_forwarded(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.run_goal = AsyncMock(return_value={
            "status": "complete", "reason": "done", "content": "ok", "turns_used": 1,
        })
        with client.stream("POST", "/api/goal", json={
            "objective": "ship it", "session_id": "s1", "token_budget": 500_000,
        }) as resp:
            assert resp.status_code == 200
            list(resp.iter_lines())
        assert mock_svc.run_goal.call_args.kwargs["token_budget"] == 500_000

    def test_streams_content_and_tools_before_done(self, mock_app):
        client, mock_svc = mock_app

        async def fake_run_goal(*_a, on_content=None, on_tool_call=None, on_tool_result=None, **kw):
            if on_tool_call:
                await on_tool_call("read_file", {"path": "a.py"})
            if on_tool_result:
                await on_tool_result("read_file", "print(1)")
            if on_content:
                await on_content("hello ")
                await on_content("world")
            on_event = kw.get("on_event")
            if on_event:
                on_event({"status": "active", "objective": "ship it", "progress": "tokens 10"})
            return {
                "status": "complete", "reason": "done", "content": "hello world", "turns_used": 1,
            }

        mock_svc.run_goal = fake_run_goal
        events = []
        with client.stream("POST", "/api/goal", json={
            "objective": "ship it", "session_id": "s1",
        }) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        kinds = [e["event"] for e in events]
        assert kinds == ["tool_call", "tool_result", "content", "content", "status", "done"]
        assert events[0]["data"]["name"] == "read_file"
        assert events[2]["data"] == "hello "
        assert events[-1]["data"]["status"] == "complete"


class TestSessionEndpoints:
    """Test /api/sessions endpoints."""

    def test_list_sessions(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        sessions = data["sessions"]
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["name"] == "Chat 1"
        assert sessions[1]["session_id"] == "s2"
        assert sessions[1]["archived"] is True

    def test_delete_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.delete("/api/sessions/s1")
        assert resp.status_code == 200
        mock_svc.delete_session.assert_called_with("s1", owner="default")

    def test_delete_nonexistent_session(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.delete_session = MagicMock(return_value=False)
        resp = client.delete("/api/sessions/nonexistent")
        assert resp.status_code == 404

    def test_rename_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/sessions/s1/rename", json={"name": "New Name"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "renamed"
        assert data["name"] == "New Name"
        mock_svc.rename_session.assert_called_with("s1", "New Name", owner="default")

    def test_rename_empty_name(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/sessions/s1/rename", json={"name": "  "})
        assert resp.status_code == 400

    def test_archive_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/sessions/s1/archive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"
        mock_svc.archive_session.assert_called_with("s1", archived=True, owner="default")

    def test_unarchive_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/sessions/s1/unarchive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unarchived"
        mock_svc.archive_session.assert_called_with("s1", archived=False, owner="default")

    def test_session_usage(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.session_usage = AsyncMock(return_value={
            "model": "openai/gpt-4o",
            "window": 128000,
            "context_tokens": 4200,
            "percent_full": 3.3,
            "messages": 4,
            "api_calls": 2,
            "cost_usd": 0.0123,
            "input_tokens": 8000,
            "output_tokens": 400,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_percent": None,
            "sections": [
                {"label": "System prompt", "tokens": 2800, "share": 0.67, "nested": False},
                {"label": "Conversation", "tokens": 1400, "share": 0.33, "nested": False},
            ],
        })
        resp = client.get("/api/sessions/s1/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == 128000
        assert data["messages"] == 4
        assert data["api_calls"] == 2
        assert data["sections"][0]["label"] == "System prompt"
        mock_svc.session_usage.assert_called_with("s1", owner="default")

    def test_session_usage(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.session_usage = AsyncMock(return_value={
            "model": "openai/gpt-4o",
            "window": 128000,
            "context_tokens": 4200,
            "percent_full": 3.3,
            "messages": 4,
            "api_calls": 2,
            "cost_usd": 0.0123,
            "input_tokens": 8000,
            "output_tokens": 400,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cache_hit_percent": None,
            "sections": [
                {"label": "System prompt", "tokens": 2800, "share": 0.67, "nested": False},
                {"label": "Conversation", "tokens": 1400, "share": 0.33, "nested": False},
            ],
        })
        resp = client.get("/api/sessions/s1/usage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == 128000
        assert data["messages"] == 4
        assert data["api_calls"] == 2
        assert data["sections"][0]["label"] == "System prompt"
        mock_svc.session_usage.assert_called_with("s1", owner="default")


class TestConfigEndpoints:
    """Test /api/status and /api/models endpoints."""

    def test_status(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "workspace" in data
        assert "supports_images" in data
        assert "media_model" in data

    def test_channels_catalog(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/channels")
        assert resp.status_code == 200
        data = resp.json()
        assert data["web_url"].endswith("/chat")
        assert "guide_url" in data
        ids = [c["id"] for c in data["catalog"]]
        assert ids[0] == "web"
        assert "wechat" in ids and "qq" in ids
        assert "channels" in data and "status" in data

    def test_browse_missing_dir_is_400(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/fs/browse", params={"path": "/no/such/agentica-dir"})
        assert resp.status_code == 400

    def test_list_models(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        # No hardcoded provider/model catalog — only current binding is returned.
        assert "current_provider" in data
        assert "current_name" in data
        assert "current" in data


class TestBaseDirEndpoint:
    """POST /api/config/base_dir — a directory must already exist; the
    server never creates one on the user's behalf. Setting a new dir is also
    how the frontend creates a new project (each dir maps 1:1 to a project)."""

    def test_existing_directory_succeeds(self, mock_app, tmp_path):
        client, mock_svc = mock_app
        mock_svc.update_work_dir = MagicMock()
        resp = client.post("/api/config/base_dir", json={"base_dir": str(tmp_path)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["base_dir"] == str(tmp_path)
        assert "created" not in data

    def test_nonexistent_directory_is_rejected_not_created(self, mock_app, tmp_path):
        client, mock_svc = mock_app
        mock_svc.update_work_dir = MagicMock()
        missing = tmp_path / "does_not_exist_yet"
        resp = client.post("/api/config/base_dir", json={"base_dir": str(missing)})
        assert resp.status_code == 400
        assert "does not exist" in resp.json()["detail"].lower()
        assert not missing.exists()

    def test_path_that_is_a_file_is_rejected(self, mock_app, tmp_path):
        client, mock_svc = mock_app
        mock_svc.update_work_dir = MagicMock()
        f = tmp_path / "some_file.txt"
        f.write_text("x")
        resp = client.post("/api/config/base_dir", json={"base_dir": str(f)})
        assert resp.status_code == 400
        assert "not a directory" in resp.json()["detail"].lower()

    def test_empty_path_is_rejected(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.update_work_dir = MagicMock()
        resp = client.post("/api/config/base_dir", json={"base_dir": "   "})
        assert resp.status_code == 400


class TestDirHistoryAndConfigFile:
    def test_missing_paths_are_dropped(self, mock_app, tmp_path):
        client, _ = mock_app
        from agentica.config import AGENTICA_CACHE_DIR
        hist = Path(AGENTICA_CACHE_DIR).expanduser() / "dir_history.json"
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(json.dumps(["/no/such/pytest-1207/gone", str(tmp_path)]))
        resp = client.get("/api/config/dir_history")
        assert resp.status_code == 200
        data = resp.json()["history"]
        assert str(tmp_path) in data
        assert not any("pytest-1207" in p for p in data)

    def test_delete_one_entry(self, mock_app, tmp_path):
        client, _ = mock_app
        other = tmp_path / "keep-me"
        gone = tmp_path / "drop-me"
        other.mkdir()
        gone.mkdir()
        from agentica.config import AGENTICA_CACHE_DIR
        hist = Path(AGENTICA_CACHE_DIR).expanduser() / "dir_history.json"
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text(json.dumps([str(other), str(gone)]))
        resp = client.delete("/api/config/dir_history", params={"path": str(gone)})
        assert resp.status_code == 200
        left = resp.json()["history"]
        assert str(gone) not in left
        assert str(other) in left

    def test_config_preview_masks_api_key(self, mock_app, tmp_path, monkeypatch):
        home = tmp_path / "aghome"
        home.mkdir()
        (home / "config.yaml").write_text(
            "active_profile: default\n"
            "profiles:\n"
            "  default:\n"
            "    model_provider: openai\n"
            "    model_name: gpt-4o\n"
            "    api_key: sk-secret-key-abcdefgh\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AGENTICA_HOME", str(home))
        client, _ = mock_app
        resp = client.get("/api/config/file")
        assert resp.status_code == 200
        content = resp.json()["content"]
        assert "sk-secret-key-abcdefgh" not in content
        assert "sk-s...efgh" in content


class TestWebPrefs:
    def test_prefs_roundtrip_on_disk(self, mock_app, tmp_path, monkeypatch):
        client, _ = mock_app
        import agentica.gateway.routes.settings as settings_routes
        monkeypatch.setattr(settings_routes, "AGENTICA_HOME", str(tmp_path))
        empty = client.get("/api/prefs")
        assert empty.status_code == 200
        assert empty.json() == {}
        saved = client.put("/api/prefs", json={
            "theme": "dark",
            "lang": "zh",
            "approval_mode": "allow-all",
            "last_session_id": "s1",
            "garbage": 1,
        })
        assert saved.status_code == 200
        body = saved.json()
        assert body == {
            "theme": "dark",
            "lang": "zh",
            "approval_mode": "allow-all",
            "last_session_id": "s1",
        }
        assert client.get("/api/prefs").json() == body
        assert (tmp_path / "gateway" / "prefs" / "default.json").is_file()

    def test_auto_extract_memory_pref_roundtrip(self, mock_app, tmp_path, monkeypatch):
        client, mock_svc = mock_app
        import agentica.gateway.routes.settings as settings_routes
        monkeypatch.setattr(settings_routes, "AGENTICA_HOME", str(tmp_path))
        saved = client.put("/api/prefs", json={"auto_extract_memory": False})
        assert saved.status_code == 200
        assert saved.json()["auto_extract_memory"] is False
        mock_svc._invalidate_cache.assert_awaited()
        assert client.get("/api/prefs").json()["auto_extract_memory"] is False


class TestMemoryEndpoints:
    def test_get_memory(self, mock_app):
        client, mock_svc = mock_app
        resp = client.get("/api/memory")
        assert resp.status_code == 200
        assert resp.json()["path"] == "/tmp/AGENTS.md"
        mock_svc.read_user_agents_md.assert_awaited_once()

    def test_put_memory(self, mock_app):
        client, mock_svc = mock_app
        resp = client.put("/api/memory", json={"content": "be brief\n"})
        assert resp.status_code == 200
        assert resp.json()["content"] == "be brief\n"
        mock_svc.write_user_agents_md.assert_awaited_once()


class TestOpenUrlAndWechatQr:
    def test_open_url(self, mock_app, monkeypatch):
        client, _ = mock_app
        import agentica.gateway.routes.settings as settings_routes
        called = []
        monkeypatch.setattr(
            settings_routes.subprocess,
            "Popen",
            lambda args, **kw: called.append(args) or MagicMock(),
        )
        resp = client.post("/api/open", json={"url": "http://127.0.0.1:8881/chat"})
        assert resp.status_code == 200
        assert called
        assert "http://127.0.0.1:8881/chat" in called[0]

    def test_open_url_rejects_javascript(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/open", json={"url": "javascript:alert(1)"})
        assert resp.status_code == 400

    def test_wechat_qr_start_and_poll(self, mock_app):
        client, _ = mock_app
        from agentica.gateway import deps
        from agentica.gateway.channels.base import ChannelType

        ch = deps.channel_manager.get_channel(ChannelType.WECHAT)
        assert ch is not None
        ch.start_web_qr = AsyncMock(return_value={
            "status": "pending", "qrcode": "qid", "png": "aaa", "expires_in": 120,
        })
        ch.poll_web_qr = AsyncMock(return_value={"status": "wait"})
        started = client.post("/api/channels/wechat/qr")
        assert started.status_code == 200
        assert started.json()["png"] == "aaa"
        polled = client.get("/api/channels/wechat/qr", params={"id": "qid"})
        assert polled.status_code == 200
        assert polled.json()["status"] == "wait"
        ch.start_web_qr.assert_awaited_once()
        ch.poll_web_qr.assert_awaited_once()


class TestTempDirAndCompactEndpoints:
    def test_temp_workspace_is_created_under_agentica_home(self, mock_app, tmp_path, monkeypatch):
        client, _ = mock_app
        monkeypatch.setenv("AGENTICA_HOME", str(tmp_path))
        import agentica.gateway.routes.settings as settings_routes
        monkeypatch.setattr(settings_routes, "AGENTICA_HOME", str(tmp_path))
        resp = client.post("/api/fs/temp", json={})
        assert resp.status_code == 200
        path = Path(resp.json()["path"])
        assert path.is_dir()
        assert path.parent == tmp_path / "tmp" / "web-chats"

    def test_compact_empty_session_is_400(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.compact_session = AsyncMock(return_value={"ok": False, "error": "No messages to compact."})
        resp = client.post("/api/sessions/s1/compact", json={})
        assert resp.status_code == 400

    def test_compact_ok(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.compact_session = AsyncMock(return_value={
            "ok": True, "native": False, "messages_before": 40, "messages_after": 6,
        })
        resp = client.post("/api/sessions/s1/compact", json={"instructions": ""})
        assert resp.status_code == 200
        assert resp.json()["messages_before"] == 40
        mock_svc.compact_session.assert_awaited()

    def test_compact_busy_session_is_409(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.compact_session = AsyncMock(
            side_effect=RuntimeError("Session 's1' already has an active run. Wait for it to complete or cancel it first.")
        )
        resp = client.post("/api/sessions/s1/compact", json={})
        assert resp.status_code == 409


class TestSchedulerEndpoints:
    """Test /api/scheduler/* — cron job CRUD + actions + run history.

    cron.jobs functions are mocked so tests never touch ~/.agentica/cron.
    """

    def _fake_job(self, **over):
        from types import SimpleNamespace
        base = dict(
            id="j1", name="test", prompt="do thing", user_id="default",
            schedule="ignored", status=SimpleNamespace(value="active"),
            enabled=True, deliver="local", next_run_at_ms=0, last_run_at_ms=0,
            last_status=None, run_count=0, timeout_seconds=0, max_retries=0,
            retry_count=0, retry_delay_ms=60000, permissions={},
        )
        base.update(over)
        return SimpleNamespace(**base)

    def _fake_run(self, **over):
        from types import SimpleNamespace
        base = dict(task_id="j1", status=SimpleNamespace(value="ok"),
                    started_at_ms=1, ended_at_ms=2, result="done", error=None)
        base.update(over)
        return SimpleNamespace(**base)

    def test_list_jobs(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.list_jobs", return_value=[self._fake_job()]), \
             patch("agentica.gateway.routes.scheduler.schedule_to_human", return_value="daily 7:30"):
            resp = client.get("/api/scheduler/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["jobs"][0]["id"] == "j1"
        assert data["jobs"][0]["schedule"] == "daily 7:30"

    def test_create_job_body(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.cronjob",
                   return_value='{"success":true,"job":{"id":"j1"}}') as m:
            resp = client.post("/api/scheduler/jobs", json={
                "prompt": "daily report", "schedule": "30 7 * * *", "name": "rep",
            })
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        _, kwargs = m.call_args
        assert kwargs["prompt"] == "daily report"
        assert kwargs["schedule"] == "30 7 * * *"
        assert kwargs["name"] == "rep"
        assert kwargs["user_id"] == "default"

    def test_a_job_owned_by_someone_else_is_not_found(self, mock_app):
        """The id of another account's job must 404, not 403 — otherwise the
        status code confirms it exists."""
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job",
                   return_value=self._fake_job(user_id="kk")):
            assert client.get("/api/scheduler/jobs/j1").status_code == 404
            assert client.post("/api/scheduler/jobs/j1/pause").status_code == 404
            assert client.delete("/api/scheduler/jobs/j1").status_code == 404

    def test_create_ignores_a_user_id_in_the_body(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.cronjob",
                   return_value='{"success":true,"job":{"id":"j1"}}') as m:
            resp = client.post("/api/scheduler/jobs", json={
                "prompt": "daily report", "schedule": "30 7 * * *", "user_id": "kk",
            })
        assert resp.status_code == 200
        _, kwargs = m.call_args
        assert kwargs["user_id"] == "default"

    def test_create_job_missing_fields(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/scheduler/jobs", json={"name": "no prompt"})
        assert resp.status_code == 422  # Pydantic validation

    def test_update_job(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler.cronjob",
                   return_value='{"success":true,"job":{"id":"j1"}}') as m:
            resp = client.put("/api/scheduler/jobs/j1", json={"schedule": "every 2h"})
        assert resp.status_code == 200
        _, kwargs = m.call_args
        assert kwargs["schedule"] == "every 2h"
        assert "name" not in kwargs  # exclude_none drops unset fields

    def test_update_job_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=None):
            resp = client.put("/api/scheduler/jobs/none", json={"name": "x"})
        assert resp.status_code == 404

    def test_update_job_empty_body(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()):
            resp = client.put("/api/scheduler/jobs/j1", json={})
        assert resp.status_code == 400

    def test_delete_job(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler.remove_job", return_value=True):
            resp = client.delete("/api/scheduler/jobs/j1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_pause_resume(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler.pause_job", return_value=self._fake_job()):
            assert client.post("/api/scheduler/jobs/j1/pause").status_code == 200
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler.resume_job", return_value=self._fake_job()):
            assert client.post("/api/scheduler/jobs/j1/resume").status_code == 200

    def test_trigger(self, mock_app):
        """Trigger runs the job for real (via _execute_job), not just marks it due."""
        client, _ = mock_app
        fake_run = {"job_id": "j1", "status": "ok", "result": "done"}
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler._execute_job", new=AsyncMock(return_value=fake_run)) as m:
            resp = client.post("/api/scheduler/jobs/j1/trigger")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["run"] == fake_run
        m.assert_awaited_once()

    def test_trigger_failed_run(self, mock_app):
        client, _ = mock_app
        fake_run = {"job_id": "j1", "status": "failed", "error": "boom"}
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler._execute_job", new=AsyncMock(return_value=fake_run)):
            resp = client.post("/api/scheduler/jobs/j1/trigger")
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_create_job_with_validate_run(self, mock_app):
        """validate_run=True runs the freshly created job immediately."""
        client, _ = mock_app
        fake_run = {"job_id": "j1", "status": "ok", "result": "done"}
        with patch("agentica.gateway.routes.scheduler.cronjob",
                   return_value='{"success":true,"job":{"job_id":"j1"}}'), \
             patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler._execute_job", new=AsyncMock(return_value=fake_run)) as m:
            resp = client.post("/api/scheduler/jobs", json={
                "prompt": "daily report", "schedule": "30 7 * * *", "validate_run": True,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["validation_run"] == fake_run
        m.assert_awaited_once()

    def test_list_runs(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.list_jobs", return_value=[self._fake_job()]), \
             patch("agentica.gateway.routes.scheduler.list_task_runs", return_value=[self._fake_run()]):
            resp = client.get("/api/scheduler/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["runs"][0]["job_id"] == "j1"
        assert data["runs"][0]["result_preview"] == "done"
        assert data["runs"][0]["result_full"] == "done"

    def test_list_runs_result_full_uses_error_when_no_result(self, mock_app):
        client, _ = mock_app
        run = self._fake_run(result=None, error="boom")
        with patch("agentica.gateway.routes.scheduler.list_jobs", return_value=[self._fake_job()]), \
             patch("agentica.gateway.routes.scheduler.list_task_runs", return_value=[run]):
            resp = client.get("/api/scheduler/runs")
        assert resp.json()["runs"][0]["result_full"] == "boom"

    def test_list_job_runs(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler.list_task_runs", return_value=[self._fake_run()]):
            resp = client.get("/api/scheduler/jobs/j1/runs")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "j1"

    def test_list_job_runs_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=None):
            resp = client.get("/api/scheduler/jobs/none/runs")
        assert resp.status_code == 404


class TestProfileCrudEndpoints:
    """Test /api/profile CRUD (global_config fns mocked so ~/.agentica is untouched)."""

    def test_get_profile_detail(self, mock_app):
        client, _ = mock_app
        fake = {
            "model_provider": "openai", "model_name": "gpt-5.6-sol",
            "base_url": "https://example/v1", "api_key": "sk-1234567890abcdef",
            "wire_api": "responses", "reasoning": "high",
            "max_tokens": 8192, "context_window": 1000000,
            "compact_token_limit": 300000,
            "temperature": 0.7, "top_p": 0.95,
            "auxiliary_model": {"model_provider": "zhipuai", "model_name": "glm-4.7-flash",
                                "base_url": "https://open.bigmodel.cn", "api_key": "sk-aux"},
            "env": {"SERPER_API_KEY": "xxx"},
        }
        with patch("agentica.gateway.routes.settings.get_profile", return_value=fake):
            resp = client.get("/api/profile/default")
        assert resp.status_code == 200
        d = resp.json()
        assert d["model_provider"] == "openai"
        assert d["api_key_masked"].startswith("sk-1") and "****" in d["api_key_masked"]
        assert d["has_api_key"] is True
        assert d["wire_api"] == "responses"
        assert d["reasoning"] == "high"
        assert d["compact_token_limit"] == 300000
        assert d["env"]["SERPER_API_KEY"] == "xxx"
        assert d["auxiliary_model"]["model_provider"] == "zhipuai"

    def test_get_profile_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.settings.get_profile", return_value={}):
            resp = client.get("/api/profile/none")
        assert resp.status_code == 404

    def test_create_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.settings.upsert_profile") as m:
            resp = client.post("/api/profile", json={
                "name": "test-p", "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",
                "api_key": "sk-xxx",
                "compact_token_limit": 300000,
            })
        assert resp.status_code == 200
        args, kwargs = m.call_args
        assert args[0] == "test-p"
        assert args[1]["model_provider"] == "deepseek"
        assert args[1]["api_key"] == "sk-xxx"
        assert args[1]["compact_token_limit"] == 300000
        assert kwargs["make_active"] is False

    def test_create_responses_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.settings.upsert_profile") as upsert:
            resp = client.post("/api/profile", json={
                "name": "a",
                "model_provider": "openai",
                "model_name": "gpt-5.6-sol",
                "base_url": "https://example/v1",
                "wire_api": "responses",
                "reasoning": "high",
            })
        assert resp.status_code == 200
        profile = upsert.call_args.args[1]
        assert profile["wire_api"] == "responses"
        assert profile["reasoning"] == "high"

    def test_create_profile_missing_fields(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/profile", json={"name": "x"})
        assert resp.status_code == 422  # Pydantic validation

    def test_update_profile_keeps_api_key(self, mock_app):
        client, _ = mock_app
        existing = {"model_provider": "deepseek", "model_name": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com", "api_key": "sk-existing"}
        with patch("agentica.gateway.routes.settings.get_profile", return_value=existing), \
             patch("agentica.gateway.routes.settings.upsert_profile") as m:
            resp = client.put("/api/profile/default", json={
                "name": "default", "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash", "temperature": 0.5,
            })
        assert resp.status_code == 200
        args, _ = m.call_args
        merged = args[1]
        # api_key kept from existing because body didn't send one
        assert merged["api_key"] == "sk-existing"
        # temperature updated
        assert merged["temperature"] == 0.5

    def test_update_profile_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.settings.get_profile", return_value={}):
            resp = client.put("/api/profile/none", json={
                "name": "none", "model_provider": "x", "model_name": "y",
            })
        assert resp.status_code == 404

    def test_delete_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.settings.get_profile", return_value={"model_provider": "x"}), \
             patch("agentica.gateway.routes.settings.delete_profile", return_value=True) as m:
            resp = client.delete("/api/profile/default")
        assert resp.status_code == 200
        m.assert_called_with("default")

    def test_delete_profile_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.settings.get_profile", return_value={}):
            resp = client.delete("/api/profile/none")
        assert resp.status_code == 404
