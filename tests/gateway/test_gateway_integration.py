"""Integration tests for gateway FastAPI endpoints using TestClient.

Tests that exercise the HTTP layer (routes, middleware, response format).
Agent calls are mocked at the AgentService level to avoid real LLM calls.
"""
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import asyncio
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-real")

import pytest

# Guard: only run if fastapi + httpx are available
pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture()
def mock_app():
    """Create a TestClient with deps patched after lifespan initialization.

    The lifespan creates real services; we patch deps.agent_service
    with a mock after startup to intercept all agent calls.
    """
    from agentica_gateway.services.agent_service import ChatResult
    from agentica_gateway.main import app
    from agentica_gateway import deps

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
    mock_svc.has_cached_session = MagicMock(return_value=False)
    mock_svc.is_session_active = MagicMock(return_value=False)
    mock_svc.cancel_session = MagicMock(return_value=False)
    mock_log = MagicMock()
    mock_log.path.exists.return_value = False
    mock_svc.session_log_for = MagicMock(return_value=mock_log)
    mock_svc.read_user_agents_md = AsyncMock(return_value={
        "content": "# User Instructions\n",
        "path": "/tmp/AGENTS.md",
        "empty_template": True,
        "user_id": "default",
    })
    mock_svc.write_user_agents_md = AsyncMock(return_value={
        "content": "be brief\n",
        "path": "/tmp/AGENTS.md",
        "empty_template": False,
        "user_id": "default",
    })
    mock_svc.read_memory_index = AsyncMock(return_value={
        "auto_extract": True,
        "index_path": "/tmp/MEMORY.md",
        "entries": [{"title": "Tea shop", "filename": "user_tea.md", "hook": "wechat tea"}],
        "user_id": "default",
    })
    mock_svc._cache = MagicMock()
    mock_svc._cache.keys = MagicMock(return_value=[])
    mock_svc._workspace = None

    from agentica_gateway.services import live_turn
    live_turn.reset()
    with TestClient(app, raise_server_exceptions=False) as client:
        # Override deps AFTER lifespan has initialized
        original_svc = deps.agent_service
        deps.agent_service = mock_svc
        yield client, mock_svc
        deps.agent_service = original_svc
        live_turn.reset()


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

    def test_chat_accepts_lone_surrogate(self, mock_app):
        """Browser JSON can carry \\udce5; FastAPI must accept it and pass clean text."""
        client, mock_svc = mock_app
        # httpx json= cannot encode a Python lone surrogate; a browser sends the
        # JSON escape, which is how unpaired surrogates actually arrive.
        resp = client.post(
            "/api/chat",
            content='{"message":"本轮\\udce5加回","session_id":"test-session"}',
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        mock_svc.chat.assert_awaited_once()
        passed = mock_svc.chat.await_args.kwargs["message"]
        passed.encode("utf-8")
        assert "\udce5" not in passed
        assert "本轮" in passed
        assert "加回" in passed

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
                on_tool_call("read_file", {"path": "a.py"})
            if on_tool_result:
                on_tool_result("read_file", "print(1)")
            if on_content:
                on_content("hello ")
                on_content("world")
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
        assert kinds[0] == "tool_call"
        assert kinds[1] == "tool_result"
        assert kinds[-2] == "status"
        assert kinds[-1] == "done"
        contents = [e["data"] for e in events if e["event"] == "content"]
        assert "".join(contents) == "hello world"
        assert events[0]["data"]["name"] == "read_file"
        assert events[-1]["data"]["status"] == "complete"


class TestSessionEndpoints:
    """Test /api/sessions endpoints."""

    @pytest.fixture(autouse=True)
    def _cli_sessions_hidden(self):
        # The pref lives in $AGENTICA_HOME/gateway/prefs; do not read the real one.
        with patch("agentica_gateway.routes.chat.show_cli_sessions_pref", return_value=False):
            yield

    def test_list_sessions(self, mock_app):
        client, mock_svc = mock_app
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        sessions = data["sessions"]
        assert len(sessions) == 2
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["name"] == "Chat 1"
        assert sessions[1]["session_id"] == "s2"
        assert sessions[1]["archived"] is True
        assert data["scope"] == "recent"
        mock_svc.list_sessions.assert_called_with(
            owner="default", scope="recent", recent_days=30, include_cli=False, hide_temp_dirs=True,
        )

    def test_list_all_sessions(self, mock_app):
        client, mock_svc = mock_app
        resp = client.get("/api/sessions?scope=all&recent_days=90")
        assert resp.status_code == 200
        assert resp.json()["scope"] == "all"
        mock_svc.list_sessions.assert_called_with(
            owner="default", scope="all", recent_days=90, include_cli=False, hide_temp_dirs=True,
        )

    def test_list_sessions_follows_the_show_cli_sessions_pref(self, mock_app):
        client, mock_svc = mock_app
        with patch("agentica_gateway.routes.chat.show_cli_sessions_pref", return_value=True):
            client.get("/api/sessions")
        assert mock_svc.list_sessions.call_args.kwargs["include_cli"] is True

    def test_fork_session(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.fork_session = MagicMock(return_value="new-id")
        resp = client.post("/api/sessions/s1/fork", json={"at_uuid": "u-3"})
        assert resp.status_code == 200 and resp.json() == {"session_id": "new-id"}
        mock_svc.fork_session.assert_called_with("s1", "u-3", owner="default")
        mock_svc.fork_session.side_effect = FileNotFoundError("x")
        assert client.post("/api/sessions/nope/fork", json={}).status_code == 404
        mock_svc.fork_session.side_effect = ValueError("No entry")
        assert client.post("/api/sessions/s1/fork", json={"at_uuid": "zz"}).status_code == 400

    def test_session_feedback(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.set_session_feedback = MagicMock(return_value={"u-3": {"rating": "down", "comment": "x"}})
        mock_svc.session_feedback = MagicMock(return_value={"u-3": {"rating": "down", "comment": "x"}})
        resp = client.post("/api/sessions/s1/feedback", json={"uuid": "u-3", "rating": "down", "comment": "x"})
        assert resp.status_code == 200
        mock_svc.set_session_feedback.assert_called_with("s1", "u-3", "down", "x", owner="default")
        assert client.get("/api/sessions/s1/feedback").json()["feedback"]["u-3"]["rating"] == "down"
        mock_svc.set_session_feedback.side_effect = ValueError("rating")
        assert client.post("/api/sessions/s1/feedback", json={"uuid": "u-3", "rating": "meh"}).status_code == 400

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

    def test_pin_and_unpin_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/sessions/s1/pin", json={"pinned": True})
        assert resp.status_code == 200 and resp.json()["status"] == "pinned"
        mock_svc.pin_session.assert_called_with("s1", pinned=True, owner="default")
        resp = client.post("/api/sessions/s1/pin", json={"pinned": False})
        assert resp.json()["status"] == "unpinned"
        mock_svc.pin_session.assert_called_with("s1", pinned=False, owner="default")

    def test_pinned_flag_round_trips_through_the_sidecar(self, tmp_path):
        from agentica.memory.session_log import SessionLog

        log = SessionLog(session_id="s1", base_dir=str(tmp_path))
        log.append("user", "hi")
        assert SessionLog.list_sessions(base_dir=str(tmp_path))[0]["pinned"] is False
        SessionLog.pin_session("s1", pinned=True, base_dir=str(tmp_path))
        assert SessionLog.list_sessions(base_dir=str(tmp_path))[0]["pinned"] is True
        SessionLog.pin_session("s1", pinned=False, base_dir=str(tmp_path))
        assert SessionLog.list_sessions(base_dir=str(tmp_path))[0]["pinned"] is False

    def test_mark_session_read(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.mark_session_read = MagicMock(return_value="2026-01-01T00:00:00.000Z")
        resp = client.post("/api/sessions/s1/read")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "read"
        assert data["last_read_at"] == "2026-01-01T00:00:00.000Z"
        mock_svc.mark_session_read.assert_called_with("s1", owner="default")

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
        import agentica_gateway.routes.settings as settings_routes
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
        import agentica_gateway.routes.settings as settings_routes
        monkeypatch.setattr(settings_routes, "AGENTICA_HOME", str(tmp_path))
        saved = client.put("/api/prefs", json={"auto_extract_memory": False})
        assert saved.status_code == 200
        assert saved.json()["auto_extract_memory"] is False
        mock_svc._invalidate_cache.assert_awaited()
        assert client.get("/api/prefs").json()["auto_extract_memory"] is False


class TestMemoryEndpoints:
    def test_get_user_agents_md(self, mock_app):
        client, mock_svc = mock_app
        resp = client.get("/api/user_agents_md")
        assert resp.status_code == 200
        assert resp.json()["path"] == "/tmp/AGENTS.md"
        mock_svc.read_user_agents_md.assert_awaited_once()

    def test_put_user_agents_md(self, mock_app):
        client, mock_svc = mock_app
        resp = client.put("/api/user_agents_md", json={"content": "be brief\n"})
        assert resp.status_code == 200
        assert resp.json()["content"] == "be brief\n"
        mock_svc.write_user_agents_md.assert_awaited_once()

    def test_get_memory_lists_index_entries(self, mock_app):
        client, mock_svc = mock_app
        resp = client.get("/api/memory")
        assert resp.status_code == 200
        body = resp.json()
        assert body["index_path"] == "/tmp/MEMORY.md"
        assert body["entries"][0]["title"] == "Tea shop"
        mock_svc.read_memory_index.assert_awaited_once()

    def test_put_memory_is_not_allowed(self, mock_app):
        client, _ = mock_app
        resp = client.put("/api/memory", json={"content": "be brief\n"})
        assert resp.status_code == 405

    def test_get_memory_entry_returns_the_topic_file_body(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.read_memory_entry = AsyncMock(return_value={
            "filename": "user_tea.md", "title": "Tea shop", "hook": "wechat tea",
            "content": "Prefers oolong.\n", "origin_work_dir": "/tmp/shop",
            "path": "/tmp/memory/user_tea.md", "user_id": "default",
        })
        resp = client.get("/api/memory/entry", params={"filename": "user_tea.md"})
        assert resp.status_code == 200
        assert resp.json()["content"] == "Prefers oolong.\n"
        assert resp.json()["path"] == "/tmp/memory/user_tea.md"
        mock_svc.read_memory_entry.assert_awaited_once()

    def test_get_memory_entry_404_on_unknown_file(self, mock_app):
        client, mock_svc = mock_app
        mock_svc.read_memory_entry = AsyncMock(return_value=None)
        resp = client.get("/api/memory/entry", params={"filename": "../AGENTS.md"})
        assert resp.status_code == 404

    def test_get_memory_entry_requires_filename(self, mock_app):
        client, _ = mock_app
        assert client.get("/api/memory/entry").status_code == 422


class TestOpenUrlAndWechatQr:
    # conftest's `opened_externally` replaces settings._open_os for every test,
    # so these never spawn anything. The route tests assert what was handed to
    # the desktop; the per-OS argv is the pure _open_command, tested below.

    def test_open_url(self, mock_app, opened_externally):
        client, _ = mock_app
        resp = client.post("/api/open", json={"url": "http://127.0.0.1:8881/chat"})
        assert resp.status_code == 200
        assert opened_externally == [("http://127.0.0.1:8881/chat", "finder")]

    def test_open_url_rejects_javascript(self, mock_app, opened_externally):
        client, _ = mock_app
        resp = client.post("/api/open", json={"url": "javascript:alert(1)"})
        assert resp.status_code == 400
        assert opened_externally == []

    def test_open_path_passes_the_app_through(self, mock_app, opened_externally, tmp_path):
        client, _ = mock_app
        target = tmp_path / "notes.md"
        target.write_text("x", encoding="utf-8")
        assert client.post("/api/open", json={"path": str(target), "app": "reveal"}).status_code == 200
        assert opened_externally == [(str(target), "reveal")]

    def test_open_missing_path_opens_nothing(self, mock_app, opened_externally, tmp_path):
        client, _ = mock_app
        assert client.post("/api/open", json={"path": str(tmp_path / "nope")}).status_code == 404
        assert opened_externally == []

    def test_reveal_selects_the_file_in_its_folder_on_linux(self, tmp_path):
        from agentica_gateway.routes.settings import _open_command

        target = tmp_path / "notes.md"
        target.write_text("x", encoding="utf-8")
        assert _open_command(str(target), "reveal", platform="linux") == ["xdg-open", str(tmp_path)]

    def test_reveal_with_a_relative_path_never_opens_an_empty_string(self, monkeypatch, tmp_path):
        from agentica_gateway.routes.settings import _open_command

        (tmp_path / "notes.md").write_text("x", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        cmd = _open_command("notes.md", "reveal", platform="linux")
        assert cmd[0] == "xdg-open" and cmd[1], "xdg-open got an empty folder"

    def test_open_commands_on_macos(self):
        from agentica_gateway.routes.settings import _open_command

        assert _open_command("/x/a.yaml", "editor", platform="darwin") == ["open", "-t", "/x/a.yaml"]
        assert _open_command("/x/a.yaml", "reveal", platform="darwin") == ["open", "-R", "/x/a.yaml"]
        assert _open_command("/x", "terminal", platform="darwin") == ["open", "-a", "Terminal", "/x"]
        assert _open_command("/x", "finder", platform="darwin") == ["open", "/x"]

    def test_open_commands_on_windows(self):
        from agentica_gateway.routes.settings import _open_command

        assert _open_command(r"C:\a.txt", "reveal", platform="win32") == ["explorer", r"/select,C:\a.txt"]
        assert _open_command(r"C:\a.txt", "editor", platform="win32") == ["notepad", r"C:\a.txt"]

    def test_only_installed_apps_are_offered_and_open_with_their_bundle(self, monkeypatch):
        import agentica_gateway.routes.settings as settings_routes

        installed = {"/Applications/Cursor.app", "/System/Library/CoreServices/Finder.app", "/Applications/cmux.app"}
        monkeypatch.setattr(settings_routes.os.path, "isdir", lambda p: p in installed)
        apps = settings_routes.installed_apps(platform="darwin")
        assert [(a["id"], a["kind"]) for a in apps] == [("finder", "files"), ("cursor", "editor"), ("cmux", "terminal")]
        assert settings_routes._open_command("/x", "cursor", platform="darwin") == [
            "open", "-a", "/Applications/Cursor.app", "/x"]
        assert settings_routes._open_command("/x", "cmux", platform="darwin") == [
            "open", "-a", "/Applications/cmux.app", "/x"]
        # An app the catalog knows but this machine lacks opens nothing.
        assert settings_routes._open_command("/x", "iterm", platform="darwin") is None
        # Finder keeps the default-app route, Terminal its own.
        assert settings_routes._open_command("/x", "finder", platform="darwin") == ["open", "/x"]

    def test_open_in_a_catalog_app_reaches_the_desktop(self, mock_app, opened_externally, tmp_path):
        client, _ = mock_app
        resp = client.post("/api/open", json={"path": str(tmp_path), "app": "cursor"})
        assert resp.status_code == 200
        assert opened_externally == [(str(tmp_path), "cursor")]

    def test_linux_terminal_without_any_terminal_opens_nothing(self, monkeypatch):
        import agentica_gateway.routes.settings as settings_routes

        monkeypatch.setattr(settings_routes.shutil, "which", lambda name: None)
        assert settings_routes._open_command("/x", "terminal", platform="linux") is None

    def test_open_rejects_an_unknown_app(self, mock_app):
        client, _ = mock_app
        assert client.post("/api/open", json={"url": "http://example.com", "app": "shell"}).status_code == 422

    def test_wechat_qr_start_and_poll(self, mock_app):
        client, _ = mock_app
        from agentica_gateway import deps
        from agentica_gateway.channels.base import ChannelType

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
        import agentica_gateway.routes.settings as settings_routes
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
    """/api/scheduler/* against a real (temporary) cron store.

    Runs are not executed here: "run now" is pinned to start in the
    background and say which conversation it lands in.
    """

    @pytest.fixture
    def cron_store(self, tmp_path):
        cron_dir = tmp_path / "cron"
        with patch("agentica.cron.jobs.CRON_DIR", cron_dir), \
             patch("agentica.cron.jobs.JOBS_FILE", cron_dir / "jobs.json"), \
             patch("agentica.cron.jobs.OUTPUT_DIR", cron_dir / "output"), \
             patch("agentica.cron.jobs.RUNS_FILE", cron_dir / "runs.jsonl"):
            yield cron_dir

    def _create(self, client, **body):
        payload = {"prompt": "daily report", "schedule": "30 7 * * *", "name": "rep", **body}
        resp = client.post("/api/scheduler/jobs", json=payload)
        assert resp.status_code == 200, resp.text
        return resp.json()["job"]

    def test_create_lists_with_the_editor_fields(self, mock_app, cron_store):
        client, _ = mock_app
        job = self._create(client, timezone="UTC")
        jobs = client.get("/api/scheduler/jobs").json()["jobs"]
        assert [j["id"] for j in jobs] == [job["id"]]
        assert jobs[0]["schedule_spec"] == {"kind": "cron", "expression": "30 7 * * *", "timezone": "UTC"}
        assert jobs[0]["target"] == "new_once" and jobs[0]["session_id"] == ""
        assert jobs[0]["next_run_at_ms"] > 0

    def test_create_ignores_a_user_id_in_the_body(self, mock_app, cron_store):
        client, _ = mock_app
        job = self._create(client, user_id="kk")
        assert job["user_id"] == "default"

    def test_target_session_needs_an_existing_conversation_of_the_caller(self, mock_app, cron_store):
        client, svc = mock_app
        resp = client.post("/api/scheduler/jobs", json={
            "prompt": "p", "schedule": "1h", "target": "session", "session_id": "nope",
        })
        assert resp.status_code == 400
        svc.session_log_for.return_value.path.exists.return_value = True
        job = self._create(client, target="session", session_id="chat-1")
        assert job["target"] == "session" and job["session_id"] == "chat-1"

    def test_work_dir_is_the_directory_created_conversations_use(self, mock_app, cron_store, tmp_path):
        client, _ = mock_app
        job = self._create(client, work_dir=str(tmp_path))
        assert job["work_dir"] == str(tmp_path)

    def test_a_job_owned_by_someone_else_is_not_found(self, mock_app, cron_store):
        from agentica.cron.jobs import create_job
        client, _ = mock_app
        other = create_job(prompt="p", schedule="1h", user_id="kk")
        assert client.get(f"/api/scheduler/jobs/{other.id}").status_code == 404
        assert client.post(f"/api/scheduler/jobs/{other.id}/pause").status_code == 404
        assert client.delete(f"/api/scheduler/jobs/{other.id}").status_code == 404
        assert client.get("/api/scheduler/jobs").json()["jobs"] == []

    def test_update_reschedules_and_changes_timezone(self, mock_app, cron_store):
        client, _ = mock_app
        job = self._create(client)
        resp = client.put(f"/api/scheduler/jobs/{job['id']}", json={"schedule": "0 9 * * 1-5", "timezone": "UTC"})
        assert resp.status_code == 200, resp.text
        spec = resp.json()["job"]["schedule_spec"]
        assert spec == {"kind": "cron", "expression": "0 9 * * 1-5", "timezone": "UTC"}
        assert client.put(f"/api/scheduler/jobs/{job['id']}", json={}).status_code == 400

    def test_pause_resume_delete(self, mock_app, cron_store):
        client, _ = mock_app
        job = self._create(client)
        assert client.post(f"/api/scheduler/jobs/{job['id']}/pause").json()["job"]["status"] == "paused"
        assert client.post(f"/api/scheduler/jobs/{job['id']}/resume").json()["job"]["status"] == "active"
        assert client.delete(f"/api/scheduler/jobs/{job['id']}").status_code == 200
        assert client.get("/api/scheduler/jobs").json()["jobs"] == []

    def test_trigger_starts_a_run_and_names_its_conversation(self, mock_app, cron_store):
        from agentica_gateway import deps
        client, _ = mock_app
        job = self._create(client)
        started = []
        with patch.object(deps.scheduler, "_spawn", side_effect=lambda j, sid: started.append((j.id, sid))):
            resp = client.post(f"/api/scheduler/jobs/{job['id']}/trigger")
        assert resp.status_code == 200
        run = resp.json()["run"]
        assert run["status"] == "queued" and run["session_id"]
        assert started == [(job["id"], run["session_id"])]

    def test_trigger_while_running_is_a_conflict(self, mock_app, cron_store):
        from agentica.cron import scheduler as sched
        client, _ = mock_app
        job = self._create(client)
        sched._running_jobs.add(job["id"])
        try:
            assert client.post(f"/api/scheduler/jobs/{job['id']}/trigger").status_code == 409
        finally:
            sched._running_jobs.discard(job["id"])

    def test_runs_carry_the_conversation(self, mock_app, cron_store):
        from agentica.cron.jobs import mark_job_run
        from agentica.cron.types import RunStatus
        client, _ = mock_app
        job = self._create(client)
        mark_job_run(job["id"], RunStatus.OK, result="done", session_id="conv-1")
        runs = client.get(f"/api/scheduler/jobs/{job['id']}/runs").json()["runs"]
        assert runs[0]["session_id"] == "conv-1" and runs[0]["result_full"] == "done"
        # A new_once job keeps delivering into that conversation.
        assert client.get(f"/api/scheduler/jobs/{job['id']}").json()["job"]["session_id"] == "conv-1"
        assert client.get("/api/scheduler/runs").json()["total"] == 1

    def test_clear_completed_keeps_active_and_paused(self, mock_app, cron_store):
        from agentica.cron.jobs import update_job
        client, _ = mock_app
        done, active = self._create(client), self._create(client)
        update_job(done["id"], {"status": "completed", "enabled": False})
        assert client.post("/api/scheduler/jobs/clear_completed").json() == {"deleted": 1}
        assert [j["id"] for j in client.get("/api/scheduler/jobs").json()["jobs"]] == [active["id"]]

    def test_status_toggle_starts_and_stops_the_ticker(self, mock_app, cron_store):
        client, _ = mock_app
        with patch("agentica_gateway.services.scheduler_service.set_setting") as saved:
            assert client.put("/api/scheduler/status", json={"enabled": True}).json()["enabled"] is True
            assert client.get("/api/scheduler/status").json()["enabled"] is True
            assert client.put("/api/scheduler/status", json={"enabled": False}).json()["enabled"] is False
        assert [c.args for c in saved.call_args_list] == [("cron.enabled", True), ("cron.enabled", False)]


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
        with patch("agentica_gateway.routes.settings.get_profile", return_value=fake):
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
        with patch("agentica_gateway.routes.settings.get_profile", return_value={}):
            resp = client.get("/api/profile/none")
        assert resp.status_code == 404

    def test_create_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica_gateway.routes.settings.upsert_profile") as m:
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
        with patch("agentica_gateway.routes.settings.upsert_profile") as upsert:
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
        with patch("agentica_gateway.routes.settings.get_profile", return_value=existing), \
             patch("agentica_gateway.routes.settings.upsert_profile") as m:
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
        with patch("agentica_gateway.routes.settings.get_profile", return_value={}):
            resp = client.put("/api/profile/none", json={
                "name": "none", "model_provider": "x", "model_name": "y",
            })
        assert resp.status_code == 404

    def test_delete_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica_gateway.routes.settings.get_profile", return_value={"model_provider": "x"}), \
             patch("agentica_gateway.routes.settings.delete_profile", return_value=True) as m:
            resp = client.delete("/api/profile/default")
        assert resp.status_code == 200
        m.assert_called_with("default")

    def test_delete_profile_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica_gateway.routes.settings.get_profile", return_value={}):
            resp = client.delete("/api/profile/none")
        assert resp.status_code == 404


class TestChatRunRegistry:
    """Background run: disconnect does not cancel; Stop waits; reconnect replays."""

    def test_create_run_returns_id_immediately(self, mock_app):
        from agentica_gateway.services.agent_service import ChatResult
        client, mock_svc = mock_app
        mock_svc.chat_stream = AsyncMock(return_value=ChatResult(
            content="hi", session_id="s1",
        ))
        resp = client.post("/api/chat/runs", json={"message": "hello", "session_id": "s1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"]
        assert body["session_id"] == "s1"
        assert body["status"] in ("starting", "running", "completed")

    def test_active_run_and_events_replay(self, mock_app):
        from agentica_gateway.services.agent_service import ChatResult
        client, mock_svc = mock_app

        async def fake_stream(*_a, on_content=None, **_k):
            if on_content:
                on_content("hello")
            return ChatResult(content="hello", session_id="s1")

        mock_svc.chat_stream = fake_stream
        created = client.post("/api/chat/runs", json={"message": "hi", "session_id": "s1"})
        run_id = created.json()["run_id"]
        events = []
        with client.stream("GET", f"/api/chat/runs/{run_id}/events") as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
        kinds = [e.get("event") for e in events]
        assert "content" in kinds or "done" in kinds
        if events:
            assert "seq" in events[0]

    def test_cancel_missing_run_is_completed(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/chat/runs/missing/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_second_run_on_same_session_409s_while_active(self, mock_app):
        from agentica_gateway.services.agent_service import ChatResult
        client, mock_svc = mock_app
        gate = {"release": False}

        async def slow_stream(*_a, **_k):
            for _ in range(50):
                if gate["release"]:
                    break
                await asyncio.sleep(0.02)
            return ChatResult(content="later", session_id="busy")

        mock_svc.chat_stream = slow_stream
        first = client.post("/api/chat/runs", json={"message": "one", "session_id": "busy"})
        assert first.status_code == 200
        second = client.post("/api/chat/runs", json={"message": "two", "session_id": "busy"})
        assert second.status_code == 409
        gate["release"] = True
        run_id = first.json()["run_id"]
        client.post(f"/api/chat/runs/{run_id}/cancel")

    def test_disconnect_does_not_cancel_run(self, mock_app):
        from agentica_gateway.services import live_turn
        from agentica_gateway.services.agent_service import ChatResult
        client, mock_svc = mock_app
        gate = {"release": False}

        async def slow_stream(*_a, on_content=None, **_k):
            if on_content:
                on_content("partial")
            for _ in range(80):
                if gate["release"]:
                    break
                await asyncio.sleep(0.02)
            if on_content:
                on_content(" done")
            return ChatResult(content="partial done", session_id="keep")

        mock_svc.chat_stream = slow_stream
        created = client.post("/api/chat/runs", json={"message": "hi", "session_id": "keep"})
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        assert live_turn.active("keep", owner="default") is not None
        active = client.get("/api/chat/runs/active", params={"session_id": "keep"})
        assert active.status_code == 200
        assert active.json()["run"]["run_id"] == run_id
        missing = client.get("/api/chat/runs/no-such-run/events")
        assert missing.status_code == 404
        gate["release"] = True
        client.post(f"/api/chat/runs/{run_id}/cancel")
