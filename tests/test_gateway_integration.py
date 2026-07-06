"""Integration tests for gateway FastAPI endpoints using TestClient.

Tests that exercise the HTTP layer (routes, middleware, response format).
Agent calls are mocked at the AgentService level to avoid real LLM calls.
"""
from unittest.mock import MagicMock, AsyncMock, patch

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
    mock_svc.model_reasoning_effort = ""
    mock_svc._invalidate_cache = AsyncMock()
    mock_svc.reload_profile = AsyncMock()
    mock_svc.has_active_runs = MagicMock(return_value=False)
    mock_svc.save_memory = AsyncMock()
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
        mock_svc.delete_session.assert_called_with("s1")

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
        mock_svc.rename_session.assert_called_with("s1", "New Name")

    def test_rename_empty_name(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/sessions/s1/rename", json={"name": "  "})
        assert resp.status_code == 400

    def test_archive_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/sessions/s1/archive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"
        mock_svc.archive_session.assert_called_with("s1", archived=True)

    def test_unarchive_session(self, mock_app):
        client, mock_svc = mock_app
        resp = client.post("/api/sessions/s1/unarchive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unarchived"
        mock_svc.archive_session.assert_called_with("s1", archived=False)


class TestConfigEndpoints:
    """Test /api/status and /api/models endpoints."""

    def test_status(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "workspace" in data

    def test_list_models(self, mock_app):
        client, _ = mock_app
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        # No hardcoded provider/model catalog — only current binding is returned.
        assert "current_provider" in data
        assert "current_name" in data
        assert "current" in data


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
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.get_job", return_value=self._fake_job()), \
             patch("agentica.gateway.routes.scheduler.cronjob",
                   return_value='{"success":true,"job":{"id":"j1"}}'):
            resp = client.post("/api/scheduler/jobs/j1/trigger")
        assert resp.status_code == 200

    def test_list_runs(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.scheduler.list_task_runs", return_value=[self._fake_run()]):
            resp = client.get("/api/scheduler/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["runs"][0]["job_id"] == "j1"
        assert data["runs"][0]["result_preview"] == "done"

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
            "model_provider": "deepseek", "model_name": "deepseek-v4-flash",
            "base_url": "https://api.deepseek.com", "api_key": "sk-1234567890abcdef",
            "reasoning_effort": "max", "max_tokens": 8192, "context_window": 1000000,
            "temperature": 0.7, "top_p": 0.95,
            "auxiliary_model": {"model_provider": "zhipuai", "model_name": "glm-4.7-flash",
                                "base_url": "https://open.bigmodel.cn", "api_key": "sk-aux"},
            "env": {"SERPER_API_KEY": "xxx"},
        }
        with patch("agentica.gateway.routes.config.get_profile", return_value=fake):
            resp = client.get("/api/profile/default")
        assert resp.status_code == 200
        d = resp.json()
        assert d["model_provider"] == "deepseek"
        assert d["api_key_masked"].startswith("sk-1") and "****" in d["api_key_masked"]
        assert d["has_api_key"] is True
        assert d["env"]["SERPER_API_KEY"] == "xxx"
        assert d["auxiliary_model"]["model_provider"] == "zhipuai"

    def test_get_profile_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.config.get_profile", return_value={}):
            resp = client.get("/api/profile/none")
        assert resp.status_code == 404

    def test_create_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.config.upsert_profile") as m:
            resp = client.post("/api/profile", json={
                "name": "test-p", "model_provider": "deepseek",
                "model_name": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",
                "api_key": "sk-xxx",
            })
        assert resp.status_code == 200
        args, kwargs = m.call_args
        assert args[0] == "test-p"
        assert args[1]["model_provider"] == "deepseek"
        assert args[1]["api_key"] == "sk-xxx"
        assert kwargs["make_active"] is False

    def test_create_profile_missing_fields(self, mock_app):
        client, _ = mock_app
        resp = client.post("/api/profile", json={"name": "x"})
        assert resp.status_code == 422  # Pydantic validation

    def test_update_profile_keeps_api_key(self, mock_app):
        client, _ = mock_app
        existing = {"model_provider": "deepseek", "model_name": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com", "api_key": "sk-existing"}
        with patch("agentica.gateway.routes.config.get_profile", return_value=existing), \
             patch("agentica.gateway.routes.config.upsert_profile") as m:
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
        with patch("agentica.gateway.routes.config.get_profile", return_value={}):
            resp = client.put("/api/profile/none", json={
                "name": "none", "model_provider": "x", "model_name": "y",
            })
        assert resp.status_code == 404

    def test_delete_profile(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.config.get_profile", return_value={"model_provider": "x"}), \
             patch("agentica.gateway.routes.config.delete_profile", return_value=True) as m:
            resp = client.delete("/api/profile/default")
        assert resp.status_code == 200
        m.assert_called_with("default")

    def test_delete_profile_not_found(self, mock_app):
        client, _ = mock_app
        with patch("agentica.gateway.routes.config.get_profile", return_value={}):
            resp = client.delete("/api/profile/none")
        assert resp.status_code == 404
