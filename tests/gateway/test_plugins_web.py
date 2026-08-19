# -*- coding: utf-8 -*-
"""Plugins panel over HTTP: skills CRUD must see its own writes, MCP CRUD."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Never touch the real ~/.agentica: skills are files and MCP is a JSON file.
    home = tmp_path / "home"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setenv("AGENTICA_HOME", str(home))
    monkeypatch.setenv("AGENTICA_SKILL_DIR", str(skills))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")

    from agentica.gateway import deps
    from agentica.gateway.main import app
    from agentica.gateway.routes import plugins

    monkeypatch.setattr(plugins, "AGENTICA_HOME", str(home))
    monkeypatch.setattr(plugins, "AGENTICA_SKILL_DIR", str(skills))
    # Only this project's dir is searched, so the developer's own
    # ~/.claude/skills tree cannot leak into the assertions.
    monkeypatch.setattr(
        plugins.SkillLoader, "get_search_paths",
        lambda self, include_system=False: [(skills, "user")],
    )

    svc = MagicMock()
    svc._ensure_initialized = AsyncMock()
    svc._invalidate_cache = AsyncMock()
    with TestClient(app, raise_server_exceptions=False) as c:
        original = deps.agent_service
        deps.agent_service = svc
        yield c, svc, home
        deps.agent_service = original


def _names(resp):
    return [s["name"] for s in resp.json()["skills"]]


class TestSkillsCrud:
    """The registry is process-global and keeps the first entry per name, so a
    route that only calls load_all() serves a stale listing forever: an edit
    returned 200 with the old body still shown, and a delete kept listing the
    skill. Every one of these assertions fails on load_all()."""

    def test_created_skill_appears_in_the_listing(self, client):
        c, svc, _ = client
        assert _names(c.get("/api/skills")) == []
        created = c.post("/api/skills", json={
            "name": "web-skill", "description": "made by the panel",
            "trigger": "/web", "content": "# Web\nBODY-ONE",
        })
        assert created.status_code == 200, created.text
        listed = c.get("/api/skills")
        assert _names(listed) == ["web-skill"]
        row = listed.json()["skills"][0]
        assert row["editable"] is True and row["trigger"] == "/web"
        # An agent already built for this session has the old catalogue frozen.
        svc._invalidate_cache.assert_awaited()

    def test_update_is_visible_to_the_next_read(self, client):
        c, _, _ = client
        c.post("/api/skills", json={
            "name": "web-skill", "description": "first", "content": "# Web\nBODY-ONE",
        })
        updated = c.put("/api/skills/web-skill", json={
            "description": "second", "content": "# Web\nBODY-TWO",
        })
        assert updated.status_code == 200, updated.text
        detail = c.get("/api/skills/web-skill").json()
        assert "BODY-TWO" in detail["content"]
        assert detail["description"] == "second"

    def test_omitted_fields_keep_their_previous_value(self, client):
        c, _, _ = client
        c.post("/api/skills", json={
            "name": "web-skill", "description": "first",
            "trigger": "/web", "content": "# Web\nBODY-ONE",
        })
        c.put("/api/skills/web-skill", json={"description": "second"})
        detail = c.get("/api/skills/web-skill").json()
        assert detail["trigger"] == "/web"
        assert "BODY-ONE" in detail["content"]

    def test_deleted_skill_leaves_the_listing(self, client):
        c, _, _ = client
        c.post("/api/skills", json={"name": "web-skill", "description": "d", "content": "x"})
        removed = c.delete("/api/skills/web-skill")
        assert removed.status_code == 200
        assert _names(c.get("/api/skills")) == []
        assert c.get("/api/skills/web-skill").status_code == 404

    def test_duplicate_name_is_refused(self, client):
        c, _, _ = client
        c.post("/api/skills", json={"name": "web-skill", "description": "d", "content": "x"})
        again = c.post("/api/skills", json={"name": "web-skill", "description": "d", "content": "x"})
        assert again.status_code == 400


class TestMcpCrud:
    def test_add_list_delete(self, client):
        c, svc, home = client
        added = c.post("/api/mcp/servers", json={
            "name": "ctx7", "command": "npx", "args": ["-y", "ctx7"], "env": {"TOKEN": "x"},
        })
        assert added.status_code == 200, added.text
        row = c.get("/api/mcp/servers").json()["servers"][0]
        assert row["type"] == "stdio" and row["args"] == ["-y", "ctx7"]
        # env values are secrets; only the key names are published.
        assert row["env_keys"] == ["TOKEN"]
        saved = json.loads((home / "mcp_config.json").read_text(encoding="utf-8"))
        assert saved["mcpServers"]["ctx7"]["env"] == {"TOKEN": "x"}
        svc._invalidate_cache.assert_awaited()

        assert c.delete("/api/mcp/servers/ctx7").status_code == 200
        assert c.get("/api/mcp/servers").json()["servers"] == []
        assert c.delete("/api/mcp/servers/ctx7").status_code == 404

    def test_url_server_is_sse(self, client):
        c, _, _ = client
        c.post("/api/mcp/servers", json={"name": "remote", "url": "https://example.com/sse"})
        row = c.get("/api/mcp/servers").json()["servers"][0]
        assert row["type"] == "sse" and row["url"] == "https://example.com/sse"

    def test_neither_command_nor_url_is_refused(self, client):
        c, _, _ = client
        resp = c.post("/api/mcp/servers", json={"name": "empty"})
        assert resp.status_code == 400
