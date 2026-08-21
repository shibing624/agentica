# -*- coding: utf-8 -*-
"""Workspace file browser: containment, list, preview cap, upload."""
import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agentica.gateway.workspace_files import (
    TEXT_PREVIEW_CHARS,
    WorkspacePathError,
    list_entries,
    read_preview_text,
    resolve_existing,
    stat_existing,
)


def test_resolve_refuses_dotdot(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "ok.txt").write_text("hi")
    (tmp_path / "secret.txt").write_text("nope")
    with pytest.raises(WorkspacePathError) as ei:
        resolve_existing(str(root), "../secret.txt")
    assert ei.value.status == 400


def test_list_and_stat(tmp_path):
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.py").write_text("print(1)\n")
    (root / "README.md").write_text("# hi\n")
    entries = list_entries(str(root), "")
    names = [e["name"] for e in entries]
    assert names[0] == "src"
    assert "README.md" in names
    found = stat_existing(str(root), ["README.md", "src/a.py", "missing.py", "../secret"])
    assert found == ["README.md", "src/a.py"]


def test_preview_truncates_at_12000_chars(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    body = "x" * (TEXT_PREVIEW_CHARS + 50)
    (root / "big.txt").write_text(body)
    text, truncated = read_preview_text(root / "big.txt")
    assert truncated is True
    assert len(text) == TEXT_PREVIEW_CHARS


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH", "false")
    from agentica.gateway.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, tmp_path


def test_http_list_preview_upload_and_escape(client):
    c, tmp_path = client
    root = tmp_path / "proj"
    root.mkdir()
    (root / "note.md").write_text("# hello\n")
    listed = c.get("/api/workspace/files", params={"root": str(root), "path": ""})
    assert listed.status_code == 200
    assert any(e["name"] == "note.md" for e in listed.json()["entries"])

    preview = c.get("/api/workspace/content", params={"root": str(root), "path": "note.md", "preview": 1})
    assert preview.status_code == 200
    assert preview.json()["content"].startswith("# hello")
    assert preview.json()["truncated"] is False

    escaped = c.get("/api/workspace/files", params={"root": str(root), "path": ".."})
    assert escaped.status_code == 400

    up = c.post(
        "/api/workspace/upload",
        data={"root": str(root), "path": ""},
        files={"file": ("new.py", b"print(2)\n", "text/x-python")},
        headers={"X-Agentica-Client": "web"},
    )
    assert up.status_code == 200, up.text
    assert (root / "new.py").read_text() == "print(2)\n"

    st = c.post("/api/workspace/stat", json={"root": str(root), "paths": ["note.md", "new.py", "nope"]})
    assert st.status_code == 200
    assert set(st.json()["existing"]) == {"note.md", "new.py"}
