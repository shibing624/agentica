# -*- coding: utf-8 -*-
"""Review-tab git actions: commit only the turn's files, push, admin only,
messages and titles on the side-task (auxiliary-first) model."""
import asyncio
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")
if shutil.which("git") is None:
    pytest.skip("requires git", allow_module_level=True)

from fastapi.testclient import TestClient


def _run(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture()
def repo(tmp_path):
    remote = tmp_path / "remote.git"
    work = tmp_path / "work"
    _run(tmp_path, "init", "--bare", "-b", "main", str(remote))
    _run(tmp_path, "init", "-b", "main", str(work))
    _run(work, "config", "user.email", "t@example.com")
    _run(work, "config", "user.name", "t")
    (work / "a.txt").write_text("one\n")
    (work / "other.txt").write_text("x\n")
    _run(work, "add", ".")
    _run(work, "commit", "-m", "init")
    _run(work, "remote", "add", "origin", str(remote))
    _run(work, "push", "-u", "origin", "main")
    return work, remote


@pytest.fixture()
def client():
    from agentica_gateway.main import app

    with TestClient(app) as c:
        yield c


def test_commit_takes_only_the_selected_files_and_pushes(client, repo):
    work, remote = repo
    (work / "a.txt").write_text("two\n")
    (work / "new.txt").write_text("fresh\n")
    (work / "other.txt").write_text("someone else's edit\n")
    # Something unrelated already staged must stay out of this commit.
    _run(work, "add", "other.txt")

    st = client.post("/api/git/status", json={"cwd": str(work), "files": [str(work / "a.txt"), "new.txt"]})
    assert st.status_code == 200, st.text
    assert st.json()["branch"] == "main" and st.json()["upstream"] == "origin/main"
    assert sorted(st.json()["changed"]) == ["a.txt", "new.txt"]

    res = client.post("/api/git/commit", json={
        "cwd": str(work), "files": [str(work / "a.txt"), "new.txt"], "message": "Update a, add new", "push": True,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["pushed"] is True
    assert sorted(body["files"]) == ["a.txt", "new.txt"]

    committed = _run(work, "show", "--name-only", "--pretty=%s", "HEAD").split()
    assert committed[0] == "Update" and sorted(committed[-2:]) == ["a.txt", "new.txt"]
    assert "other.txt" not in committed
    # other.txt is still staged, untouched by the commit.
    assert _run(work, "diff", "--cached", "--name-only").split() == ["other.txt"]
    # The push reached the remote.
    assert _run(remote, "log", "-1", "--pretty=%s", "main").strip() == "Update a, add new"


def test_status_marks_gitignored_files(client, repo):
    work, _ = repo
    (work / ".gitignore").write_text("exp/\n")
    (work / "exp").mkdir()
    (work / "exp" / "notes.md").write_text("draft\n")
    (work / "a.txt").write_text("edited\n")
    st = client.post("/api/git/status", json={"cwd": str(work), "files": ["exp/notes.md", "a.txt"]}).json()
    assert st["changed"] == ["a.txt"]
    assert st["ignored"] == ["exp/notes.md"]


def test_nothing_to_commit_and_outside_paths_are_refused(client, repo):
    work, _ = repo
    clean = client.post("/api/git/commit", json={"cwd": str(work), "files": ["a.txt"], "message": "m"})
    assert clean.status_code == 409
    outside = client.post("/api/git/commit", json={"cwd": str(work), "files": ["/etc/hosts"], "message": "m"})
    assert outside.status_code == 400
    empty = client.post("/api/git/commit", json={"cwd": str(work), "files": ["a.txt"], "message": "  "})
    assert empty.status_code == 400


def test_git_actions_are_admin_only():
    from fastapi import HTTPException
    from agentica_gateway.auth import Principal
    from agentica_gateway.routes.git import _admin

    class _Req:
        class state:
            principal = Principal("alice", "session", "user")

    with pytest.raises(HTTPException) as excinfo:
        _admin(_Req())
    assert excinfo.value.status_code == 403
    _Req.state.principal = Principal("default", "session", "admin")
    _admin(_Req())


def test_commit_message_is_drafted_on_the_side_task_model(client, repo):
    from agentica_gateway import deps

    work, _ = repo
    (work / "a.txt").write_text("changed\n")
    with patch.object(deps.agent_service, "side_task", AsyncMock(return_value="```\nFix a\n```")) as side:
        res = client.post("/api/git/message", json={"cwd": str(work), "files": ["a.txt"]})
    assert res.status_code == 200, res.text
    assert res.json()["message"] == "Fix a"
    system, prompt = side.await_args.args
    assert "commit message" in system
    assert "+changed" in prompt and "init" in prompt  # the diff and the history samples


def test_side_task_model_prefers_auxiliary():
    from agentica_gateway.services.agent_service import AgentService

    svc = AgentService.__new__(AgentService)
    aux = SimpleNamespace(name="aux")
    with patch.object(AgentService, "_build_sibling_model", return_value=aux), \
         patch("agentica_gateway.services.agent_service.create_model") as main:
        assert svc.side_task_model() is aux
        main.assert_not_called()
    with patch.object(AgentService, "_build_sibling_model", return_value=None), \
         patch.object(AgentService, "model_provider", "openai"), \
         patch.object(AgentService, "model_name", "main-model"), \
         patch("agentica_gateway.services.agent_service.create_model", return_value="MAIN") as main:
        assert svc.side_task_model() == "MAIN"
        assert main.call_args.args[:2] == ("openai", "main-model")


def test_first_exchange_names_an_unnamed_session(tmp_path):
    from agentica.memory.session_log import SessionLog
    from agentica_gateway.services.agent_service import AgentService

    svc = AgentService.__new__(AgentService)
    svc._titled, svc._title_tasks = set(), set()
    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    renamed = SessionLog(session_id="s2", base_dir=str(tmp_path))
    renamed.set_name("Mine")
    logs = {"s1": log, "s2": renamed}

    async def go():
        with patch.object(AgentService, "session_log_for", lambda self, sid, owner=None: logs[sid]), \
             patch.object(AgentService, "side_task", AsyncMock(return_value='"修复登录页超时"\n')) as side:
            svc._schedule_title("s1", None, "登录页老是超时", "已修复")
            svc._schedule_title("s2", None, "q", "a")
            svc._schedule_title("s1", None, "again", "again")  # one attempt per session
            await asyncio.gather(*svc._title_tasks)
            return side

    side = asyncio.run(go())
    assert log.get_name() == "修复登录页超时"
    assert renamed.get_name() == "Mine"
    assert side.await_count == 1
