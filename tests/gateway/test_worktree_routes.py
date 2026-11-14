import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("agentica_gateway")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentica_gateway import deps

ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = ROOT / "gateway" / "src" / "agentica_gateway" / "routes" / "worktree.py"
spec = importlib.util.spec_from_file_location("agentica_gateway.routes.worktree", ROUTE_PATH)
assert spec and spec.loader
worktree_routes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = worktree_routes
spec.loader.exec_module(worktree_routes)


class FakeAgentService:
    def __init__(self, root: Path) -> None:
        self.root = str(root)
        self.work_dirs: dict[tuple[str, str], str] = {}
        self.invalidated: list[tuple[str, str]] = []

    def get_session_work_dir(self, session_id: str, owner: str) -> str:
        return self.work_dirs.get((session_id, owner), self.root)

    def set_session_work_dir(self, session_id: str, work_dir: str, owner: str) -> None:
        self.work_dirs[(session_id, owner)] = work_dir

    def invalidate_session_agent(self, session_id: str, owner: str) -> None:
        self.invalidated.append((session_id, owner))


@pytest.fixture()
def gateway_repo(tmp_path: Path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    service = FakeAgentService(tmp_path)
    app = FastAPI()
    app.include_router(worktree_routes.router)
    previous_service = deps.agent_service
    deps.agent_service = service

    @app.middleware("http")
    async def principal(request, call_next):
        request.state.principal = SimpleNamespace(user_id="alice")
        return await call_next(request)

    with TestClient(app) as client:
        yield client, tmp_path, service
    deps.agent_service = previous_service


def test_worktree_api_binds_the_session_work_dir(gateway_repo):
    client, root, service = gateway_repo
    created = client.post("/api/worktrees", json={
        "session_id": "s1", "action": "new", "name": "feature-x",
    })
    assert created.status_code == 200
    assert "wt/feature-x" in created.json()["output"]

    listed = client.get("/api/worktrees", params={"session_id": "s1"}).json()
    names = [entry["name"] for entry in listed["worktrees"]]
    assert "feature-x" in names

    bound = client.post("/api/worktrees", json={
        "session_id": "s1", "action": "bind", "name": "feature-x",
    })
    assert bound.status_code == 200
    expected = str((root / ".worktrees" / "feature-x").resolve())
    assert bound.json()["work_dir"] == expected
    assert service.get_session_work_dir("s1", "alice") == expected
    assert service.invalidated == [("s1", "alice")]
