# -*- coding: utf-8 -*-
"""A chat's background commands over HTTP: the Web's ``/ps`` and ``/stop``."""
import os
import sys
import time

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")
if os.name != "posix":
    pytest.skip("background commands are process-group based here", allow_module_level=True)

from fastapi.testclient import TestClient

from agentica.tools.background_processes import BackgroundProcessRegistry

SESSION = "bg-session"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_AUTH", "false")
    monkeypatch.setenv("AGENTICA_HOME", str(tmp_path / "home"))
    from agentica_gateway import deps
    from agentica_gateway.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        svc = deps.agent_service
        yield c, svc, tmp_path
        svc.stop_all_background_jobs()
        svc._session_background.clear()


def _registry(svc, session_id=SESSION, owner=None):
    return svc._session_background.setdefault(
        svc._sk(session_id, owner), BackgroundProcessRegistry(),
    )


def _start(registry, tmp_path, script):
    return registry.start(f"{sys.executable} -u -c '{script}'", cwd=str(tmp_path))


def _job(client, job_id):
    jobs = client.get(f"/api/sessions/{SESSION}/background").json()["jobs"]
    return next(j for j in jobs if j["id"] == job_id)


def _until_settled(client, job_id):
    deadline = time.time() + 15
    while time.time() < deadline:
        job = _job(client, job_id)
        if job["status"] != "running":
            return job
        time.sleep(0.05)
    raise AssertionError(f"{job_id} never settled")


def test_a_chat_with_no_jobs_lists_none(client):
    c, _svc, _tmp = client
    assert c.get("/api/sessions/never-ran/background").json() == {"jobs": []}


def test_success_failure_and_their_real_durations(client):
    c, svc, tmp_path = client
    registry = _registry(svc)
    good = _start(registry, tmp_path, 'print("hello from job")')
    bad = _start(registry, tmp_path, "raise SystemExit(3)")

    done, failed = _until_settled(c, good.id), _until_settled(c, bad.id)

    assert (done["status"], done["exit_code"]) == ("completed", 0)
    assert (failed["status"], failed["exit_code"]) == ("failed", 3)
    assert done["finished_at_ms"] >= done["started_at_ms"]
    frozen = _job(c, good.id)["finished_at_ms"]
    time.sleep(0.05)
    assert _job(c, good.id)["finished_at_ms"] == frozen
    output = c.get(f"/api/sessions/{SESSION}/background/{good.id}/output").json()
    assert output["output"] == "hello from job"


def test_stop_ends_only_the_named_job_and_reads_as_stopped(client):
    c, svc, tmp_path = client
    registry = _registry(svc)
    target = _start(registry, tmp_path, "import time; time.sleep(60)")
    other = _start(registry, tmp_path, "import time; time.sleep(60)")

    stopped = c.post(f"/api/sessions/{SESSION}/background/{target.id}/stop")

    assert stopped.status_code == 200
    assert _until_settled(c, target.id)["status"] == "stopped"
    assert _job(c, other.id)["status"] == "running"


def test_all_is_not_a_job_id(client):
    """The registry reads the word "all" as every command; a URL must not."""
    c, svc, tmp_path = client
    job = _start(_registry(svc), tmp_path, "import time; time.sleep(60)")

    assert c.post(f"/api/sessions/{SESSION}/background/all/stop").status_code == 404
    assert _job(c, job.id)["status"] == "running"


def test_another_accounts_job_is_invisible_and_unstoppable(client):
    c, svc, tmp_path = client
    job = _start(_registry(svc, owner="someone-else"), tmp_path, "import time; time.sleep(60)")

    assert c.get(f"/api/sessions/{SESSION}/background").json() == {"jobs": []}
    assert c.post(f"/api/sessions/{SESSION}/background/{job.id}/stop").status_code == 404
    assert c.get(f"/api/sessions/{SESSION}/background/{job.id}/output").status_code == 404
    assert job.running


def test_deleting_the_chat_stops_its_commands(client):
    c, svc, tmp_path = client
    job = _start(_registry(svc), tmp_path, "import time; time.sleep(60)")

    svc.delete_session(SESSION)

    assert job.finished.wait(timeout=15)
    assert job.stop_requested is True
    assert c.get(f"/api/sessions/{SESSION}/background").json() == {"jobs": []}
