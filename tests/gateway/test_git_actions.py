# -*- coding: utf-8 -*-
"""Review-tab git actions as logic: which argv git runs, what the routes decide
from its output, and which model drafts the commit message.

The old suite built a real bare remote plus a real work repository and pushed
through it. That is environment risk (git on PATH, hooks, child-process latency)
for a layer whose actual decisions are argv construction and output parsing;
``git_ops._git`` — the one seam every git command goes through — is stubbed
here. Real git is exercised by using the product, not by every test run.
Dropped: "the push reached the remote" — that only proved git itself works.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from agentica_gateway import git_ops


class FakeGit:
    """Stands in for ``git_ops._git``: records argv/cwd, answers canned output.

    The real helper returns a ``CompletedProcess``; only ``returncode`` /
    ``stdout`` / ``stderr`` are consumed, so no process is needed.
    """

    def __init__(
        self,
        root,
        *,
        branch="main",
        upstream="origin/main",
        ahead=0,
        behind=0,
        porcelain="",
        ignored=(),
        staged="",
        diff="",
        untracked=(),
        subjects=(),
        sha="abc1234",
        toplevel_code=0,
        push_code=0,
        push_error="",
    ):
        self.root = str(root)
        self.branch = branch
        self.upstream = upstream
        self.ahead = ahead
        self.behind = behind
        self.porcelain = porcelain
        self.ignored = list(ignored)
        self.staged = staged
        self.diff = diff
        self.untracked = list(untracked)
        self.subjects = list(subjects)
        self.sha = sha
        self.toplevel_code = toplevel_code
        self.push_code = push_code
        self.push_error = push_error
        self.calls = []

    # -- the seam ---------------------------------------------------------
    def __call__(self, cwd, *args, timeout=None, check=True):
        self.calls.append((str(cwd), args))
        proc = self._answer(args)
        if check and proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip() or f"git {args[0]} failed"
            # Mirror the real helper: a non-zero exit on a checked call is a
            # 409 the route turns into an HTTP error.
            raise git_ops.GitError(409, detail)
        return proc

    @property
    def paths(self):
        """The argv of every recorded command, in order."""
        return [args for _, args in self.calls]

    # -- canned output ----------------------------------------------------
    @staticmethod
    def _done(stdout="", stderr="", code=0):
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)

    def _answer(self, args):
        head = args[0]
        if head == "rev-parse":
            if "--show-toplevel" in args:
                return self._done(self.root + "\n", code=self.toplevel_code)
            if "--symbolic-full-name" in args:
                if self.upstream:
                    return self._done(self.upstream + "\n")
                return self._done("", "fatal: no upstream", 128)
            if "--short" in args:
                return self._done(self.sha + "\n")
            return self._done(self.branch + "\n")
        if head == "rev-list":
            return self._done(f"{self.behind}\t{self.ahead}\n")
        if head == "status":
            return self._done(self.porcelain)
        if head == "check-ignore":
            return self._done("".join(p + "\n" for p in self.ignored))
        if head == "add":
            return self._done()
        if head == "diff":
            return self._done(self.staged if "--cached" in args else self.diff)
        if head == "ls-files":
            return self._done("".join(p + "\n" for p in self.untracked))
        if head == "commit":
            return self._done()
        if head == "log":
            return self._done("".join(s + "\n" for s in self.subjects))
        if head == "push":
            return self._done("", self.push_error, self.push_code)
        raise AssertionError(f"unexpected git call: {args}")


@pytest.fixture()
def stub_git(tmp_path, monkeypatch):
    """Install a FakeGit rooted at ``tmp_path``; return it for assertions."""

    def install(**kwargs):
        fake = FakeGit(tmp_path, **kwargs)
        monkeypatch.setattr(git_ops, "_git", fake)
        return fake

    return install


@pytest.fixture()
def client():
    from agentica_gateway.main import app

    with TestClient(app) as c:
        yield c


def test_commit_commits_only_the_selected_files_and_pushes(client, stub_git):
    fake = stub_git(
        branch="main",
        upstream="origin/main",
        ahead=1,
        behind=0,
        porcelain=" M a.txt\0?? new.txt\0",
        staged="a.txt\nnew.txt\n",
    )
    cwd = fake.root

    st = client.post(
        "/api/git/status",
        json={"cwd": cwd, "files": [f"{cwd}/a.txt", "new.txt"]},
    )
    assert st.status_code == 200, st.text
    body = st.json()
    assert body["branch"] == "main" and body["upstream"] == "origin/main"
    assert (body["ahead"], body["behind"]) == (1, 0)
    assert sorted(body["changed"]) == ["a.txt", "new.txt"]
    assert body["ignored"] == []

    res = client.post(
        "/api/git/commit",
        json={
            "cwd": cwd,
            "files": [f"{cwd}/a.txt", "new.txt"],
            "message": "Update a, add new",
            "push": True,
        },
    )
    assert res.status_code == 200, res.text
    assert res.json() == {"sha": "abc1234", "files": ["a.txt", "new.txt"], "pushed": True}

    paths = fake.paths
    # The pathspec is the whole decision: the selected files go in, and
    # nothing else already staged can ride along.
    assert ("add", "-A", "--", "a.txt", "new.txt") in paths
    assert ("diff", "--cached", "--name-only", "--", "a.txt", "new.txt") in paths
    assert ("commit", "-m", "Update a, add new", "--", "a.txt", "new.txt") in paths
    assert all("other.txt" not in args for args in paths)
    # A configured upstream is pushed as-is; no explicit branch is forced.
    assert ("push",) in paths
    assert ("rev-parse", "--short", "HEAD") in paths
    # Every command ran in the repository root, not in the caller's subdir.
    assert {cwd for cwd, _ in fake.calls} == {fake.root}


def test_status_reports_ignored_files_and_a_detached_head(client, stub_git):
    fake = stub_git(porcelain=" M a.txt\0", ignored=["exp/notes.md"])
    st = client.post(
        "/api/git/status",
        json={"cwd": fake.root, "files": ["exp/notes.md", "a.txt"]},
    )
    assert st.status_code == 200, st.text
    body = st.json()
    assert body["changed"] == ["a.txt"]
    assert body["ignored"] == ["exp/notes.md"]
    assert ("status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "exp/notes.md", "a.txt") in fake.paths
    assert ("check-ignore", "--", "exp/notes.md", "a.txt") in fake.paths

    fake.branch, fake.upstream = "HEAD", ""
    detached = client.post("/api/git/status", json={"cwd": fake.root, "files": []}).json()
    # A detached head has no branch name and no upstream to compare with.
    assert detached["branch"] == "" and detached["upstream"] == ""
    assert (detached["ahead"], detached["behind"]) == (0, 0)


def test_push_creates_the_upstream_when_missing(client, stub_git):
    fake = stub_git(upstream="", staged="a.txt\n")
    res = client.post(
        "/api/git/commit",
        json={"cwd": fake.root, "files": ["a.txt"], "message": "m", "push": True},
    )
    assert res.status_code == 200, res.text
    assert res.json()["pushed"] is True
    assert ("push", "-u", "origin", "main") in fake.paths


def test_push_failure_is_reported_but_the_commit_stands(client, stub_git):
    fake = stub_git(staged="a.txt\n", push_code=1, push_error="! [rejected] main -> main")
    res = client.post(
        "/api/git/commit",
        json={"cwd": fake.root, "files": ["a.txt"], "message": "m", "push": True},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["sha"] == "abc1234" and body["pushed"] is False
    assert "rejected" in body["pushError"]
    assert any(args[0] == "commit" for args in fake.paths)


def test_detached_head_without_an_upstream_cannot_push(client, stub_git):
    fake = stub_git(branch="HEAD", upstream="", staged="a.txt\n")
    res = client.post(
        "/api/git/commit",
        json={"cwd": fake.root, "files": ["a.txt"], "message": "m", "push": True},
    )
    assert res.status_code == 409
    assert "detached HEAD" in res.json()["detail"]


def test_nothing_to_commit_and_outside_paths_are_refused(client, stub_git):
    fake = stub_git(staged="")

    clean = client.post("/api/git/commit", json={"cwd": fake.root, "files": ["a.txt"], "message": "m"})
    assert clean.status_code == 409
    assert not any(args[0] == "commit" for args in fake.paths)

    outside = client.post("/api/git/commit", json={"cwd": fake.root, "files": ["/etc/hosts"], "message": "m"})
    assert outside.status_code == 400
    assert "outside the repository" in outside.json()["detail"]

    blank = client.post("/api/git/commit", json={"cwd": fake.root, "files": ["a.txt"], "message": "  "})
    assert blank.status_code == 400
    no_files = client.post("/api/git/commit", json={"cwd": fake.root, "files": [], "message": "m"})
    assert no_files.status_code == 400

    # Both body validations happen before git is touched; the outside path is
    # rejected before anything is staged.
    assert fake.paths == [
        ("rev-parse", "--show-toplevel"),
        ("add", "-A", "--", "a.txt"),
        ("diff", "--cached", "--name-only", "--", "a.txt"),
        ("rev-parse", "--show-toplevel"),
    ]


def test_status_outside_a_repository_is_404(client, stub_git):
    fake = stub_git(toplevel_code=128)
    res = client.post("/api/git/status", json={"cwd": fake.root, "files": []})
    assert res.status_code == 404
    assert "not a git repository" in res.json()["detail"]


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


def test_commit_message_is_drafted_on_the_side_task_model(client, stub_git):
    from agentica_gateway import deps

    fake = stub_git(diff="diff --git a/a.txt b/a.txt\n+changed\n", subjects=["init"])
    with patch.object(deps.agent_service, "side_task", AsyncMock(return_value="```\nFix a\n```")) as side:
        res = client.post("/api/git/message", json={"cwd": fake.root, "files": ["a.txt"]})
    assert res.status_code == 200, res.text
    assert res.json()["message"] == "Fix a"
    system, prompt = side.await_args.args
    assert "commit message" in system
    # Both the history samples and the working diff reach the model.
    assert "Recent commit subjects" in prompt and "init" in prompt and "+changed" in prompt
    assert ("diff", "HEAD", "--", "a.txt") in fake.paths
    assert ("log", "-8", "--pretty=%s") in fake.paths


def test_commit_message_with_no_diff_is_a_conflict(client, stub_git):
    from agentica_gateway import deps

    fake = stub_git(diff="")
    with patch.object(deps.agent_service, "side_task", AsyncMock()) as side:
        res = client.post("/api/git/message", json={"cwd": fake.root, "files": ["a.txt"]})
    assert res.status_code == 409
    assert "nothing to commit" in res.json()["detail"]
    side.assert_not_awaited()


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
