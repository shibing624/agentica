# -*- coding: utf-8 -*-
"""Web session list: CLI sessions behind a pref, temp-dir sessions hidden; fork and feedback."""
import os
from unittest.mock import patch

import pytest

os.environ.setdefault("OPENAI_API_KEY", "fake_openai_key")

pytest.importorskip("fastapi")

from agentica.gateway.services import agent_service as svc_mod  # noqa: E402
from agentica.gateway.services import live_turn  # noqa: E402
from agentica.gateway.services.agent_service import AgentService, is_system_temp_path  # noqa: E402
from agentica.memory.session_log import SessionLog  # noqa: E402


def _session(sid, work_dir, source=None, users=("hello",)):
    log = SessionLog(sid, work_dir=str(work_dir), user_id="default")
    for text in users:
        log.append("user", text)
        log.append("assistant", f"re: {text}")
    if source:
        log.set_source(source)
    return log


@pytest.fixture
def svc(tmp_path):
    live_turn.reset()
    return AgentService(workspace_path=str(tmp_path / "ws"))


@pytest.fixture
def dirs(tmp_path):
    out = {}
    for name in ("proj", "scratch"):
        d = tmp_path / name
        d.mkdir()
        out[name] = d
    # tmp_path itself is under the system temp dir; treat only "scratch" as temp.
    with patch.object(svc_mod, "is_system_temp_path", lambda p: os.path.basename(str(p)) == "scratch"):
        yield out


class TestListFilters:
    def test_cli_and_sdk_sessions_hidden_unless_asked(self, svc, dirs):
        _session("web-1", dirs["proj"], source="gateway")
        _session("cli-1", dirs["proj"], source="cli")
        _session("old-1", dirs["proj"])  # no source: legacy, written by an old CLI
        _session("sdk-1", dirs["proj"], source="sdk")

        shown = {r["session_id"] for r in svc.list_sessions(owner="default", scope="all", include_cli=False)}
        assert shown == {"web-1"}
        everything = {r["session_id"] for r in svc.list_sessions(owner="default", scope="all")}
        assert everything == {"web-1", "cli-1", "old-1", "sdk-1"}

    def test_temp_dir_sessions_hidden(self, svc, dirs):
        _session("real", dirs["proj"], source="gateway")
        _session("demo", dirs["scratch"], source="gateway")
        shown = {r["session_id"] for r in svc.list_sessions(owner="default", scope="all", hide_temp_dirs=True)}
        assert shown == {"real"}

    def test_a_running_session_is_always_listed(self, svc, dirs):
        _session("cli-run", dirs["scratch"], source="cli")
        with patch.object(svc, "is_session_active", side_effect=lambda sid, owner=None: sid == "cli-run"):
            rows = svc.list_sessions(owner="default", scope="all", include_cli=False, hide_temp_dirs=True)
        assert [r["session_id"] for r in rows] == ["cli-run"]


class TestSystemTempPath:
    def test_paths(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTICA_HOME", "/Users/someone/.agentica")
        assert is_system_temp_path("/tmp/vpet-demo-ag")
        assert is_system_temp_path("/private/tmp/x/proj")
        assert is_system_temp_path("/var/folders/ab/cd/T/vpet-ag-inject/proj")
        assert not is_system_temp_path("/Users/someone/Documents/Codes/agentica")
        assert not is_system_temp_path("")
        assert not is_system_temp_path("/tmpfoo/bar")

    def test_agentica_home_under_tmp_is_not_temp(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        monkeypatch.setenv("AGENTICA_HOME", str(home))
        assert not is_system_temp_path(str(home / "tmp" / "web-chats" / "chat-1"))
        assert is_system_temp_path(str(tmp_path / "elsewhere"))


class TestForkAndFeedback:
    def test_fork_at_an_entry_copies_up_to_it_and_lists_as_web(self, svc, dirs):
        log = _session("orig", dirs["proj"], source="cli", users=("first", "second"))
        rows = list(log._iter_entries())
        first_answer = next(e["uuid"] for e in rows if e.get("type") == "assistant")

        new_id = svc.fork_session("orig", at_uuid=first_answer, owner="default")

        branch = svc.session_log_for(new_id, owner="default")
        texts = [e.get("content") for e in branch._iter_entries() if e.get("type") in ("user", "assistant")]
        assert texts == ["first", "re: first"]
        listed = {r["session_id"]: r for r in svc.list_sessions(owner="default", scope="all", include_cli=False)}
        assert new_id in listed and "orig" not in listed
        assert listed[new_id]["name"] == "first (fork)"
        assert listed[new_id]["source"] == "gateway"

    def test_fork_rejects_unknown_session_and_uuid(self, svc, dirs):
        _session("orig", dirs["proj"])
        with pytest.raises(FileNotFoundError):
            svc.fork_session("nope", owner="default")
        with pytest.raises(ValueError):
            svc.fork_session("orig", at_uuid="not-a-uuid", owner="default")

    def test_feedback_round_trip_and_clear(self, svc, dirs):
        log = _session("s1", dirs["proj"], source="gateway")
        answer = next(e["uuid"] for e in log._iter_entries() if e.get("type") == "assistant")

        fb = svc.set_session_feedback("s1", answer, "down", "wrong file", owner="default")
        assert fb[answer]["rating"] == "down" and fb[answer]["comment"] == "wrong file"
        assert svc.session_feedback("s1", owner="default")[answer]["rating"] == "down"

        svc.set_session_feedback("s1", answer, "up", owner="default")
        assert svc.session_feedback("s1", owner="default")[answer] == {
            **svc.session_feedback("s1", owner="default")[answer], "rating": "up", "comment": "",
        }
        svc.set_session_feedback("s1", answer, None, owner="default")
        assert svc.session_feedback("s1", owner="default") == {}

    def test_feedback_validates(self, svc, dirs):
        log = _session("s1", dirs["proj"])
        answer = next(e["uuid"] for e in log._iter_entries() if e.get("type") == "assistant")
        with pytest.raises(ValueError):
            svc.set_session_feedback("s1", answer, "meh", owner="default")
        with pytest.raises(ValueError):
            svc.set_session_feedback("s1", "missing", "up", owner="default")
        # The transcript itself is untouched by a vote.
        before = log.path.read_text()
        svc.set_session_feedback("s1", answer, "up", owner="default")
        assert log.path.read_text() == before
