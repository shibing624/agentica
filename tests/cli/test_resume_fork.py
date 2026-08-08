# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: `resume ... at <uuid>` must branch into a real new session.
"""
from unittest.mock import MagicMock, patch

import pytest

from agentica.memory.session_log import SessionLog
from agentica.workspace import Workspace


class FakeDeepAgent:
    """Stand-in for DeepAgent: create_agent's session wiring is what's tested."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = kwargs.get("session_id")
        self.tools = []


@pytest.fixture
def source_session(tmp_path):
    """A three-message session to branch from, plus its fork point."""
    log = SessionLog("source-session", work_dir=str(tmp_path), user_id=Workspace.DEFAULT_USER_ID)
    log.append("user", "first question")
    fork_point = log.append("assistant", "first answer")
    log.append("user", "second question that the branch must not inherit")
    return log, fork_point


def _create_agent(config, work_dir):
    from agentica.cli.runtime import create_agent

    full_config = {
        "model_provider": "deepseek",
        "model_name": "deepseek-v4-flash",
        "debug": False,
        "work_dir": work_dir,
        "api_key": "fake_openai_key",
        **config,
    }
    with (
        patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
        patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
    ):
        agent = create_agent(full_config, extra_tools=[], workspace=None, skills_registry=None)
    return agent, full_config


def test_resume_at_uuid_creates_a_new_session(source_session, tmp_path):
    log, fork_point = source_session

    agent, _ = _create_agent(
        {"session_id": "source-session", "_resume_at_uuid": fork_point}, str(tmp_path)
    )

    assert agent.session_id != "source-session"
    forked = SessionLog(
        agent.session_id, work_dir=str(tmp_path), user_id=Workspace.DEFAULT_USER_ID
    )
    assert forked.exists()


def test_the_branch_stops_at_the_fork_point(source_session, tmp_path):
    log, fork_point = source_session

    agent, _ = _create_agent(
        {"session_id": "source-session", "_resume_at_uuid": fork_point}, str(tmp_path)
    )

    forked = SessionLog(
        agent.session_id, work_dir=str(tmp_path), user_id=Workspace.DEFAULT_USER_ID
    )
    contents = forked.path.read_text(encoding="utf-8")
    assert "first question" in contents
    assert "second question that the branch must not inherit" not in contents


def test_the_original_session_is_left_intact(source_session, tmp_path):
    log, fork_point = source_session
    before = log.path.read_text(encoding="utf-8")

    _create_agent({"session_id": "source-session", "_resume_at_uuid": fork_point}, str(tmp_path))

    # Appending the branch to the source is the bug this whole path exists to
    # avoid: it would leave two lines of work in one file, neither resumable.
    assert log.path.read_text(encoding="utf-8") == before


def test_the_fork_point_is_consumed_so_a_rebuild_does_not_re_fork(source_session, tmp_path):
    log, fork_point = source_session

    first, config = _create_agent(
        {"session_id": "source-session", "_resume_at_uuid": fork_point}, str(tmp_path)
    )
    # A later `/model` switch rebuilds the agent from the same config dict.
    assert "_resume_at_uuid" not in config
    second, _ = _create_agent(config, str(tmp_path))

    assert second.session_id == first.session_id


def test_forking_the_whole_log_keeps_every_entry(source_session, tmp_path):
    """`/fork` with no point sets `_fork_session`: branch at the tip."""
    log, _fork_point = source_session

    agent, config = _create_agent(
        {"session_id": "source-session", "_fork_session": True}, str(tmp_path)
    )

    assert agent.session_id != "source-session"
    forked = SessionLog(
        agent.session_id, work_dir=str(tmp_path), user_id=Workspace.DEFAULT_USER_ID
    )
    assert "second question that the branch must not inherit" in forked.path.read_text(
        encoding="utf-8"
    )
    # Consumed like `_resume_at_uuid`, so a later `/model` rebuild does not
    # branch a second time from the same session.
    assert "_fork_session" not in config


def test_a_fork_records_its_parent(source_session, tmp_path):
    log, fork_point = source_session

    agent, _ = _create_agent(
        {"session_id": "source-session", "_resume_at_uuid": fork_point}, str(tmp_path)
    )

    forked = SessionLog(
        agent.session_id, work_dir=str(tmp_path), user_id=Workspace.DEFAULT_USER_ID
    )
    assert forked.get_forked_from() == "source-session"
    assert log.get_forked_from() is None


def test_a_plain_resume_keeps_its_session(source_session, tmp_path):
    _log, _fork_point = source_session

    agent, _ = _create_agent({"session_id": "source-session"}, str(tmp_path))

    assert agent.session_id == "source-session"


def test_a_missing_source_log_is_not_a_crash(tmp_path):
    agent, _ = _create_agent(
        {
            "session_id": "never-existed",
            "_resume_at_uuid": "12345678-1234-1234-1234-123456789abc",
        },
        str(tmp_path),
    )

    assert agent.session_id == "never-existed"
