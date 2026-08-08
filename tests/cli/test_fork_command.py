# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: `/fork` branches the current chat at a listed message id.
"""
from unittest.mock import MagicMock, patch

import pytest

from agentica.cli.commands.runtime import _cmd_fork
from agentica.memory.session_log import SessionLog
from agentica.workspace import Workspace


class FakeDeepAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = kwargs.get("session_id")
        self.tools = []
        self.working_memory = MagicMock(runs=[])
        self.auxiliary_model = None
        self.model = MagicMock()
        self._session_log = SessionLog(
            self.session_id,
            work_dir=kwargs.get("work_dir"),
            user_id=Workspace.DEFAULT_USER_ID,
        )


@pytest.fixture
def session(tmp_path):
    """Two full exchanges: the branch point is between them."""
    log = SessionLog("live-session", work_dir=str(tmp_path), user_id=Workspace.DEFAULT_USER_ID)
    log.append("user", "first question")
    log.append("assistant", "first answer")
    log.append("user", "second question")
    log.append("assistant", "second answer")
    return log


@pytest.fixture
def ctx(tmp_path, session):
    agent = FakeDeepAgent(session_id="live-session", work_dir=str(tmp_path))
    return MagicMock(
        current_agent=agent,
        agent_config={
            "model_provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "debug": False,
            "work_dir": str(tmp_path),
            "api_key": "fake_openai_key",
            "session_id": "live-session",
        },
        extra_tools=[],
        workspace=None,
        skills_registry=None,
        peer_session=None,
    )


def _fork(ctx, args):
    with (
        patch("agentica.cli.runtime.get_model", return_value=MagicMock()),
        patch("agentica.agent.deep.DeepAgent", FakeDeepAgent),
        patch("agentica.cli.commands.runtime.hydrate_resumed_session", return_value=([], 1)),
        patch("agentica.cli.commands.runtime.display_resumed_transcript", return_value=MagicMock(tool_result_count=0)),
    ):
        return _cmd_fork(ctx, args)


def _forked_log(result, tmp_path):
    return SessionLog(
        result["current_agent"].session_id,
        work_dir=str(tmp_path),
        user_id=Workspace.DEFAULT_USER_ID,
    )


def test_list_only_lists_and_does_not_branch(ctx, tmp_path):
    assert _fork(ctx, "list") is None
    assert ctx.current_agent.session_id == "live-session"


class TestForkAtTheTip:
    """`/fork` with no argument branches here and now, keeping the history."""

    def test_it_creates_a_new_session(self, ctx, tmp_path):
        result = _fork(ctx, "")

        assert result["current_agent"].session_id != "live-session"
        assert _forked_log(result, tmp_path).exists()

    def test_the_whole_conversation_carries_over(self, ctx, tmp_path):
        result = _fork(ctx, "")

        contents = _forked_log(result, tmp_path).path.read_text(encoding="utf-8")
        for text in ("first question", "first answer", "second question", "second answer"):
            assert text in contents

    def test_the_original_session_is_left_intact(self, ctx, tmp_path, session):
        before = session.path.read_text(encoding="utf-8")

        _fork(ctx, "")

        assert session.path.read_text(encoding="utf-8") == before

    def test_the_branch_records_what_it_came_from(self, ctx, tmp_path):
        result = _fork(ctx, "")

        assert _forked_log(result, tmp_path).get_forked_from() == "live-session"

    def test_each_fork_gets_its_own_session(self, ctx, tmp_path):
        first = _fork(ctx, "")
        second = _fork(ctx, "")

        assert first["current_agent"].session_id != second["current_agent"].session_id


def test_forking_at_a_message_number_creates_a_new_session(ctx, tmp_path):
    result = _fork(ctx, "1")

    assert result["current_agent"].session_id != "live-session"
    assert _forked_log(result, tmp_path).exists()


def test_the_branch_drops_the_chosen_message_and_everything_after(ctx, tmp_path):
    # "1" is the newest user message ("second question"), so the branch must end
    # on the answer before it — the state the model was in when it was asked.
    result = _fork(ctx, "1")

    contents = _forked_log(result, tmp_path).path.read_text(encoding="utf-8")
    assert "first answer" in contents
    assert "second question" not in contents
    assert "second answer" not in contents


def test_a_uuid_prefix_selects_the_same_point_as_its_number(ctx, tmp_path, session):
    newest = session.list_user_messages()[0]

    by_number = _fork(ctx, "1")
    by_uuid = _fork(ctx, newest["uuid"][:8])

    # Each branch is its own session, so only the inherited conversation matches.
    assert _forked_log(by_number, tmp_path).load() == _forked_log(by_uuid, tmp_path).load()


def test_forking_at_the_very_first_message_is_refused(ctx, tmp_path):
    # Nothing precedes it, so the branch would be an empty session.
    assert _fork(ctx, "2") is None


def test_an_unknown_fork_point_is_refused(ctx):
    assert _fork(ctx, "9") is None
    assert _fork(ctx, "deadbeef") is None


def test_an_all_digit_uuid_prefix_is_not_mistaken_for_an_index(ctx, tmp_path, session):
    # Hex ids are digits often enough that "looks like a number" cannot decide
    # between the two forms; only a number that indexes the list is an index.
    newest = session.list_user_messages()[0]["uuid"]
    digits = "18973649" + newest[8:]
    session.path.write_text(
        session.path.read_text(encoding="utf-8").replace(newest, digits), encoding="utf-8"
    )

    result = _fork(ctx, "18973649")

    contents = _forked_log(result, tmp_path).path.read_text(encoding="utf-8")
    assert "second question" not in contents


def test_the_original_session_is_left_intact(ctx, tmp_path, session):
    before = session.path.read_text(encoding="utf-8")

    _fork(ctx, "1")

    assert session.path.read_text(encoding="utf-8") == before


class TestUuidBefore:
    def test_returns_the_preceding_entry(self, session):
        entries = [
            line for line in session.path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        import json

        uuids = [json.loads(line)["uuid"] for line in entries]

        assert session.uuid_before(uuids[2]) == uuids[1]

    def test_returns_none_for_the_first_entry(self, session):
        first = session.list_user_messages()[-1]

        assert session.uuid_before(first["uuid"]) is None

    def test_returns_none_for_an_unknown_uuid(self, session):
        assert session.uuid_before("not-a-real-uuid") is None
