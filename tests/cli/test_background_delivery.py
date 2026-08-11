# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Handing a finished background command back to the agent.
"""
import time
from unittest.mock import MagicMock

from agentica.cli.commands.context import PendingQueue
from agentica.cli.interactive.btw import (
    _background_result_for_agent,
    hand_to_agent,
    promote_late_steer,
)
from agentica.cli.interactive.session_state import SessionState
from agentica.goals import CONTINUATION_PROMPT_PREFIX
from agentica.tools.background_processes import BackgroundProcessCompleted


def _event(returncode=0, command="pytest -q", log_path="", kind="command", label=""):
    now = time.time()
    return BackgroundProcessCompleted(
        id="term_3",
        num=3,
        pid=4242,
        command=command,
        cwd="/repo",
        log_path=log_path,
        started_at=now - 198,
        completed_at=now,
        returncode=returncode,
        kind=kind,
        label=label,
    )


class TestReportText:
    def test_a_success_report_carries_command_status_and_log(self):
        text = _background_result_for_agent(_event())

        assert "#3 (term_3) finished" in text
        assert "exit 0" in text
        assert "03:18" in text
        assert "Command: pytest -q" in text

    def test_a_failure_is_named_as_one(self):
        assert "failed" in _background_result_for_agent(_event(returncode=1))

    def test_the_full_command_is_never_truncated(self):
        command = "cd /very/long/path && " + " && ".join(f"step_{i}" for i in range(40))

        assert command in _background_result_for_agent(_event(command=command))

    def test_the_output_tail_is_included_when_the_log_has_one(self, tmp_path):
        log = tmp_path / "term_3.log"
        log.write_text("$ pytest -q\n\n30 passed\n", encoding="utf-8")

        text = _background_result_for_agent(_event(log_path=str(log)))

        assert "30 passed" in text
        # The `$ cmd` header is already stated above as `Command:`; repeating it
        # would spend context on the same string twice.
        assert text.count("pytest -q") == 1

    def test_a_missing_log_still_produces_a_report(self, tmp_path):
        text = _background_result_for_agent(_event(log_path=str(tmp_path / "gone.log")))

        assert "#3" in text
        assert "Output tail" not in text


class TestDelegatedTaskReport:
    """A delegated session hands back an answer, not a command log."""

    @staticmethod
    def _delegated(tmp_path, answer, returncode=0):
        log = tmp_path / "term_3.log"
        log.write_text(
            f"$ /usr/bin/python -m agentica.cli.main --query 'port the parser'\n\n{answer}\n",
            encoding="utf-8",
        )
        return _background_result_for_agent(
            _event(
                kind="delegate",
                label="parser port",
                log_path=str(log),
                returncode=returncode,
                command="/usr/bin/python -m agentica.cli.main --query 'port the parser'",
            )
        )

    def test_the_report_is_named_after_the_task_not_the_terminal(self, tmp_path):
        text = self._delegated(tmp_path, "Ported. The v1 shim is gone.")

        assert 'Delegated task "parser port" finished' in text
        assert "Background terminal" not in text

    def test_the_workers_answer_is_the_body_of_the_report(self, tmp_path):
        text = self._delegated(tmp_path, "Ported. The v1 shim is gone.")

        assert "Ported. The v1 shim is gone." in text
        # The command line that started it is a python -m invocation carrying
        # the whole task; repeating it would spend context on what the label
        # already says.
        assert "agentica.cli.main" not in text

    def test_a_long_answer_is_not_cut_to_a_command_tail(self, tmp_path):
        answer = "\n".join(f"line {i}" for i in range(100))

        text = self._delegated(tmp_path, answer)

        assert "line 0" in text
        assert "line 99" in text

    def test_a_failed_worker_says_so_and_does_not_invite_a_retry(self, tmp_path):
        text = self._delegated(tmp_path, "Traceback: no such file", returncode=1)

        assert 'Delegated task "parser port" failed' in text
        assert "Exit code: 1" in text
        assert "do not simply delegate the same task again" in text

    def test_a_silent_worker_still_produces_a_report(self, tmp_path):
        text = self._delegated(tmp_path, "")

        assert "no output" in text


class TestHandToAgent:
    def test_a_running_agent_is_steered_rather_than_queued(self):
        state = SessionState()
        state.agent_running = True
        state.current_agent = MagicMock()
        state.current_agent.steer.return_value = True
        pending = PendingQueue()

        hand_to_agent(state, pending, "report")

        # relayed=True marks provenance so a parked (undrained) report can be
        # re-queued with the __RELAYED__ tag instead of as plain typed input.
        state.current_agent.steer.assert_called_once_with("report", relayed=True)
        assert pending.empty()

    def test_an_idle_agent_gets_a_queued_turn(self):
        state = SessionState()
        state.agent_running = False
        state.current_agent = MagicMock()
        pending = PendingQueue()

        hand_to_agent(state, pending, "report")

        state.current_agent.steer.assert_not_called()
        # Tagged, so the turn is not echoed as if the user had typed the report.
        assert pending.peek_all() == [("__RELAYED__", "report")]

    def test_a_refused_steer_falls_back_to_the_queue(self):
        # steer() returns False when the run ended between the check and the
        # call; dropping the text there would lose a finished job's result.
        state = SessionState()
        state.agent_running = True
        state.current_agent = MagicMock()
        state.current_agent.steer.return_value = False
        pending = PendingQueue()

        hand_to_agent(state, pending, "report")

        assert pending.peek_all() == [("__RELAYED__", "report")]

    def test_text_is_kept_even_before_the_first_agent_exists(self):
        state = SessionState()
        state.agent_running = True
        state.current_agent = None
        pending = PendingQueue()

        hand_to_agent(state, pending, "report")

        assert pending.peek_all() == [("__RELAYED__", "report")]


class TestPromoteLateSteer:
    """Steering accepted during a run's final inference never reached the
    model; after the run it must become queued input, never vanish."""

    def test_late_steer_becomes_queued_input_in_order(self):
        state = SessionState()
        state.agent_running = False
        state.current_agent = MagicMock()
        state.current_agent.pop_undelivered_steer.return_value = [
            ("first", False),
            ("second", False),
        ]
        pending = PendingQueue()

        promoted = promote_late_steer(state, pending)

        assert promoted == ["first", "second"]
        # Plain str payloads: the user typed these, so the queued turn echoes
        # like any other typed input.
        assert pending.peek_all() == ["first", "second"]

    def test_parked_relayed_text_goes_back_tagged(self):
        # A peer/bg line accepted by steer() during the final inference was
        # never shown to the model. When it is promoted, the __RELAYED__ tag
        # must ride along — without it the queued turn regains the user-
        # message echo and slash-command dispatch, which the peer policy bans.
        state = SessionState()
        state.current_agent = MagicMock()
        state.current_agent.pop_undelivered_steer.return_value = [
            ("#3 (term_2) finished: exit=0", True),
        ]
        pending = PendingQueue()

        promote_late_steer(state, pending)

        assert pending.peek_all() == [("__RELAYED__", "#3 (term_2) finished: exit=0")]

    def test_promotion_jumps_ahead_of_a_goal_continuation(self):
        state = SessionState()
        state.current_agent = MagicMock()
        state.current_agent.pop_undelivered_steer.return_value = [("user correction", False)]
        pending = PendingQueue()
        pending.put(f"{CONTINUATION_PROMPT_PREFIX} keep going")

        promote_late_steer(state, pending)

        assert pending.peek_all()[0] == "user correction"

    def test_nothing_parked_is_a_noop(self):
        state = SessionState()
        state.current_agent = MagicMock()
        state.current_agent.pop_undelivered_steer.return_value = []
        pending = PendingQueue()

        assert promote_late_steer(state, pending) == []
        assert pending.empty()

    def test_no_agent_yet_is_a_noop(self):
        state = SessionState()
        state.current_agent = None
        pending = PendingQueue()

        assert promote_late_steer(state, pending) == []
        assert pending.empty()
