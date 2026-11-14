# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: One report shape for N finished jobs, and the quiet delivery mode.

Two things the completion report has to get right, and they pull against each
other: the user must still see every finished terminal, and the model must not
be woken once per terminal to be told what the user already read. So printing
stays per job (CLI) while the hand-over to the agent is one combined report, and
``background_report_mode: quiet`` can keep the hand-over for the next turn
instead of opening one.
"""
import time

from agentica.tools.background_processes import (
    BackgroundProcessCompleted,
    render_batch_report,
)


def _ev(num=1, returncode=0, command="pytest -q", log_path=""):
    now = time.time()
    return BackgroundProcessCompleted(
        id=f"term_{num}", num=num, pid=4000 + num, command=command, cwd="/repo",
        log_path=log_path, started_at=now - 120, completed_at=now,
        returncode=returncode, kind="command", label="", stop_requested=False,
    )


class TestSingleReport:
    def test_it_no_longer_asks_the_model_to_respond(self):
        """The old text ended in "report the outcome to the user", which made
        every finished job buy a model turn the user had not asked for."""
        text = render_batch_report([_ev()])

        assert "report the outcome" not in text
        assert "No reply is needed" in text

    def test_it_still_names_the_job_command_and_log(self):
        text = render_batch_report([_ev(returncode=1, log_path="/tmp/a.log")])

        assert "#1 (term_1) failed" in text
        assert "exit 1" in text
        assert "Command: pytest -q" in text
        assert "/tmp/a.log" in text


class TestBatchReport:
    def test_several_jobs_become_one_report_naming_each_of_them(self):
        text = render_batch_report([_ev(1), _ev(2, returncode=1), _ev(3, returncode=-15)])

        assert text.startswith("[3 background terminals finished]")
        for num in (1, 2, 3):
            assert f"#{num} (term_{num})" in text
        assert "1 clean, 2 not" in text

    def test_each_settled_job_keeps_its_own_status_word(self):
        text = render_batch_report([_ev(1, returncode=0), _ev(2, returncode=-15)])

        assert "#1 (term_1) finished: exit 0" in text
        assert "#2 (term_2) stopped: SIGTERM" in text

    def test_a_batch_does_not_ask_for_a_reply_either(self):
        assert "report the outcome" not in render_batch_report([_ev(1), _ev(2)])

    def test_one_job_delegates_to_the_single_report_shape(self):
        """A pass that collected one report must read exactly like an event that
        never went through the sweep, so the model sees one format."""
        single = render_batch_report([_ev()])

        assert single.startswith("[Background terminal #1 (term_1) finished")

    def test_every_job_carries_its_log_path(self):
        text = render_batch_report([_ev(1, log_path="/tmp/a.log"), _ev(2, log_path="/tmp/b.log")])

        assert "/tmp/a.log" in text
        assert "/tmp/b.log" in text
