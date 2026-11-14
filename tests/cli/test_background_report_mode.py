# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: `background_report_mode: quiet` — every completion still shows, none wakes the agent.

The setting exists because the two halves of a background report have different
costs. Printing a completion block costs nothing and is the reason to watch a
long command at all. Handing it to the idle model buys a turn, and the user then
reads a summary of a report already on screen above it. ``quiet`` keeps the first
and defers the second to the next turn a person starts, so nothing is dropped.
"""
from agentica.cli.interactive.app import should_steer_report
from agentica.cli.interactive.session_state import SessionState


class TestSteerDecision:
    def test_an_idle_agent_is_not_woken_in_quiet_mode(self):
        assert should_steer_report(agent_running=False, quiet=True) is False

    def test_a_running_agent_is_steered_even_in_quiet_mode(self):
        """Steering costs no extra turn and lands mid-task, where it is useful."""
        assert should_steer_report(agent_running=True, quiet=True) is True

    def test_the_default_mode_still_hands_the_report_to_an_idle_agent(self):
        assert should_steer_report(agent_running=False, quiet=False) is True
        assert should_steer_report(agent_running=True, quiet=False) is True


class TestHeldReports:
    def test_a_fresh_session_holds_nothing(self):
        assert SessionState().pending_background_reports == []

    def test_held_reports_accumulate_in_arrival_order(self):
        """They ride the next turn in the order they happened, so a later
        completion never reads as if it came first."""
        state = SessionState()
        state.pending_background_reports.append("#1 finished")
        state.pending_background_reports.append("#2 failed")

        assert state.pending_background_reports == ["#1 finished", "#2 failed"]

    def test_held_reports_are_per_session(self):
        """Two sessions in one process must not leak reports into each other."""
        first, second = SessionState(), SessionState()
        first.pending_background_reports.append("#1 finished")

        assert second.pending_background_reports == []
