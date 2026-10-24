# -*- coding: utf-8 -*-
"""Ctrl+D exits only on a second press inside the confirm window."""

import os

os.environ["OPENAI_API_KEY"] = "fake_openai_key_for_tests"

from agentica.cli.display.help_header import print_header
from agentica.cli.interactive.tui import (
    CTRL_D_EXIT_HINT,
    CTRL_D_EXIT_WINDOW_SECONDS,
    handle_ctrl_d,
)


class _State:
    def __init__(self):
        self.last_ctrl_d = 0.0
        self.should_exit = False


def _press(state, tui, now, exits):
    scheduled = []

    def schedule(delay, fn):
        scheduled.append((delay, fn))

    def exit_app():
        exits.append(now)

    handle_ctrl_d(state, tui, now=now, schedule=schedule, exit_app=exit_app)
    return scheduled


def test_second_press_at_the_window_edge_exits():
    state = _State()
    tui = {}
    exits = []
    scheduled = _press(state, tui, 10.0, exits)

    assert exits == []
    assert state.should_exit is False
    assert tui["exit_hint"] == CTRL_D_EXIT_HINT
    assert state.last_ctrl_d == 10.0
    assert [delay for delay, _fn in scheduled] == [CTRL_D_EXIT_WINDOW_SECONDS]

    edge = 10.0 + CTRL_D_EXIT_WINDOW_SECONDS
    _press(state, tui, edge, exits)

    assert exits == [edge]
    assert state.should_exit is True
    assert tui["exit_hint"] == ""
    assert state.last_ctrl_d == 0.0


def test_press_after_the_window_arms_again():
    state = _State()
    tui = {}
    exits = []
    _press(state, tui, 10.0, exits)
    later = 10.0 + CTRL_D_EXIT_WINDOW_SECONDS + 0.01
    scheduled = _press(state, tui, later, exits)

    assert exits == []
    assert state.should_exit is False
    assert tui["exit_hint"] == CTRL_D_EXIT_HINT
    assert state.last_ctrl_d == later
    assert len(scheduled) == 1


def test_expire_disarms_and_a_later_press_starts_over():
    state = _State()
    tui = {}
    exits = []
    scheduled = _press(state, tui, 10.0, exits)

    scheduled[0][1]()

    assert tui["exit_hint"] == ""
    assert state.last_ctrl_d == 0.0
    _press(state, tui, 11.0, exits)
    assert exits == []
    assert tui["exit_hint"] == CTRL_D_EXIT_HINT
    assert state.last_ctrl_d == 11.0


def test_stale_expire_does_not_clear_a_newer_arm():
    state = _State()
    tui = {}
    exits = []
    first = _press(state, tui, 10.0, exits)
    _press(state, tui, 11.0, exits)

    first[0][1]()

    assert exits == []
    assert tui["exit_hint"] == CTRL_D_EXIT_HINT
    assert state.last_ctrl_d == 11.0


def test_header_says_ctrl_d_needs_two_presses(monkeypatch):
    lines = []

    class _Con:
        width = 80

        def print(self, *args, **kwargs):
            lines.append("" if not args else str(args[0]))

    monkeypatch.setattr("agentica.cli.display.help_header.get_console", lambda: _Con())
    print_header("openai", "gpt-5.6-sol")
    assert any("Press twice to exit" in line for line in lines)
