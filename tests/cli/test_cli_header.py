# -*- coding: utf-8 -*-
"""Startup banner includes the session id so it is copyable for /resume."""

from __future__ import annotations

import os

os.environ["OPENAI_API_KEY"] = "fake_openai_key_for_tests"

from agentica.cli.display.help_header import print_header


class _Con:
    width = 80

    def __init__(self):
        self.lines = []

    def print(self, *args, **kwargs):
        self.lines.append("" if not args else str(args[0]))


def test_header_shows_the_session_id(monkeypatch):
    con = _Con()
    monkeypatch.setattr("agentica.cli.display.help_header.get_console", lambda: con)

    print_header("openai", "gpt-5.6-sol", session_id="c1392649-f07d-4f05-808b-f852c3190236")

    out = "\n".join(con.lines)
    assert "Session:" in out
    assert "c1392649-f07d-4f05-808b-f852c3190236" in out
    assert "openai/gpt-5.6-sol" in out


def test_header_omits_session_when_none(monkeypatch):
    con = _Con()
    monkeypatch.setattr("agentica.cli.display.help_header.get_console", lambda: con)

    print_header("openai", "gpt-5.6-sol")

    assert "Session:" not in "\n".join(con.lines)
