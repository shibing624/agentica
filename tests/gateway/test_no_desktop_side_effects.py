# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests must never open a window on the developer's desktop.

Every authorised ``POST /api/open`` used to end in a real ``open -R /tmp``, so
each test run popped Finder. conftest's ``opened_externally`` now replaces the
two exits (``settings._open_os`` and ``webbrowser.open``) for every test; this
file pins that in place.
"""
import webbrowser

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica_gateway.routes import settings as settings_routes  # noqa: E402


def test_the_desktop_opener_is_replaced_in_every_test(opened_externally):
    settings_routes._open_os("/tmp", "reveal")
    webbrowser.open("file:///tmp/qr.png")
    assert opened_externally == [("/tmp", "reveal"), ("file:///tmp/qr.png", "browser")]


def test_the_real_opener_only_spawns_what_open_command_returns(monkeypatch):
    """The real ``_open_os`` (the one conftest swaps out) is just Popen of the
    pure command, so testing ``_open_command`` covers what it would run."""
    real = settings_routes._open_os.real
    spawned = []
    monkeypatch.setattr(settings_routes.subprocess, "Popen", lambda argv, **kw: spawned.append(argv))
    monkeypatch.setattr(settings_routes, "_open_command", lambda target, app, platform=None: ["opener", app, target])
    real("/x", "reveal")
    monkeypatch.setattr(settings_routes, "_open_command", lambda target, app, platform=None: None)
    real("/x", "terminal")
    assert spawned == [["opener", "reveal", "/x"]]
