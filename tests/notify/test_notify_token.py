# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for the notify token.

A missing or wrong token is not a crash — it is a channel that quietly stops
working, or worse, one that any local process can use. So the properties worth
pinning are: the file is 0600 from the moment it exists, an existing token is
never clobbered (that would 401 the other side out of its own secret), and a
token that cannot be prepared degrades to "no token" rather than an exception.
"""

from __future__ import annotations

import os
import stat
import tempfile

import pytest

from agentica.notify.config import NotifyConfig
from agentica.notify.token import TOKEN_BYTES, ensure_token


def _cfg(token_file: str, **kw) -> NotifyConfig:
    base = dict(enabled=True, token_file=token_file)
    base.update(kw)
    return NotifyConfig(**base)


class TestTokenCreation:
    def test_a_missing_token_file_is_created(self, tmp_path):
        path = str(tmp_path / "notify.token")
        token = ensure_token(_cfg(path))

        assert token and len(token) == TOKEN_BYTES * 2
        with open(path, encoding="utf-8") as fh:
            assert fh.read().strip() == token

    def test_the_file_is_0600_from_the_start(self, tmp_path):
        """Set at creation, so the secret never exists world-readable."""
        path = str(tmp_path / "notify.token")
        ensure_token(_cfg(path))

        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == 0o600, f"token file mode is {oct(mode)}, expected 0o600"

    def test_an_existing_token_is_reused_not_replaced(self, tmp_path):
        """The other side may have written it; clobbering would 401 us out."""
        path = tmp_path / "notify.token"
        path.write_text("the-desktop-apps-token\n", encoding="utf-8")

        assert ensure_token(_cfg(str(path))) == "the-desktop-apps-token"
        assert path.read_text(encoding="utf-8").strip() == "the-desktop-apps-token"

    def test_an_inline_token_wins_and_touches_no_file(self, tmp_path):
        path = tmp_path / "notify.token"
        assert ensure_token(_cfg(str(path), token="inline-token")) == "inline-token"
        assert not path.exists()

    def test_two_creates_produce_one_stable_token(self, tmp_path):
        path = str(tmp_path / "notify.token")
        first = ensure_token(_cfg(path))
        second = ensure_token(_cfg(path))
        assert first == second

    def test_a_new_token_each_time_it_is_really_absent(self, tmp_path):
        a = ensure_token(_cfg(str(tmp_path / "a.token")))
        b = ensure_token(_cfg(str(tmp_path / "b.token")))
        assert a != b

    def test_missing_parent_directories_are_created(self, tmp_path):
        path = str(tmp_path / "VPet" / "nested" / "notify.token")
        assert ensure_token(_cfg(path))
        assert os.path.exists(path)


class TestTokenDegradesSafely:
    def test_an_unwritable_location_yields_no_token_but_does_not_raise(self, tmp_path):
        """A channel that cannot authenticate is degraded, not fatal — the
        server's 401 already falls back to the terminal."""
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")

        # Parent path is a file, so the directory cannot be created.
        assert ensure_token(_cfg(str(blocker / "notify.token"))) is None

    def test_a_read_failure_falls_through_to_creating(self, tmp_path):
        path = tmp_path / "notify.token"
        path.write_text("   \n", encoding="utf-8")  # whitespace only

        # An empty/whitespace token is not a token: replace it rather than
        # sending an empty Authorization header forever.
        token = ensure_token(_cfg(str(path)))
        assert token and token.strip() == token and len(token) == TOKEN_BYTES * 2
