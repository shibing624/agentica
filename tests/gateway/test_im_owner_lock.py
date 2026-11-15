# -*- coding: utf-8 -*-
"""One gateway per data root serves IM: the lock decision, in-process.

The previous suite spawned two gateway child processes and waited for their
JSON lines. What matters here is the decision — a held lock blocks a second
claim, a released one allows it, and the owner file names a live process — and
that is testable without children.
"""
import os

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica_gateway import runtime


@pytest.fixture
def im_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "IM_LOCK_FILE", tmp_path / "im.lock")
    monkeypatch.setattr(runtime, "IM_OWNER_FILE", tmp_path / "im.owner")
    monkeypatch.setattr(runtime, "_im_lock_fd", None)
    yield
    fd = runtime._im_lock_fd
    runtime._im_lock_fd = None
    if fd is not None:
        os.close(fd)


def _hold_elsewhere(path):
    """A second open file description on the lock, as another process would have."""
    import fcntl

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_the_first_claim_wins_and_names_this_process(im_lock):
    assert runtime.claim_im_channels() is True
    assert runtime.IM_OWNER_FILE.read_text(encoding="utf-8").strip() == str(os.getpid())
    # Idempotent: the process that already holds it is not refused.
    assert runtime.claim_im_channels() is True


def test_a_second_claim_on_a_held_lock_is_refused(im_lock):
    holder = _hold_elsewhere(runtime.IM_LOCK_FILE)
    try:
        assert runtime.claim_im_channels() is False
        assert not runtime.IM_OWNER_FILE.exists()
    finally:
        os.close(holder)


def test_the_lock_is_free_again_once_the_holder_is_gone(im_lock):
    holder = _hold_elsewhere(runtime.IM_LOCK_FILE)
    os.close(holder)
    assert runtime.claim_im_channels() is True


def test_im_lock_holder_reports_a_live_owner_and_ignores_a_dead_one(im_lock):
    runtime.IM_OWNER_FILE.parent.mkdir(parents=True, exist_ok=True)
    runtime.IM_OWNER_FILE.write_text(str(os.getpid()), encoding="utf-8")
    assert runtime.im_lock_holder() == os.getpid()
    runtime.IM_OWNER_FILE.write_text("999999", encoding="utf-8")
    assert runtime.im_lock_holder() is None


def test_im_lock_holder_is_none_without_an_owner_file(im_lock):
    assert runtime.im_lock_holder() is None
