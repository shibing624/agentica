# -*- coding: utf-8 -*-
"""One gateway per data root serves IM.

A terminal gateway and the one a desktop shell spawned both connected the same
WeChat bot: each received every message and each ran it, so the user got every
"started" notice and every answer twice.
"""
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")


def _claim_in_child(cache_dir, hold_seconds=0.0):
    code = textwrap.dedent(f"""
        import os, sys, time
        os.environ["AGENTICA_CACHE_DIR"] = {str(cache_dir)!r}
        from agentica_gateway import runtime
        ok = runtime.claim_im_channels()
        print("claimed" if ok else "refused", flush=True)
        time.sleep({hold_seconds})
    """)
    env = {**os.environ, "AGENTICA_CACHE_DIR": str(cache_dir)}
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)


def test_a_second_gateway_does_not_connect_im_while_the_first_lives(tmp_path):
    first = _claim_in_child(tmp_path, hold_seconds=30)
    try:
        assert first.stdout.readline().strip() == "claimed"
        second = _claim_in_child(tmp_path)
        assert second.communicate(timeout=30)[0].strip() == "refused"
    finally:
        first.kill()
        first.wait()


def test_the_lock_is_free_again_once_its_owner_is_gone(tmp_path):
    """Released by the kernel on any exit, so a crash never blocks the next start."""
    first = _claim_in_child(tmp_path, hold_seconds=30)
    assert first.stdout.readline().strip() == "claimed"
    first.kill()
    first.wait()
    second = _claim_in_child(tmp_path)
    assert second.communicate(timeout=30)[0].strip() == "claimed"
