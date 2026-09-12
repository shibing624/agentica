# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: The non-interactive paths must never get a blocking sink.

``build_noninteractive_approve`` covers ``--print`` / SDK / cron / unattended
POST — runs with nobody at a terminal. Wiring "wait for a human to approve" into
one of those is a guaranteed hang: the answer would never come, and the run
would sit there. Its ``publish`` is a no-op and ``get_registry`` is ``None``
(so the manual path denies immediately).

These tests pin that contract from the outside, so a future edit cannot quietly
give those paths a blocking call. The sink itself is not to blame — what matters
is that nothing on these paths asks it anything.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agentica.notify import install_sink, reset_sink_for_tests
from agentica.notify.config import NotifyConfig

from tests.notify.test_notify_sink import _FakeDesktop


@pytest.fixture(autouse=True)
def _clean_process_sink():
    reset_sink_for_tests()
    yield
    reset_sink_for_tests()


class _StubAgent:
    """Just enough agent for build_noninteractive_approve."""

    class _ToolConfig:
        permission_mode = "ask"

    tool_config = _ToolConfig()
    work_dir = "/tmp/proj"
    user_id = "u1"

    def __init__(self):
        self._cancelled = False


class TestNonInteractiveNeverBlocks:
    def test_its_publish_is_a_no_op_even_with_the_sink_on(self):
        """The strongest statement: turning the sink on changes nothing here."""
        desktop = _FakeDesktop(decision_body={"decision": "allow"})
        try:
            install_sink(NotifyConfig(enabled=True, socket=desktop.socket_path,
                                      approve_from_desktop=True))
            from agentica.cli.approvals import build_noninteractive_approve

            approve = build_noninteractive_approve(_StubAgent())

            # A command that would park a human in `ask` mode: this is the case
            # where a blocking sink would hang a headless run forever.
            class _FC:
                call_id = "call_1"
                arguments = {"command": "rm -f /tmp/x"}
                function = type("F", (), {"name": "bash",
                                          "arguments": {"command": "rm -f /tmp/x"}})()
                approval_trace = None
                approval_waited = False

            started = time.monotonic()
            decision = asyncio.run(approve(_FC()))
            elapsed = time.monotonic() - started

            # get_registry() is None -> the manual path denies at once. No wait.
            assert decision == "deny"
            assert elapsed < 2.0, f"non-interactive approval took {elapsed:.1f}s"
            time.sleep(0.2)
            assert desktop.requests == [], "a headless run must not offer an approval"
        finally:
            desktop.close()

    def test_the_registry_is_none_on_that_path(self):
        """The documented invariant the sink logic keys off."""
        from agentica.agent.approvals import make_approve
        from agentica.cli.approvals import build_noninteractive_approve

        # build_noninteractive_approve passes get_registry=lambda: None; there is
        # no registry to decide into, which is exactly why nothing may block.
        import inspect
        src = inspect.getsource(build_noninteractive_approve)
        assert "get_registry=lambda: None" in src
        assert "publish=lambda pending: None" in src
