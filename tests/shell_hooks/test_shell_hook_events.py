# -*- coding: utf-8 -*-
"""Session and request-resolution notices on the shell-hook wire."""

from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import pytest

from agentica.shell_hooks.config import HookConsumer, ShellHooksConfig
from agentica.shell_hooks.egress import install_hook_egress, reset_hook_egress_for_tests
from agentica.shell_hooks.events import (
    emit_request_resolved,
    emit_session_ended,
    emit_session_started,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_hook_egress_for_tests()
    yield
    reset_hook_egress_for_tests()


def _wait_for(path, count):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists():
            docs = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(docs) >= count:
                return docs
        time.sleep(0.02)
    raise AssertionError(f"{path} never reached {count} events")


def test_session_and_resolved_events_carry_identity(tmp_path):
    output = tmp_path / "events.jsonl"
    script = tmp_path / "hook.py"
    script.write_text(
        "import json,sys\n"
        f"open({str(output)!r},'a').write(json.dumps(json.load(sys.stdin))+'\\n')\n",
        encoding="utf-8",
    )
    install_hook_egress(
        ShellHooksConfig(
            enabled=True,
            consumers=[
                HookConsumer(name="desktop", command=[sys.executable, str(script)])
            ],
        )
    )
    transcript = tmp_path / "session.jsonl"
    agent = SimpleNamespace(
        session_id="session-1",
        work_dir=str(tmp_path),
        run_context=None,
        model=SimpleNamespace(id="model-1"),
        tool_config=SimpleNamespace(permission_mode="ask"),
        session_log=SimpleNamespace(path=transcript),
    )

    emit_session_started(agent, source="startup", profile="main")
    emit_request_resolved(
        agent,
        request_id="request-1",
        event="needs.approval",
        decided_by="terminal",
        decision="allow",
    )
    emit_session_ended(agent, reason="exit")

    docs = _wait_for(output, 3)
    by_name = {doc["hook_event_name"]: doc for doc in docs}
    started = by_name["session.started"]
    assert started["session_id"] == "session-1"
    assert started["model"] == "model-1"
    assert started["profile"] == "main"
    assert started["permission_mode"] == "ask"
    assert started["transcript_path"] == str(transcript)
    assert by_name["needs.resolved"]["request_id"] == "request-1"
    assert by_name["needs.resolved"]["decided_by"] == "terminal"
    assert by_name["session.ended"]["reason"] == "exit"
