# -*- coding: utf-8 -*-
"""Dispatch identity (``agentica.ticket``) and the four places it lands.

The rules these tests pin down, from the module docstring:

1. A ticket is minted once per dispatch and *inherited* by everything that
   dispatch starts — never re-minted by a child.
2. It is a name, not a state machine: binding it changes no behaviour, and a
   reader that finds no ticket must report "unmanaged" rather than invent one.
3. It survives a process boundary through the environment, which is the only
   channel ``delegate`` has.
"""
from __future__ import annotations

import json
import os
import types

import pytest

from agentica.memory.session_log import SessionLog
from agentica.tools.background_processes import BackgroundProcessRegistry
from agentica.ticket import (
    TICKET_ENV,
    bind_ticket,
    enter_ticket,
    current_ticket_id,
    new_ticket_id,
    reset_ticket_id,
    set_ticket_id,
    ticket_scope,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Both inheritance channels are process-wide; keep each test's explicit.

    The contextvar in particular outlives a test — it is bound for a whole
    dispatch in production and reset by the runner — so a test that left it set
    would silently make the next one inherit a ticket it never asked for.
    """
    monkeypatch.delenv(TICKET_ENV, raising=False)
    set_ticket_id(None)


def test_a_fresh_ticket_has_the_documented_shape():
    ticket = new_ticket_id()
    assert ticket.startswith("tk_")
    assert len(ticket) == len("tk_") + 12
    assert new_ticket_id() != new_ticket_id()


def test_enter_mints_then_inherits_and_reset_clears_it():
    ticket, token = enter_ticket()
    assert ticket
    # A nested dispatch (a subagent, a task) inherits rather than minting.
    assert bind_ticket() == ticket

    reset_ticket_id(token)
    assert current_ticket_id() is None


def test_enter_ticket_restores_the_previous_value_not_its_own():
    """``set_ticket_id(bind_ticket(...))`` would restore the ticket it just set."""
    outer, outer_token = enter_ticket()
    # A new dispatch names itself; a bare ``enter_ticket()`` would inherit.
    inner, inner_token = enter_ticket(new_ticket_id())
    assert inner != outer
    reset_ticket_id(inner_token)
    assert current_ticket_id() == outer
    reset_ticket_id(outer_token)
    assert current_ticket_id() is None


def test_bind_prefers_an_explicit_id_over_the_inherited_one():
    inherited = bind_ticket()
    explicit = new_ticket_id()
    assert bind_ticket(explicit) == explicit
    assert explicit != inherited


def test_the_environment_is_the_inheritance_channel(monkeypatch):
    """A delegated child reads its parent's ticket from its environment."""
    monkeypatch.setenv(TICKET_ENV, "tk_fromparent")
    assert current_ticket_id() == "tk_fromparent"
    assert bind_ticket() == "tk_fromparent"


def test_bind_does_not_export_the_environment():
    """Writing TICKET_ENV here would outlive the turn and poison every later one."""
    bind_ticket()
    assert TICKET_ENV not in os.environ


def test_a_malformed_id_is_treated_as_unmanaged():
    assert bind_ticket("   ") != "   "
    assert current_ticket_id() != ""


def test_ticket_scope_restores_the_previous_value():
    outer = bind_ticket()
    with ticket_scope(new_ticket_id()) as inner:
        assert inner and inner != outer
        assert current_ticket_id() == inner
    assert current_ticket_id() == outer


def test_session_sidecar_records_only_the_first_ticket(tmp_path):
    log = SessionLog("s1", base_dir=str(tmp_path), work_dir=str(tmp_path))
    log.append("user", "hello")
    assert log.get_ticket_id() is None

    log.set_ticket_id("tk_first")
    assert log.get_ticket_id() == "tk_first"
    # A later turn of the same conversation continues the same dispatch, so a
    # second value would only make the sidecar disagree with its records.
    log.set_ticket_id("tk_second")
    assert log.get_ticket_id() == "tk_first"


def test_an_empty_ticket_never_writes_the_sidecar(tmp_path):
    log = SessionLog("s1", base_dir=str(tmp_path), work_dir=str(tmp_path))
    log.set_ticket_id(None)
    log.set_ticket_id("")
    assert not log.meta_path.exists()
    assert log.get_ticket_id() is None


def test_session_rows_expose_the_ticket(tmp_path):
    log = SessionLog("s1", base_dir=str(tmp_path), work_dir=str(tmp_path))
    log.append("user", "hello")
    log.set_ticket_id("tk_row")
    rows = SessionLog.list_sessions(base_dir=str(tmp_path), work_dir=str(tmp_path))
    assert [r["ticket_id"] for r in rows] == ["tk_row"]


def test_a_record_written_before_tickets_reads_as_unmanaged(tmp_path):
    """No ticket on disk must never become a made-up one."""
    log = SessionLog("s2", base_dir=str(tmp_path), work_dir=str(tmp_path))
    log.append("user", "hello")
    log.meta_path.write_text(json.dumps({"name": "old session"}), encoding="utf-8")
    assert log.get_ticket_id() is None


def test_cron_job_mints_a_ticket_and_round_trips_it(tmp_path, monkeypatch):
    from agentica import cron

    monkeypatch.setattr(cron.jobs, "JOBS_FILE", tmp_path / "jobs.json")
    job = cron.jobs.create_job(prompt="say hi", schedule="every 60s", name="hi")
    assert job.ticket_id.startswith("tk_")

    restored = cron.jobs.CronJob.from_dict(job.to_dict())
    assert restored.ticket_id == job.ticket_id

    legacy = job.to_dict()
    legacy.pop("ticket_id")
    assert cron.jobs.CronJob.from_dict(legacy).ticket_id == ""


def test_the_command_context_reads_the_shared_job_view(tmp_path):
    """``/ps`` must see a job this process never started.

    This is the regression: the registry's in-memory list is empty for a job
    started by the gateway, while the record on disk is complete.
    """
    from agentica.cli.commands.runtime import _session_job_views

    records = tmp_path / "background"
    records.mkdir()
    (records / "term_1.json").write_text(
        json.dumps(
            {
                "id": "term_1",
                "num": 1,
                "pid": os.getpid(),
                "owner_pid": os.getpid() - 1,
                "session_id": "s1",
                "ticket_id": "tk_elsewhere",
                "kind": "command",
                "label": "",
                "command": "echo elsewhere",
                "cwd": str(tmp_path),
                "log_path": str(records / "term_1.log"),
                "started_at": 1_700_000_000.0,
                "finished_at": None,
                "returncode": None,
                "stop_requested": False,
            }
        ),
        encoding="utf-8",
    )

    registry = BackgroundProcessRegistry(
        session_id="s1", records_dir=records,
    )
    ctx = types.SimpleNamespace(background_processes=registry)

    # The old read path saw nothing: the job was never started in this process.
    assert registry.list(include_finished=True) == []
    views = _session_job_views(ctx)
    assert [v["id"] for v in views] == ["term_1"]
    assert views[0]["ticket_id"] == "tk_elsewhere"
