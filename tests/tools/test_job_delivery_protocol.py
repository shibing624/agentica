# -*- coding: utf-8 -*-
"""Durable delivery of a background job's report.

The report used to be an event in the memory of the process that started the
job. A job that outlived that process ended silently. These tests pin the
replacement contract:

* a finished record with no ``delivered_at`` is owed, whoever owns it;
* the claim makes sure only one process pays it;
* the marker makes a second pass a no-op;
* a job that was stopped is settled, not left permanently "pending".
"""
from __future__ import annotations

import json
import os
import sys
import time

from agentica.ticket import ticket_scope
from agentica.tools.background_processes import (
    BackgroundProcessRegistry,
    claim_job_delivery,
    deliver_undelivered_jobs,
    delivery_owner,
    load_session_job_views,
    mark_job_delivered,
    record_is_settled,
    stop_session_job,
    undelivered_job_records,
)

DEAD_PID = 999_999


def write_record(
    records_dir,
    job_id="term_1",
    *,
    num=1,
    finished=True,
    returncode=0,
    stop_requested=False,
    delivered_at=None,
    ticket_id="",
    age=1.0,
    command="echo hello",
    log_text="hello\n",
):
    """A record exactly as ``BackgroundProcessRegistry`` writes one."""
    records_dir.mkdir(parents=True, exist_ok=True)
    log_path = records_dir / f"{job_id}.log"
    log_path.write_text(f"$ {command}\n\n{log_text}", encoding="utf-8")
    now = time.time()
    record = {
        "id": job_id,
        "num": num,
        "pid": DEAD_PID,
        "owner_pid": DEAD_PID,
        "session_id": "s1",
        "ticket_id": ticket_id,
        "delivered_at": delivered_at,
        "delivery_owner": None,
        "delivery_attempts": 0,
        "kind": "command",
        "label": "",
        "command": command,
        "cwd": str(records_dir),
        "log_path": str(log_path),
        "started_at": now - age - 2,
        "finished_at": (now - age) if finished else None,
        "returncode": returncode if finished else None,
        "stop_requested": stop_requested,
    }
    path = records_dir / f"{job_id}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def read_record(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# The claim
# --------------------------------------------------------------------------

def test_only_one_claimer_wins(tmp_path):
    path = write_record(tmp_path)
    assert claim_job_delivery(path, "owner-a") is True
    assert claim_job_delivery(path, "owner-b") is False


def test_a_claim_from_a_dead_process_is_stealable(tmp_path):
    path = write_record(tmp_path)
    claim = path.with_suffix(".claim")
    claim.write_text(json.dumps({"owner": "gone", "pid": DEAD_PID, "at": time.time()}), encoding="utf-8")
    assert claim_job_delivery(path, "owner") is True


def test_a_claim_whose_owner_is_alive_is_never_stolen(tmp_path):
    """Age must not decide this.

    A delivery legitimately waits on a terminal that is mid-turn, which is
    minutes, so an age-based expiry would hand the same record to a second
    process and deliver the report twice. The owner being alive is the signal.
    """
    path = write_record(tmp_path)
    claim = path.with_suffix(".claim")
    claim.write_text(
        json.dumps({"owner": "slow", "pid": os.getpid(), "at": time.time() - 10_000}),
        encoding="utf-8",
    )
    assert claim_job_delivery(path, "owner", ttl=1.0) is False


def test_a_claim_that_was_never_written_is_not_stolen_before_the_ttl(tmp_path):
    """The O_CREAT..os.write window must not read as "abandoned"."""
    path = write_record(tmp_path)
    path.with_suffix(".claim").write_text("", encoding="utf-8")
    assert claim_job_delivery(path, "owner") is False


def test_a_half_written_claim_is_left_alone_until_it_ages_out(tmp_path):
    path = write_record(tmp_path)
    claim = path.with_suffix(".claim")
    claim.write_text('{"owner": "tor', encoding="utf-8")
    assert claim_job_delivery(path, "owner") is False


def test_the_claim_file_is_not_read_as_a_job_record(tmp_path):
    """``*.json`` is the job-record glob; a claim must never join it."""
    path = write_record(tmp_path)
    claim_job_delivery(path, "owner")
    assert [p.name for p in tmp_path.glob("*.json")] == ["term_1.json"]
    assert [v["id"] for v in load_session_job_views(tmp_path)] == ["term_1"]


# --------------------------------------------------------------------------
# Delivery
# --------------------------------------------------------------------------

def test_a_finished_undelivered_job_is_delivered_once_and_marked(tmp_path):
    path = write_record(tmp_path)
    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 1
    assert len(seen) == 1
    assert "echo hello" in seen[0]
    assert read_record(path)["delivered_at"]
    assert read_record(path)["delivery_owner"]

    # Second pass: nothing owed.
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 0
    assert len(seen) == 1


def test_a_record_marked_delivered_is_not_owed(tmp_path):
    path = write_record(tmp_path)
    mark_job_delivered(path, "someone")
    assert undelivered_job_records(tmp_path) == []
    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 0
    assert seen == []


def test_a_stopped_job_is_settled_without_a_report(tmp_path):
    """"Pending" means someone still owes this. Nobody owes a stopped job."""
    path = write_record(tmp_path, stop_requested=True)
    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 0
    assert seen == []
    assert read_record(path)["delivered_at"], "it must stop being counted as owed"


def test_a_running_job_is_not_owed_yet(tmp_path):
    write_record(tmp_path, finished=False)
    assert undelivered_job_records(tmp_path) == []


def test_a_report_older_than_the_horizon_is_not_replayed(tmp_path):
    write_record(tmp_path, age=48 * 3600)
    assert undelivered_job_records(tmp_path) == []


def test_the_report_survives_the_process_that_started_the_job(tmp_path):
    """The whole point: nobody from the original process is here to deliver it."""
    path = write_record(tmp_path, ticket_id="tk_dead")
    registry = BackgroundProcessRegistry(session_id="s1", records_dir=tmp_path)
    assert registry.list(include_finished=True) == []

    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 1
    assert "echo hello" in seen[0]
    assert read_record(path)["delivered_at"]


def test_one_poisoned_record_does_not_block_the_next(tmp_path):
    write_record(tmp_path, "term_1")
    write_record(tmp_path, "term_2", num=2, command="echo second")

    delivered = []

    def handler(rec, report):
        if rec["id"] == "term_1":
            raise RuntimeError("boom")
        delivered.append(rec["id"])

    deliver_undelivered_jobs(tmp_path, handler)
    assert delivered == ["term_2"]
    # The failed one is retried rather than lost, and its counter records why.
    assert read_record(tmp_path / "term_1.json")["delivered_at"] is None
    assert read_record(tmp_path / "term_1.json")["delivery_attempts"] == 1


def test_a_record_that_keeps_failing_stops_being_retried(tmp_path):
    path = write_record(tmp_path)
    for _ in range(3):
        deliver_undelivered_jobs(tmp_path, lambda rec, report: (_ for _ in ()).throw(RuntimeError("x")))
    assert read_record(path)["delivery_attempts"] == 3
    assert undelivered_job_records(tmp_path) == []


def test_the_sweep_delivers_at_most_a_handful_per_pass(tmp_path):
    for n in range(1, 8):
        write_record(tmp_path, f"term_{n}", num=n)
    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(rec["id"]), max_reports=5) == 5
    assert len(seen) == 5
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(rec["id"]), max_reports=5) == 2


# --------------------------------------------------------------------------
# Ticket stamping
# --------------------------------------------------------------------------

def test_starting_a_job_stamps_the_current_dispatch(tmp_path):
    with ticket_scope("tk_turn"):
        registry = BackgroundProcessRegistry(session_id="s1", records_dir=tmp_path)
        item = registry.start(f"{sys.executable} -c 'pass'", cwd=str(tmp_path))
        assert item.finished.wait(15)

    record = read_record(tmp_path / "term_1.json")
    assert record["ticket_id"] == "tk_turn"
    views = load_session_job_views(tmp_path, registry)
    assert views[0]["ticket_id"] == "tk_turn"


def test_a_job_started_outside_any_dispatch_is_unmanaged(tmp_path):
    registry = BackgroundProcessRegistry(session_id="s1", records_dir=tmp_path)
    item = registry.start(f"{sys.executable} -c 'pass'", cwd=str(tmp_path))
    assert item.finished.wait(15)
    assert read_record(tmp_path / "term_1.json")["ticket_id"] == ""


# --------------------------------------------------------------------------
# Cross-process stop
# --------------------------------------------------------------------------

def test_a_job_this_process_never_started_can_be_stopped_by_id(tmp_path):
    path = write_record(tmp_path, finished=False)
    rec = read_record(path)
    rec["pid"] = DEAD_PID
    path.write_text(json.dumps(rec), encoding="utf-8")

    view = stop_session_job(tmp_path, "term_1", None, user_id="default")
    assert view is not None
    assert view["status"] == "stopped"
    assert read_record(path)["stop_requested"] is True


def test_the_same_owner_string_is_stable_enough_to_compare(tmp_path):
    assert delivery_owner("cli").endswith(f"pid:{os.getpid()}")
    assert delivery_owner("cli") != delivery_owner("gateway")


# --------------------------------------------------------------------------
# The two failure modes the sweep cannot see
# --------------------------------------------------------------------------

def test_record_is_settled_requires_the_completion_write(tmp_path):
    """A record that exists but never recorded its end can't be adopted.

    The start-time record is written too, and the completion write is
    best-effort, so "the file is there" is not "the report is owed here".
    """
    assert record_is_settled(tmp_path, "term_1") is False
    write_record(tmp_path, finished=False)
    assert record_is_settled(tmp_path, "term_1") is False
    write_record(tmp_path, finished=True)
    assert record_is_settled(tmp_path, "term_1") is True
    assert record_is_settled(None, "term_1") is False


def test_a_marker_that_cannot_be_written_keeps_the_claim(tmp_path, monkeypatch):
    """An unverified marker would re-deliver on the next poll.

    ``_write_json_atomic`` swallows OSError, so the marker is read back. When it
    did not land the claim stays, which spaces the retry by the TTL instead of
    by the poll interval.
    """
    import agentica.tools.background_processes as bp

    path = write_record(tmp_path)
    assert claim_job_delivery(path, "owner") is True
    monkeypatch.setattr(bp, "_write_json_atomic", lambda *a, **k: None)
    assert mark_job_delivered(path, "owner") is False
    assert path.with_suffix(".claim").exists()
    assert read_record(path)["delivered_at"] is None


def test_a_record_from_before_delivery_state_is_never_replayed(tmp_path):
    """Upgrading must not re-deliver every report the old path already handled.

    A pre-upgrade record has no ``delivered_at`` key at all. Its report was
    delivered (or lost) by the in-memory path and there is no way to tell which,
    so it is treated as settled — the same rule as a missing ticket.
    """
    path = write_record(tmp_path)
    raw = read_record(path)
    for key in ("delivered_at", "delivery_owner", "delivery_attempts"):
        raw.pop(key, None)
    path.write_text(json.dumps(raw), encoding="utf-8")

    assert undelivered_job_records(tmp_path) == []
    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 0
    assert seen == []


def test_a_job_whose_owner_died_still_gets_a_report(tmp_path):
    """The job outlived the process that started it; nobody wrote an exit code.

    Silence is the bug this whole module exists to remove, so an orphan is
    reported as "ended while nobody was watching" rather than quietly settled.
    """
    path = write_record(tmp_path)
    raw = read_record(path)
    raw["returncode"] = None  # the watcher died before it could record one
    path.write_text(json.dumps(raw), encoding="utf-8")

    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 1
    assert "ended while nobody was watching" in seen[0]
    assert "echo hello" in seen[0]
    assert "hello" in seen[0], "the log tail is the only evidence left"
    assert read_record(path)["delivered_at"]


def test_a_stopped_orphan_still_has_nothing_to_say(tmp_path):
    write_record(tmp_path, stop_requested=True, returncode=None)
    seen = []
    assert deliver_undelivered_jobs(tmp_path, lambda rec, report: seen.append(report)) == 0
    assert seen == []
