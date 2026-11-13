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


def test_a_claim_older_than_the_ttl_is_stealable(tmp_path):
    path = write_record(tmp_path)
    claim = path.with_suffix(".claim")
    claim.write_text(
        json.dumps({"owner": "slow", "pid": os.getpid(), "at": time.time() - 10_000}),
        encoding="utf-8",
    )
    assert claim_job_delivery(path, "owner", ttl=1.0) is True


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
