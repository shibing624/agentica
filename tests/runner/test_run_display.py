# -*- coding: utf-8 -*-
"""Tests for UI-agnostic RunResponse display classification."""

from agentica.run_response import RunEvent, RunResponse


def test_classifies_stream_content_delta():
    from agentica.run_display import RunDisplayEventKind, classify_run_response

    response = RunResponse(event=RunEvent.run_response.value, content="hello")

    assert classify_run_response(response).kind == RunDisplayEventKind.CONTENT_DELTA


def test_classifies_final_content_when_requested():
    from agentica.run_display import RunDisplayEventKind, classify_run_response

    response = RunResponse(event=RunEvent.run_response.value, content="final")

    assert classify_run_response(response, is_final=True).kind == RunDisplayEventKind.FINAL_CONTENT


def test_classifies_tool_lifecycle_events():
    from agentica.run_display import RunDisplayEventKind, classify_run_response

    started = RunResponse(event=RunEvent.tool_call_started.value, tools=[{"tool_name": "search"}])
    completed = RunResponse(event=RunEvent.tool_call_completed.value, tools=[{"tool_name": "search"}])

    assert classify_run_response(started).kind == RunDisplayEventKind.TOOL_STARTED
    assert classify_run_response(completed).kind == RunDisplayEventKind.TOOL_COMPLETED


def test_skips_metadata_events():
    from agentica.run_display import RunDisplayEventKind, classify_run_response

    response = RunResponse(event=RunEvent.run_started.value)

    assert classify_run_response(response).kind == RunDisplayEventKind.METADATA_SKIP


def test_classifies_failures_as_telemetry_only():
    from agentica.run_display import RunDisplayEventKind, classify_run_response

    response = RunResponse(event=RunEvent.run_failed.value, content="boom")

    assert classify_run_response(response).kind == RunDisplayEventKind.TELEMETRY_ONLY


def test_preserves_unknown_content_events_for_display():
    from agentica.run_display import RunDisplayEventKind, classify_run_response

    response = RunResponse(event="RunTimeout", content="Stream run timed out")

    assert classify_run_response(response).kind == RunDisplayEventKind.CONTENT_DELTA
