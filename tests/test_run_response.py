# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Tests for RunResponse and RunEvent models.
"""
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agentica.run_response import RunResponse, RunEvent, RunResponseExtraData
from agentica.model.message import Message


# ===========================================================================
# TestRunResponse
# ===========================================================================


class TestRunResponse:
    """Tests for RunResponse data model."""

    def test_creation_with_defaults(self):
        resp = RunResponse()
        assert resp.content is None
        assert resp.content_type == "str"
        assert resp.event == RunEvent.run_response.value

    def test_creation_with_content(self):
        resp = RunResponse(content="Hello world")
        assert resp.content == "Hello world"

    def test_run_id_field(self):
        resp = RunResponse(run_id="run-123")
        assert resp.run_id == "run-123"

    def test_session_id_field(self):
        resp = RunResponse(session_id="sess-456")
        assert resp.session_id == "sess-456"

    def test_agent_id_field(self):
        resp = RunResponse(agent_id="agent-789")
        assert resp.agent_id == "agent-789"

    def test_with_tools(self):
        tools = [{"tool_name": "calc", "content": "42"}]
        resp = RunResponse(content="OK", tools=tools)
        assert resp.tools == tools

    def test_with_reasoning_content(self):
        resp = RunResponse(content="Answer", reasoning_content="Thinking...")
        assert resp.reasoning_content == "Thinking..."

    def test_with_extra_data(self):
        extra = RunResponseExtraData(references=None)
        resp = RunResponse(content="OK", extra_data=extra)
        assert resp.extra_data is not None

    def test_to_json(self):
        resp = RunResponse(content="Hello", run_id="r1")
        json_str = resp.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["content"] == "Hello"

    def test_created_at_auto_set(self):
        resp = RunResponse()
        assert resp.created_at > 0

    def test_model_field(self):
        resp = RunResponse(model="gpt-4o")
        assert resp.model == "gpt-4o"


# ===========================================================================
# TestRunEvent
# ===========================================================================


class TestRunEvent:
    """Tests for RunEvent enum."""

    def test_all_expected_events_defined(self):
        expected = [
            "run_started", "run_response", "run_completed",
            "tool_call_started", "tool_call_completed",
            "reasoning_started", "reasoning_step", "reasoning_completed",
            "updating_memory",
            "workflow_started", "workflow_completed",
        ]
        for event_name in expected:
            assert hasattr(RunEvent, event_name), f"Missing event: {event_name}"

    def test_event_values_are_strings(self):
        for event in RunEvent:
            assert isinstance(event.value, str)

    def test_event_names_unique(self):
        values = [e.value for e in RunEvent]
        assert len(values) == len(set(values)), "Duplicate event values found"

    def test_tool_call_events_present(self):
        assert RunEvent.tool_call_started.value == "ToolCallStarted"
        assert RunEvent.tool_call_completed.value == "ToolCallCompleted"

    def test_reasoning_events_present(self):
        assert RunEvent.reasoning_started.value == "ReasoningStarted"
        assert RunEvent.reasoning_step.value == "ReasoningStep"
        assert RunEvent.reasoning_completed.value == "ReasoningCompleted"

    def test_workflow_events_present(self):
        assert RunEvent.workflow_started.value == "WorkflowStarted"
        assert RunEvent.workflow_completed.value == "WorkflowCompleted"

    def test_multi_round_events_present(self):
        assert hasattr(RunEvent, "multi_round_turn")
        assert hasattr(RunEvent, "multi_round_completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
