# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Tests for the SDK-first run lifecycle types (arch_v5.md Phase 0/3).

Covers:
- TaskAnchor.from_message handles str/dict/Message-like inputs
- TaskAnchor.to_prompt_block renders empty for empty anchors
- RunContext lifecycle transitions (running/completed/failed/cancelled)
- RunEventRecord.to_dict shape
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentica.run_context import RunContext, RunSource, RunStatus, TaskAnchor
from agentica.run_events import RunEventRecord, RunEventType


class TestTaskAnchor:
    def test_from_str_message(self):
        a = TaskAnchor.from_message("Build me a thing")
        assert a.goal == "Build me a thing"
        assert a.source_query == "Build me a thing"

    def test_from_dict_message(self):
        a = TaskAnchor.from_message({"role": "user", "content": "hello"})
        assert a.goal == "hello"
        assert a.source_query == "hello"

    def test_from_object_with_content(self):
        class _Msg:
            content = "fix bug X"
        a = TaskAnchor.from_message(_Msg())
        assert a.goal == "fix bug X"

    def test_from_unsupported_message_is_empty(self):
        a = TaskAnchor.from_message(None)
        assert a.goal == ""
        assert a.source_query == ""

    def test_to_prompt_block_empty(self):
        assert TaskAnchor().to_prompt_block() == ""

    def test_default_source_is_message(self):
        """from_message and bare constructor default to source='message'."""
        assert TaskAnchor().source == "message"
        assert TaskAnchor.from_message("hi").source == "message"

    def test_to_prompt_block_renders_goal_and_constraints(self):
        a = TaskAnchor(
            goal="Refactor X",
            source_query="Refactor X",
            acceptance_criteria=["pass tests", "no breakage"],
            constraints=["don't touch DB"],
            confirmed_facts=["repo=foo"],
            next_step_hint="start with module Y",
            source="goal",
        )
        block = a.to_prompt_block()
        assert "<original_task>" in block and "</original_task>" in block
        assert "GOAL: Refactor X" in block
        assert "ACCEPTANCE CRITERIA:" in block
        assert "- pass tests" in block
        assert "CONSTRAINTS:" in block
        assert "- don't touch DB" in block
        assert "NEXT STEP HINT: start with module Y" in block

    def test_message_sourced_anchor_never_renders(self):
        """Even multi-line message-sourced anchors stay out of the prompt.

        Guards the private-chat seed / resume / workflow-handoff bug where a
        transcript fed as `agent.run(msg)`'s first message used to leak into
        every turn's system prompt.
        """
        transcript = "User: hi\nAssistant: hi back\nUser: continue please"
        a = TaskAnchor.from_message(transcript)
        assert a.source == "message"
        assert a.to_prompt_block() == ""

    def test_message_sourced_with_structured_fields_still_does_not_render(self):
        """source gate is authoritative — structured fields don't bypass it."""
        a = TaskAnchor(
            goal="x",
            source_query="x",
            acceptance_criteria=["a"],
            constraints=["b"],
            source="message",
        )
        assert a.to_prompt_block() == ""

    def test_goal_sourced_anchor_renders_even_single_line(self):
        """source='goal' is the explicit opt-in: render unconditionally."""
        a = TaskAnchor(goal="ship the feature", source_query="ship the feature", source="goal")
        block = a.to_prompt_block()
        assert "<original_task>" in block
        assert "GOAL: ship the feature" in block

    def test_goal_sourced_empty_goal_does_not_render(self):
        """Empty goal + no structured fields → empty even with source='goal'."""
        assert TaskAnchor(source="goal").to_prompt_block() == ""


class TestRunContext:
    def test_default_run_id_is_unique(self):
        a = RunContext()
        b = RunContext()
        assert a.run_id != b.run_id
        assert a.status == RunStatus.created
        assert a.source == RunSource.sdk

    def test_run_source_has_product_entrypoints(self):
        assert RunSource.cli.value == "cli"
        assert RunSource.gateway.value == "gateway"
        assert RunSource.cron.value == "cron"
        assert RunSource.workflow.value == "workflow"

    def test_run_config_source_defaults_to_sdk(self):
        from agentica.run_config import RunConfig

        assert RunConfig().source == RunSource.sdk

    def test_lifecycle_transitions(self):
        ctx = RunContext()
        ctx.mark_running()
        assert ctx.status == RunStatus.running
        ctx.mark_completed()
        assert ctx.status == RunStatus.completed
        assert ctx.ended_at is not None
        assert (ctx.duration_seconds or 0) >= 0

    def test_mark_failed_records_error(self):
        ctx = RunContext()
        ctx.mark_failed("kaboom")
        assert ctx.status == RunStatus.failed
        assert ctx.error == "kaboom"
        assert ctx.ended_at is not None

    def test_mark_cancelled_default_reason(self):
        ctx = RunContext()
        ctx.mark_cancelled()
        assert ctx.status == RunStatus.cancelled
        assert ctx.error == "user_cancelled"

    def test_to_dict_round_trip_shape(self):
        anchor = TaskAnchor(goal="g", source_query="g", source="goal")
        ctx = RunContext(
            session_id="s1",
            agent_id="a1",
            task_anchor=anchor,
            metadata={"foo": "bar"},
        )
        d = ctx.to_dict()
        assert d["run_id"] == ctx.run_id
        assert d["status"] == "created"
        assert d["task_anchor"]["goal"] == "g"
        assert d["task_anchor"]["source_query"] == "g"
        assert d["task_anchor"]["source"] == "goal"
        assert d["metadata"] == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_agent_run_respects_explicit_run_config_source(self):
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run_config import RunConfig

        response = MagicMock()
        response.content = "ok"
        response.parsed = None
        response.audio = None
        response.reasoning_content = None
        response.created_at = None

        with patch.object(OpenAIChat, "response", new_callable=AsyncMock, return_value=response):
            agent = Agent(name="A", model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"))
            await agent.run("Hi", config=RunConfig(source=RunSource.cli))

        assert agent.run_context.source == RunSource.cli

    @pytest.mark.asyncio
    async def test_plain_run_anchor_is_message_sourced_and_not_in_prompt(self):
        """Regression: `agent.run(msg)` must NOT inject msg into system prompt.

        The bug was: first message of a session was pinned as TaskAnchor and
        rendered into the system prompt every turn. For private-chat seed /
        resume / workflow-handoff this caused transcript leakage. Fix: default
        anchor source is "message" → to_prompt_block returns "".
        """
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run_config import RunConfig

        response = MagicMock()
        response.content = "ok"
        response.parsed = None
        response.audio = None
        response.reasoning_content = None
        response.created_at = None

        long_transcript = "User: hi\nAssistant: hi back\nUser: continue\n" * 5

        with patch.object(OpenAIChat, "response", new_callable=AsyncMock, return_value=response):
            agent = Agent(name="A", model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"))
            await agent.run(long_transcript, config=RunConfig(source=RunSource.sdk))

        assert agent.task_anchor is not None
        assert agent.task_anchor.source == "message"
        # The anchor block must NOT have been rendered into a system prompt
        # block. Easiest check: to_prompt_block stays empty.
        assert agent.task_anchor.to_prompt_block() == ""

    @pytest.mark.asyncio
    async def test_subagent_source_overrides_explicit_run_config_source(self):
        from agentica.agent import Agent
        from agentica.model.openai import OpenAIChat
        from agentica.run_config import RunConfig

        response = MagicMock()
        response.content = "ok"
        response.parsed = None
        response.audio = None
        response.reasoning_content = None
        response.created_at = None

        with patch.object(OpenAIChat, "response", new_callable=AsyncMock, return_value=response):
            agent = Agent(name="A", model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"))
            agent._parent_run_id = "parent-run"
            await agent.run("Hi", config=RunConfig(source=RunSource.cli))

        assert agent.run_context.source == RunSource.subagent


class TestRunEventRecord:
    def test_to_dict_flatten_payload(self):
        rec = RunEventRecord(
            run_id="r1",
            event_type=RunEventType.run_started,
            agent_id="a1",
            payload={"source_query": "hi"},
        )
        d = rec.to_dict()
        assert d["type"] == "run.started"
        assert d["run_id"] == "r1"
        assert d["agent_id"] == "a1"
        assert d["source_query"] == "hi"
