# -*- coding: utf-8 -*-
"""Session usage payload shown by the web context chip (CLI /usage).

All tests mock LLM API keys — no real API usage.
"""
import asyncio

import pytest

pytest.importorskip("agentica_gateway", reason="requires agentica-gateway")

from agentica.agent import Agent
from agentica.model.cost import CostTracker
from agentica_gateway.services.session_usage import turn_usage_payload, usage_payload
from agentica.model.openai import OpenAIChat
from agentica.model.usage import RequestUsage, TokenDetails


def _agent(**kwargs) -> Agent:
    return Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"), **kwargs)


def test_fresh_session_has_prompt_breakdown_and_no_api_calls():
    payload = asyncio.run(usage_payload(_agent(), model_provider="openai"))
    labels = [row["label"] for row in payload["sections"]]
    assert "System prompt" in labels
    assert payload["context_tokens"] == sum(row["tokens"] for row in payload["sections"])
    assert payload["window"] > 0
    assert payload["api_calls"] == 0
    assert payload["cost_usd"] == 0
    assert payload["messages"] >= 0
    assert payload["model"] == "openai/gpt-4o-mini"
    assert all("nested" not in row for row in payload["sections"])
    shares = [row["share"] for row in payload["sections"]]
    if shares:
        assert abs(sum(shares) - 1.0) < 1e-9


def test_request_entries_become_session_api_calls_and_input():
    agent = _agent()
    agent.model.usage.add(RequestUsage(
        input_tokens=1200,
        output_tokens=80,
        total_tokens=1280,
        input_tokens_details=TokenDetails(cached_tokens=200),
    ))
    payload = asyncio.run(usage_payload(agent, model_provider="openai"))
    assert payload["api_calls"] == 1
    assert payload["input_tokens"] == 1200
    assert payload["output_tokens"] == 80
    assert payload["cache_read_tokens"] == 200
    assert payload["cost_usd"] >= 0


def test_turn_usage_is_this_run_cache_split_not_session_total():
    """The message footer must not reuse the session-cumulative /usage cache."""
    from types import SimpleNamespace
    agent = _agent()
    agent.model.usage.add(RequestUsage(
        input_tokens=5000,
        output_tokens=10,
        total_tokens=5010,
        input_tokens_details=TokenDetails(cached_tokens=4000),
    ))
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", input_tokens=200, output_tokens=50, cache_read_tokens=800)
    agent.run_response = SimpleNamespace(cost_tracker=tracker)
    payload = turn_usage_payload(agent)
    assert payload is not None
    assert payload["cache_read_tokens"] == 800
    assert payload["output_tokens"] == 50
    assert payload["input_tokens"] == 1000  # fresh 200 + cache 800
    assert payload["cache_hit_percent"] == 80.0
    session = asyncio.run(usage_payload(agent, model_provider="openai"))
    assert session["cache_read_tokens"] == 4000


def test_turn_usage_carries_what_the_footer_detail_shows():
    """The click-open usage panel lists cache write, request count and models."""
    from types import SimpleNamespace
    agent = _agent()
    tracker = CostTracker()
    tracker.record("gpt-4o-mini", input_tokens=100, output_tokens=20, cache_write_tokens=300)
    tracker.record("openai/gpt-4o", input_tokens=50, output_tokens=5, cache_read_tokens=400)
    tracker.record("gpt-4o-mini", input_tokens=10, output_tokens=1)
    agent.run_response = SimpleNamespace(cost_tracker=tracker)
    payload = turn_usage_payload(agent)
    assert payload["cache_write_tokens"] == 300
    assert payload["api_calls"] == 3
    assert payload["models"] == ["gpt-4o-mini", "gpt-4o"]


def test_payload_window_is_the_budget_the_policy_compacts_against():
    """The web renders ``context_tokens / window``.

    Reporting the model limit here made a capped session read as half empty
    while the runner was evicting against the cap. The ratio every consumer
    shows must be the policy's own ratio.
    """
    from agentica.agent.config import ToolConfig

    agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"),
        tool_config=ToolConfig(compact_token_limit=32_000),
    )
    agent.model.context_window = 128_000

    payload = asyncio.run(usage_payload(agent, model_provider="openai"))

    assert payload["window"] == 32_000


def test_payload_window_is_the_model_limit_without_a_cap():
    agent = _agent()
    agent.model.context_window = 128_000

    payload = asyncio.run(usage_payload(agent, model_provider="openai"))

    assert payload["window"] == 128_000


def test_payload_message_count_matches_the_measured_conversation():
    """`messages` and `sections` must describe the same set.

    The count read ``working_memory.messages`` while the Conversation row
    measured the last N runs. Those are separate stores, and on a resumed
    session the first is empty — the chip showed "0 messages" beside a 594K
    conversation.
    """
    from agentica.memory.models import AgentRun
    from agentica.model.message import Message
    from agentica.run.response import RunResponse

    agent = _agent(add_history_to_context=True)
    agent.model.context_window = 128_000
    agent.working_memory.add_run(AgentRun(response=RunResponse(messages=[
        Message(role="user", content="q " * 100),
        Message(role="assistant", content="a " * 100),
    ])))

    payload = asyncio.run(usage_payload(agent, model_provider="openai"))

    assert payload["messages"] == 2


def test_payload_message_count_is_zero_when_history_is_not_replayed():
    """Nothing in the prompt means nothing to count, whatever is stored."""
    from agentica.memory.models import AgentRun
    from agentica.model.message import Message
    from agentica.run.response import RunResponse

    agent = _agent(add_history_to_context=False)
    agent.model.context_window = 128_000
    agent.working_memory.add_run(AgentRun(response=RunResponse(
        messages=[Message(role="user", content="ignored " * 100)]
    )))

    payload = asyncio.run(usage_payload(agent, model_provider="openai"))

    assert payload["messages"] == 0


def test_session_stats_come_from_the_log_and_survive_a_gateway_restart(tmp_path, monkeypatch):
    """The cached Agent forgets its usage on restart; the log does not."""
    from unittest.mock import AsyncMock

    from agentica.memory.session_log import SessionLog
    from agentica_gateway.services import agent_service as svc_mod
    from agentica_gateway.services.agent_service import AgentService

    log = SessionLog(session_id="s1", base_dir=str(tmp_path))
    log.append("user", "count files")
    log.append_event("request_begin", timestamp="2026-01-01T00:00:01.000Z")
    log.append_event("token_usage", timestamp="2026-01-01T00:00:03.000Z",
                     request={"input": 100, "cache_read": 900, "cache_write": 0, "output": 40, "total": 1040})
    log.append_event("request_end", timestamp="2026-01-01T00:00:03.000Z", status="completed")
    log.append("assistant", "12 files", metrics={"time_to_first_token": 0.5})

    svc = AgentService(workspace_path=str(tmp_path))
    monkeypatch.setattr(svc, "_ensure_initialized", AsyncMock())
    monkeypatch.setattr(svc, "_get_agent", AsyncMock(return_value=None))
    # A freshly built Agent: no requests in memory.
    monkeypatch.setattr(svc_mod, "usage_payload", AsyncMock(return_value={"api_calls": 0, "cost_usd": 0}))
    monkeypatch.setattr(svc, "session_log_for", lambda sid, owner=None: log)

    stats = asyncio.run(svc.session_usage("s1"))["stats"]
    assert stats["rounds"] == 1 and stats["steps"] == 1
    assert stats["llm_ms"] == 2000
    assert stats["ttft_ms"] == 500
    assert stats["tps"] == 20.0
    assert stats["uncached_input_tokens"] == 100
    assert stats["cache_read_tokens"] == 900
    assert stats["output_tokens"] == 40
    assert stats["cache_hit_percent"] == 90.0
