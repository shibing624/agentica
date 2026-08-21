# -*- coding: utf-8 -*-
"""Session usage payload shown by the web context chip (CLI /usage).

All tests mock LLM API keys — no real API usage.
"""
import asyncio

import pytest

pytest.importorskip("fastapi", reason="Gateway tests require agentica[gateway]")

from agentica.agent import Agent
from agentica.cost_tracker import CostTracker
from agentica.gateway.services.session_usage import turn_usage_payload, usage_payload
from agentica.model.openai import OpenAIChat
from agentica.model.usage import RequestUsage, TokenDetails


def _agent() -> Agent:
    return Agent(model=OpenAIChat(id="gpt-4o-mini", api_key="fake_openai_key"))


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
