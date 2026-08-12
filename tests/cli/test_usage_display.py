# -*- coding: utf-8 -*-
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands import model_config
from agentica.cli.usage_display import (
    ProviderUsageSummary,
    format_cache_hit,
    format_turn_usage_summary,
)
from agentica.cost_tracker import CostTracker
from agentica.model.usage import RequestUsage, TokenDetails, Usage


def test_turn_usage_summary_surfaces_provider_cache_hit_rate():
    summary = ProviderUsageSummary(
        input_tokens=38_100,
        cache_read_tokens=37_100,
        output_tokens=3_000,
        total_tokens_override=41_100,
    )

    assert format_cache_hit(summary) == "cache 37.1K / 97.4%"
    assert (
        format_turn_usage_summary(summary)
        == "+41.1K · in 38.1K · cache 37.1K / 97.4% · out 3K"
    )


def test_provider_usage_summary_uses_cost_tracker_disjoint_parts():
    tracker = CostTracker()
    tracker.record(
        "gpt-4o",
        input_tokens=1_000,
        output_tokens=3_000,
        cache_read_tokens=37_100,
    )

    summary = ProviderUsageSummary.from_cost_tracker(tracker)

    assert summary.input_tokens == 38_100
    assert summary.fresh_input_tokens == 1_000
    assert summary.cache_read_tokens == 37_100
    assert summary.prompt_tokens == 38_100
    assert summary.total_tokens == 41_100
    assert summary.cache_hit_percent == 97.4


def test_cache_hit_percent_keeps_near_full_hits_below_100():
    summary = ProviderUsageSummary(
        input_tokens=6_020,
        cache_read_tokens=6_016,
        output_tokens=35,
    )

    assert summary.cache_hit_percent == 99.9
    assert format_cache_hit(summary) == "cache 6.0K / 99.9%"


def test_provider_usage_summary_uses_provider_total_for_inclusive_cache_details():
    usage = RequestUsage(
        input_tokens=38_100,
        output_tokens=3_000,
        total_tokens=41_100,
        input_tokens_details=TokenDetails(cache_read_tokens=37_100),
    )

    summary = ProviderUsageSummary.from_request_entries(
        [usage],
        cache_counts_inside_input=True,
        cost_usd=0.04,
    )

    assert summary.prompt_tokens == 38_100
    assert summary.total_tokens == 41_100
    assert format_turn_usage_summary(summary) == (
        "+41.1K · in 38.1K · cache 37.1K / 97.4% · out 3K"
    )


def test_provider_usage_summary_adds_native_anthropic_cache_to_input():
    usage = RequestUsage(
        input_tokens=1_000,
        output_tokens=3_000,
        total_tokens=4_000,
        input_tokens_details=TokenDetails(
            cache_read_tokens=37_100,
            cache_creation_tokens=0,
        ),
    )

    summary = ProviderUsageSummary.from_request_entries(
        [usage],
        cache_counts_inside_input=False,
    )

    assert summary.prompt_tokens == 38_100
    assert summary.total_tokens == 41_100
    assert format_turn_usage_summary(summary) == (
        "+41.1K · in 38.1K · cache 37.1K / 97.4% · out 3K"
    )


def test_turn_usage_summary_omits_cache_write_segment():
    summary = ProviderUsageSummary(
        input_tokens=10_731,
        cache_read_tokens=10_718,
        cache_write_tokens=11,
        output_tokens=15,
        total_tokens_override=10_746,
    )

    rendered = format_turn_usage_summary(summary)

    assert rendered == "+10.7K · in 10.7K · cache 10.7K / 99.9% · out 15"
    assert "cache write" not in rendered


def test_usage_command_prints_real_provider_cache_breakdown(monkeypatch):
    tracker = CostTracker()
    tracker.record(
        "gpt-4o",
        input_tokens=1_000,
        output_tokens=3_000,
        cache_read_tokens=37_100,
    )
    usage = Usage()
    usage.add(
        RequestUsage(
            input_tokens=38_100,
            output_tokens=3_000,
            total_tokens=41_100,
            input_tokens_details=TokenDetails(cache_read_tokens=37_100),
        )
    )
    agent = SimpleNamespace(
        run_response=SimpleNamespace(cost_tracker=tracker),
        model=SimpleNamespace(usage=usage),
        working_memory=SimpleNamespace(messages=["user", "assistant"]),
    )
    ctx = CommandContext(
        agent_config={"model_provider": "openai", "model_name": "gpt-4o"},
        current_agent=agent,
        tui_state={
            "active_seconds": 5,
            "cost_usd": tracker.total_cost_usd,
            "total_api_calls": tracker.turns,
        },
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr(model_config, "get_console", lambda: console)
    monkeypatch.setattr(model_config, "_render_context_breakdown", lambda con, agent: None)

    model_config._cmd_usage(ctx)

    out = buf.getvalue()
    assert "Latest Turn API Usage" in out
    assert "Input tokens (total):" in out
    assert "38,100" in out
    assert "Fresh input tokens:" in out
    assert "1,000" in out
    assert "Cached input tokens:" in out
    assert "37,100 / 97.4% hit" in out
    assert "Output tokens:" in out
    assert "3,000" in out
    assert "Total tokens:" in out
    assert "41,100" in out
