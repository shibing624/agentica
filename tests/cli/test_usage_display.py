# -*- coding: utf-8 -*-
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from agentica.cli.commands.context import CommandContext
from agentica.cli.commands import model_config
from agentica.cli.usage_display import (
    ProviderUsageSummary,
    format_cache_hit,
    format_cost_usd,
    format_turn_usage_summary,
)
from agentica.cost_tracker import CostTracker
from agentica.model.usage import RequestUsage, TokenDetails, Usage


def test_turn_usage_summary_surfaces_provider_cache_hit_rate():
    # 38.1K prompt = 37.1K cache read + 1K fresh; net new = 1K fresh + 3K out.
    summary = ProviderUsageSummary(
        input_tokens=38_100,
        fresh_input_tokens=1_000,
        cache_read_tokens=37_100,
        output_tokens=3_000,
        total_tokens_override=41_100,
    )

    assert format_cache_hit(summary) == "cache 37.1K / 97.4%"
    assert (
        format_turn_usage_summary(summary)
        == "+4K · in 38.1K · cache 37.1K / 97.4% · out 3K"
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
        cost_usd=0.04,
    )

    assert summary.prompt_tokens == 38_100
    assert summary.total_tokens == 41_100
    assert format_turn_usage_summary(summary) == (
        "+4K · in 38.1K · cache 37.1K / 97.4% · out 3K"
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
    )

    assert summary.prompt_tokens == 38_100
    assert summary.total_tokens == 41_100
    assert format_turn_usage_summary(summary) == (
        "+4K · in 38.1K · cache 37.1K / 97.4% · out 3K"
    )


def test_provider_usage_summary_normalises_mixed_provider_entries_per_request():
    openai_usage = RequestUsage(
        input_tokens=1_000,
        output_tokens=10,
        input_tokens_details=TokenDetails(cached_tokens=900, cache_read_tokens=900),
    )
    anthropic_usage = RequestUsage(
        input_tokens=100,
        output_tokens=20,
        input_tokens_details=TokenDetails(cache_read_tokens=800),
    )

    summary = ProviderUsageSummary.from_request_entries(
        [openai_usage, anthropic_usage],
    )

    assert summary.prompt_tokens == 1_900
    assert summary.fresh_input_tokens == 200
    assert summary.cache_read_tokens == 1_700
    assert summary.output_tokens == 30
    assert summary.total_tokens == 1_930
    assert summary.cache_hit_percent == 89.5


def test_turn_usage_summary_omits_cache_write_segment():
    # 10,731 prompt = 10,718 cache read + 11 cache write + 2 fresh.
    # Net new = 2 fresh + 11 write + 15 out = 28; only the re-read 10,718 is excluded.
    summary = ProviderUsageSummary(
        input_tokens=10_731,
        fresh_input_tokens=2,
        cache_read_tokens=10_718,
        cache_write_tokens=11,
        output_tokens=15,
        total_tokens_override=10_746,
    )

    rendered = format_turn_usage_summary(summary)

    assert rendered == "+28 · in 10.7K · cache 10.7K / 99.9% · out 15"
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
            "last_turn_tool_count": 3,
        },
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr(model_config, "get_console", lambda: console)
    monkeypatch.setattr(model_config, "_render_context_breakdown", lambda con, agent: None)

    model_config._cmd_usage(ctx)

    out = buf.getvalue()
    assert "Latest Turn API Usage" in out
    assert "Input tokens:" in out
    assert "38,100" in out
    assert "Fresh input tokens:" in out
    assert "1,000" in out
    assert "Cached input tokens:" in out
    assert "37,100 / 97.4% hit" in out
    assert "Output tokens:" in out
    assert "3,000" in out
    assert "Net new tokens:" in out
    assert "4,000" in out
    assert "Tool calls this turn:" in out
    assert "Total tokens (billed):" in out
    assert "41,100" in out


def test_usage_command_omits_net_new_row_when_nothing_re_read(monkeypatch):
    # With cache_read == 0, net new equals billed total, so the extra row would
    # just repeat the next line under a second label.
    tracker = CostTracker()
    tracker.record("gpt-4o", input_tokens=1_000, output_tokens=500)
    usage = Usage()
    usage.add(RequestUsage(input_tokens=1_000, output_tokens=500, total_tokens=1_500))
    agent = SimpleNamespace(
        run_response=SimpleNamespace(cost_tracker=tracker),
        model=SimpleNamespace(usage=usage),
        working_memory=SimpleNamespace(messages=["user"]),
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
    assert "Net new tokens:" not in out
    assert "Total tokens (billed):" in out
    assert "1,500" in out


def test_format_cost_usd_adaptive_precision():
    assert format_cost_usd(0.004) == "$0.0040"
    assert format_cost_usd(0.04) == "$0.04"
    assert format_cost_usd(0.009999) == "$0.01"


def test_format_cost_usd_floors_sub_precision_cost_instead_of_showing_zero():
    # A cost too small for 4 decimals must not render as "$0.0000" — that is the
    # same "looks free" lie the adaptive precision exists to kill.
    assert format_cost_usd(0.000014) == "<$0.0001"
    assert format_cost_usd(0.0001) == "$0.0001"
    assert format_cost_usd(0.0) == "$0.0000"


def test_format_cost_usd_signed_prefixes_delta_but_not_the_floor():
    assert format_cost_usd(0.004, signed=True) == "+$0.0040"
    assert format_cost_usd(1.5, signed=True) == "+$1.50"
    assert format_cost_usd(0.000014, signed=True) == "<$0.0001"


def test_net_new_tokens_counts_cache_writes_as_new():
    # Cache writes are content sent for the first time (and billed at a
    # premium), so a prefix-rebuilding turn must not read as a cheap one.
    summary = ProviderUsageSummary(
        input_tokens=38_100,
        fresh_input_tokens=1_000,
        cache_read_tokens=36_600,
        cache_write_tokens=500,
        output_tokens=3_000,
    )
    assert summary.net_new_tokens == 4_500


def test_net_new_tokens_excludes_re_read_cache():
    summary = ProviderUsageSummary(
        input_tokens=38_100,
        fresh_input_tokens=1_000,
        cache_read_tokens=37_100,
        output_tokens=3_000,
    )
    assert summary.net_new_tokens == 4_000


def test_turn_usage_summary_omits_leading_zero_for_full_cache_turn():
    summary = ProviderUsageSummary(
        input_tokens=76_200,
        fresh_input_tokens=0,
        cache_read_tokens=76_200,
        output_tokens=0,
    )
    assert format_turn_usage_summary(summary) == "in 76.2K · cache 76.2K / 100.0%"
