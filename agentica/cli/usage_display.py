# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Provider-reported token usage formatting for CLI displays
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

from agentica.model.usage import split_prompt_usage

if TYPE_CHECKING:
    from agentica.cost_tracker import CostTracker
    from agentica.model.usage import RequestUsage


@dataclass(frozen=True)
class ProviderUsageSummary:
    """Provider-reported usage parts prepared for CLI display."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    fresh_input_tokens: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0
    total_tokens_override: int = 0

    @classmethod
    def from_cost_tracker(cls, tracker: "CostTracker") -> "ProviderUsageSummary":
        return cls(
            input_tokens=tracker.total_prompt_tokens,
            output_tokens=tracker.total_output_tokens,
            cache_read_tokens=tracker.total_cache_read_tokens,
            cache_write_tokens=tracker.total_cache_write_tokens,
            fresh_input_tokens=tracker.total_input_tokens,
            cost_usd=tracker.total_cost_usd,
            api_calls=tracker.turns,
        )

    @classmethod
    def from_request_entries(
        cls,
        entries: Iterable["RequestUsage"],
        *,
        cache_counts_inside_input: bool,
        cost_usd: float = 0.0,
    ) -> "ProviderUsageSummary":
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cache_read_tokens = 0
        cache_write_tokens = 0
        fresh_input_tokens = 0
        api_calls = 0

        for entry in entries:
            api_calls += 1
            output_tokens += entry.output_tokens
            details = entry.input_tokens_details
            detail_dict = {}
            if details is not None:
                detail_dict = {
                    "cached_tokens": details.cached_tokens,
                    "cache_read_tokens": details.cache_read_tokens,
                    "cache_creation_tokens": details.cache_creation_tokens,
                }
            fresh, cache_read, cache_write = split_prompt_usage(entry.input_tokens, detail_dict)
            cache_read_tokens += cache_read
            cache_write_tokens += cache_write
            prompt_tokens = fresh + cache_read + cache_write
            fresh_input_tokens += fresh
            input_tokens += prompt_tokens
            total_tokens += prompt_tokens + entry.output_tokens

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            fresh_input_tokens=fresh_input_tokens,
            cost_usd=cost_usd,
            api_calls=api_calls,
            total_tokens_override=total_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def total_tokens(self) -> int:
        if self.total_tokens_override > 0:
            return self.total_tokens_override
        return self.prompt_tokens + self.output_tokens

    @property
    def cache_hit_percent(self) -> float | None:
        if self.prompt_tokens <= 0 or self.cache_read_tokens <= 0:
            return None
        raw_percent = self.cache_read_tokens / self.prompt_tokens * 100
        rounded = round(raw_percent, 1)
        if raw_percent < 100 and rounded >= 100:
            return 99.9
        return rounded


def format_tokens_short(n: int) -> str:
    """Format token count with K/M suffix for compact CLI use."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{int(v)}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{int(v)}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def cache_counts_inside_input_for_model(model: object) -> bool:
    """Whether provider input_tokens already includes cache read/write tokens."""
    module = model.__class__.__module__
    return not module.startswith("agentica.model.anthropic.")


def format_cache_hit(summary: ProviderUsageSummary) -> str | None:
    """Return the cache-hit segment, or None when provider reported no hit."""
    hit_percent = summary.cache_hit_percent
    if hit_percent is None:
        return None
    return f"cache {format_tokens_short(summary.cache_read_tokens)} / {hit_percent:.1f}%"


def format_turn_usage_summary(summary: ProviderUsageSummary) -> str:
    """Compact footer text for provider-reported per-turn usage."""
    if summary.total_tokens <= 0:
        return ""

    parts = [
        f"+{format_tokens_short(summary.total_tokens)}",
        f"in {format_tokens_short(summary.prompt_tokens)}",
    ]

    cache_hit = format_cache_hit(summary)
    if cache_hit is not None:
        parts.append(cache_hit)
    if summary.output_tokens > 0:
        parts.append(f"out {format_tokens_short(summary.output_tokens)}")
    return " · ".join(parts)
