# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified Usage model for cross-request token aggregation.

Provides type-safe token usage tracking that aggregates across multiple LLM calls
within a single agent run, following the OpenAI Agent SDK Usage pattern.
"""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


def split_prompt_usage(
    prompt_tokens: int,
    prompt_details: Optional[Dict[str, Any]],
) -> Tuple[int, int, int]:
    """Split a prompt into disjoint (fresh_input, cache_read, cache_write).

    Providers disagree on whether the cached prefix sits inside the headline
    prompt figure, and the two conventions need opposite arithmetic:

    - OpenAI reports ``prompt_tokens`` INCLUSIVE of
      ``prompt_tokens_details.cached_tokens`` — the cached count is a subset
      breakdown, so it has to be carved out.
    - Anthropic reports native ``input_tokens`` EXCLUSIVE of
      ``cache_read_input_tokens`` / ``cache_creation_input_tokens``. Some
      OpenAI-compatible gateways expose those names in ``prompt_tokens_details``
      while still keeping ``prompt_tokens`` inclusive, so the numeric relation is
      the only reliable discriminator at this boundary.

    The key names are the discriminator, because they come from whichever API
    contract produced the response. Treating an inclusive figure as exclusive
    counts the cached prefix twice — once at full price and once at the cache
    rate for cost, and again on top of itself for context size.

    Returns three parts that sum to the true prompt size and can each be priced
    exactly once.
    """
    details = prompt_details or {}
    cache_creation = details.get("cache_creation_tokens") or 0
    exclusive_read = details.get("cache_read_tokens") or 0
    if exclusive_read or cache_creation:
        if exclusive_read + cache_creation <= prompt_tokens:
            return (
                max(prompt_tokens - exclusive_read - cache_creation, 0),
                exclusive_read,
                cache_creation,
            )
        return prompt_tokens, exclusive_read, cache_creation

    inclusive_read = details.get("cached_tokens") or 0
    inclusive_write = details.get("cache_write_tokens") or 0
    return (
        max(prompt_tokens - inclusive_read - inclusive_write, 0),
        inclusive_read,
        inclusive_write,
    )


class TokenDetails(BaseModel):
    """Detailed token breakdown (cached, reasoning, etc.)."""
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    # Anthropic-style prompt-cache accounting (also returned by OpenAI-compatible
    # proxies that front Claude, e.g. Venus). ``cached_tokens`` is kept as the
    # OpenAI-native alias for cache reads; ``cache_read_tokens`` is the
    # Anthropic name for the same concept. ``cache_creation_tokens`` is the
    # one-time write cost (priced higher than a normal input token).
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def _details_dict(details: Optional[TokenDetails]) -> Dict[str, Any]:
    if details is None:
        return {}
    return {
        "cached_tokens": details.cached_tokens,
        "cache_read_tokens": details.cache_read_tokens,
        "cache_creation_tokens": details.cache_creation_tokens,
        "cache_write_tokens": 0,
    }


class RequestUsage(BaseModel):
    """Token usage for a single LLM request."""
    request_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[TokenDetails] = None
    output_tokens_details: Optional[TokenDetails] = None
    response_time: Optional[float] = None

    def cache_hit_ratio(self) -> Optional[float]:
        """cached / (fresh + cached + write) for this request, or None.

        None when the provider reported no cache counters at all — a 0.0 ratio
        would wrongly imply "cache missed" for providers without a cache.
        Conventions (inclusive vs exclusive cached counts) are normalised by
        ``split_prompt_usage``.
        """
        details = _details_dict(self.input_tokens_details)
        if not any(details.values()):
            return None
        fresh, hit, write = split_prompt_usage(self.input_tokens, details)
        total = fresh + hit + write
        return hit / total if total > 0 else None


class Usage(BaseModel):
    """Cross-request aggregated usage statistics.

    Accumulates token usage across multiple LLM calls within a single agent run.
    Provides both totals and per-request detail entries.

    Example::

        usage = Usage()
        usage.add(RequestUsage(input_tokens=100, output_tokens=50, total_tokens=150))
        usage.add(RequestUsage(input_tokens=200, output_tokens=80, total_tokens=280))
        assert usage.requests == 2
        assert usage.total_tokens == 430
        assert len(usage.request_usage_entries) == 2
    """

    # Aggregated totals
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Request count
    requests: int = 0

    # Aggregated detail breakdowns
    input_tokens_details: TokenDetails = Field(default_factory=TokenDetails)
    output_tokens_details: TokenDetails = Field(default_factory=TokenDetails)

    # Per-request entries
    request_usage_entries: List[RequestUsage] = Field(default_factory=list)

    def cache_hit_ratio(self) -> Optional[float]:
        """Aggregate cached share of all prompt tokens; None with no cache data."""
        details = _details_dict(self.input_tokens_details)
        if not any(details.values()):
            return None
        fresh, hit, write = split_prompt_usage(self.input_tokens, details)
        total = fresh + hit + write
        return hit / total if total > 0 else None

    def add(self, entry: RequestUsage) -> None:
        """Add a single request's usage to the aggregate."""
        entry.request_index = self.requests
        self.input_tokens += entry.input_tokens
        self.output_tokens += entry.output_tokens
        self.total_tokens += entry.total_tokens
        self.requests += 1
        if entry.input_tokens_details:
            self.input_tokens_details.cached_tokens += entry.input_tokens_details.cached_tokens
            self.input_tokens_details.reasoning_tokens += entry.input_tokens_details.reasoning_tokens
            self.input_tokens_details.cache_read_tokens += entry.input_tokens_details.cache_read_tokens
            self.input_tokens_details.cache_creation_tokens += entry.input_tokens_details.cache_creation_tokens
        if entry.output_tokens_details:
            self.output_tokens_details.cached_tokens += entry.output_tokens_details.cached_tokens
            self.output_tokens_details.reasoning_tokens += entry.output_tokens_details.reasoning_tokens
            self.output_tokens_details.cache_read_tokens += entry.output_tokens_details.cache_read_tokens
            self.output_tokens_details.cache_creation_tokens += entry.output_tokens_details.cache_creation_tokens
        self.request_usage_entries.append(entry)

    def merge(self, other: "Usage") -> None:
        """Merge another Usage into this one (e.g., subagent usage into parent)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.requests += other.requests
        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens
        self.input_tokens_details.reasoning_tokens += other.input_tokens_details.reasoning_tokens
        self.input_tokens_details.cache_read_tokens += other.input_tokens_details.cache_read_tokens
        self.input_tokens_details.cache_creation_tokens += other.input_tokens_details.cache_creation_tokens
        self.output_tokens_details.cached_tokens += other.output_tokens_details.cached_tokens
        self.output_tokens_details.reasoning_tokens += other.output_tokens_details.reasoning_tokens
        self.output_tokens_details.cache_read_tokens += other.output_tokens_details.cache_read_tokens
        self.output_tokens_details.cache_creation_tokens += other.output_tokens_details.cache_creation_tokens
        self.request_usage_entries.extend(other.request_usage_entries)
