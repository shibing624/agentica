# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Unified Usage model for cross-request token aggregation.

Provides type-safe token usage tracking that aggregates across multiple LLM calls
within a single agent run, following the OpenAI Agent SDK Usage pattern.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class TokenDetails(BaseModel):
    """Detailed token breakdown (cached, reasoning, etc.)."""
    cached_tokens: int = 0
    reasoning_tokens: int = 0


class RequestUsage(BaseModel):
    """Token usage for a single LLM request."""
    request_index: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[TokenDetails] = None
    output_tokens_details: Optional[TokenDetails] = None
    response_time: Optional[float] = None


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
        if entry.output_tokens_details:
            self.output_tokens_details.cached_tokens += entry.output_tokens_details.cached_tokens
            self.output_tokens_details.reasoning_tokens += entry.output_tokens_details.reasoning_tokens
        self.request_usage_entries.append(entry)

    def merge(self, other: "Usage") -> None:
        """Merge another Usage into this one (e.g., subagent usage into parent)."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.requests += other.requests
        self.input_tokens_details.cached_tokens += other.input_tokens_details.cached_tokens
        self.input_tokens_details.reasoning_tokens += other.input_tokens_details.reasoning_tokens
        self.output_tokens_details.cached_tokens += other.output_tokens_details.cached_tokens
        self.output_tokens_details.reasoning_tokens += other.output_tokens_details.reasoning_tokens
        self.request_usage_entries.extend(other.request_usage_entries)
