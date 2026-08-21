# -*- coding: utf-8 -*-
"""Session-level context occupancy, same numbers as CLI ``/usage``.

``measure_context`` is the only source for the window breakdown (system
prompt / rules / skills / tools / conversation). Provider usage on the
cached Agent is the source for billed tokens, API calls, and cost.
"""
from typing import Any, Dict, Iterable, Optional

from agentica.cli.context_usage import measure_context
from agentica.cli.usage_display import ProviderUsageSummary
from agentica.cost_tracker import CostTracker
from agentica.model.usage import RequestUsage, split_prompt_usage


def _cost_from_entries(model_id: str, entries: Iterable[RequestUsage]) -> float:
    """Price every recorded request with the same CostTracker the run uses."""
    tracker = CostTracker()
    for entry in entries:
        details: Dict[str, Any] = {}
        token_details = entry.input_tokens_details
        if token_details is not None:
            details = {
                "cached_tokens": token_details.cached_tokens,
                "cache_read_tokens": token_details.cache_read_tokens,
                "cache_creation_tokens": token_details.cache_creation_tokens,
            }
        fresh, cache_read, cache_write = split_prompt_usage(entry.input_tokens, details)
        tracker.record(model_id, fresh, entry.output_tokens, cache_read, cache_write)
    return tracker.total_cost_usd


async def usage_payload(agent: Any, *, model_provider: str = "") -> Dict[str, Any]:
    """CLI ``/usage`` as JSON: occupancy breakdown plus session billing."""
    breakdown = await measure_context(agent)
    usage = agent.model.usage
    entries = list(usage.request_usage_entries) if usage is not None else []
    model_id = agent.model.id or ""
    cost_usd = _cost_from_entries(model_id, entries)
    if not entries:
        tracker = agent.run_response.cost_tracker if agent.run_response else None
        if tracker is not None:
            summary = ProviderUsageSummary.from_cost_tracker(tracker)
        else:
            summary = ProviderUsageSummary()
    else:
        summary = ProviderUsageSummary.from_request_entries(entries, cost_usd=cost_usd)

    total = breakdown.total
    sections = [
        {
            "label": label,
            "tokens": tokens,
            "share": (tokens / total) if total else 0.0,
        }
        for label, tokens in breakdown.visible_sections()
    ]
    provider = model_provider.strip()
    model = f"{provider}/{model_id}" if provider and model_id else (model_id or provider)

    return {
        "model": model,
        "window": breakdown.window,
        "context_tokens": total,
        "percent_full": round(breakdown.percent_full, 1),
        "messages": len(agent.working_memory.messages),
        "api_calls": summary.api_calls,
        "cost_usd": round(summary.cost_usd, 6),
        "input_tokens": summary.prompt_tokens,
        "output_tokens": summary.output_tokens,
        "cache_read_tokens": summary.cache_read_tokens,
        "cache_write_tokens": summary.cache_write_tokens,
        "cache_hit_percent": summary.cache_hit_percent,
        "sections": sections,
    }


def turn_usage_payload(agent: Any) -> Optional[Dict[str, Any]]:
    """Per-turn usage from this run's CostTracker — the same split the CLI footer shows.

    Session ``usage_payload`` is cumulative. The message footer needs this
    turn's cache read / hit, which live on ``run_response.cost_tracker``.
    """
    tracker = agent.run_response.cost_tracker if agent.run_response else None
    if not isinstance(tracker, CostTracker):
        return None
    summary = ProviderUsageSummary.from_cost_tracker(tracker)
    return {
        "input_tokens": summary.prompt_tokens,
        "output_tokens": summary.output_tokens,
        "cache_read_tokens": summary.cache_read_tokens,
        "cache_write_tokens": summary.cache_write_tokens,
        "cache_hit_percent": summary.cache_hit_percent,
        "cost_usd": round(summary.cost_usd, 6),
        "net_new_tokens": summary.net_new_tokens,
    }
