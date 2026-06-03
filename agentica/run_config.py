# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: RunConfig - Per-run configuration that overrides Agent defaults.

Separates "each run may differ" parameters from Agent construction.
Agent fields are defaults; RunConfig overrides them for a specific run.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Type,
    Union,
)

from agentica.hooks import RunHooks
from agentica.run_context import RunSource

if TYPE_CHECKING:
    from agentica.model.base import Model


@dataclass
class RunConfig:
    """Per-run configuration. Overrides Agent defaults when provided.

    Packs run-level parameters that would otherwise be repeated across
    run()/run_stream()/run_sync()/run_stream_sync() signatures.

    Example:
        >>> response = await agent.run("Analyze data", config=RunConfig(
        ...     run_timeout=30,
        ...     response_model=AnalysisReport,
        ...     enabled_tools=["web_search", "read_file"],
        ... ))
    """
    response_model: Optional[Type[Any]] = None
    use_structured_outputs: Optional[bool] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_timeout: Optional[float] = None
    first_token_timeout: Optional[float] = None
    # Max seconds between consecutive streaming tokens before cancelling.
    # Detects "silent hang" (connection alive but no data flowing).
    # Mirrors CC's stream idle watchdog in claude.ts.
    idle_timeout: Optional[float] = None
    save_response_to_file: Optional[str] = None
    stream_intermediate_steps: bool = False
    hooks: Optional[RunHooks] = None
    # Query-level whitelist: None = keep agent defaults, list = only these tools/skills
    enabled_tools: Optional[List[str]] = None
    enabled_skills: Optional[List[str]] = None
    # Cost budget: stop the run when total cost exceeds this amount (USD).
    # None = no limit. Works with CostTracker (always active).
    max_cost_usd: Optional[float] = None
    # Runner-level API call attempts per model for this run. None keeps the
    # Agent default. 1 means no same-model retry; fallback can still switch
    # models immediately when configured.
    max_api_retry: Optional[int] = None
    # Explicit provenance for this run. Product entry points should set this.
    source: RunSource = RunSource.sdk
    # Fallback model chain (cross-provider). Triggered per-call by:
    #   - finish_reason == "content_filter" (provider-side content moderation)
    #   - fallback-only API errors (connection / 502 / 503 / bad gateway)
    #   - retryable API errors that exhausted local backoff (timeout / 429)
    # Each call starts from the primary agent.model; fallbacks are tried in order.
    # Primary model is always retried on the next call (per-call switch, not per-run).
    # Use cross-provider models — same provider often shares the moderation layer.
    fallback_models: List["Model"] = field(default_factory=list)
    # Break recovery: when the agentic loop is aborted by a safety check
    # (death spiral / max turns / cost budget) the assistant content may be empty
    # or partial. If True AND `fallback_models` is non-empty, the Runner does one
    # final tool-free inference with the fallback chain, replaying the full
    # history (including the failed tool calls — so the model sees what went
    # wrong), and uses that as the answer. The run still reports break_reason;
    # `RunResponse.fallback_used` flips True and `RunResponse.model` reflects the
    # fallback that answered. Only loop-breaks trigger this — a raised exception
    # is left to the caller. Defaults to False (opt-in, no behavior change).
    fallback_on_break: bool = False
