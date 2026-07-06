# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Model factory: instantiates configured LLM models and loads cron tools.

Extracted from AgentService to keep model creation logic independently testable
and to reduce AgentService's responsibility count.
"""
from typing import Any, List, Optional

from agentica.utils.log import logger


def create_model(
    model_provider: str,
    model_name: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 0,
    temperature: float = 0.0,
    top_p: float = 0.0,
    context_window: int = 0,
    reasoning_effort: str = "",
    thinking: str = "",
) -> Any:
    """Instantiate the configured LLM model.

    Provider instantiation is delegated to the central ``provider_registry``
    (core classes openai/claude/kimi/azure + every OpenAI-compatible factory in
    ``PROVIDER_FACTORIES`` are seeded there). This factory only owns the
    gateway-specific request tuning (thinking mode, reasoning effort, sampling).

    Args:
        model_provider: Provider slug (e.g. "openai", "zhipuai", "ark").
        model_name: Model identifier (e.g. "gpt-4o", "glm-4.7-flash").
        base_url: Optional override for the provider's base URL.
        api_key: Optional override for the provider's API key.
        max_tokens: Output token limit (0 = leave model default).
        temperature: Sampling temperature (0 = leave default).
        top_p: Nucleus sampling probability (0 = leave default).
        context_window: Context limit; overrides the catalog auto-detected
            value. NOT sent to the API — set on the instance for budget /
            compression / status display only.
        reasoning_effort: low|medium|high|max (OpenAI/DeepSeek); ignored for
            Anthropic-family providers which use a thinking budget instead.
        thinking: enabled|disabled|auto — Anthropic-family providers use a
            ``thinking`` dict; others get it via ``extra_body``.

    Returns:
        A Model instance ready for use by an Agent.

    Raises:
        ValueError: If the provider is not recognized or tuning is invalid.
    """
    _ANTHROPIC_PROVIDERS = {"kimi", "anthropic", "claude"}

    params: dict[str, Any] = {"id": model_name, "timeout": 300}
    if base_url:
        params["base_url"] = base_url
    if api_key:
        params["api_key"] = api_key
    if max_tokens:
        params["max_tokens"] = max_tokens
    if temperature:
        params["temperature"] = temperature
    if top_p:
        params["top_p"] = top_p

    # Thinking mode — Anthropic-family gets a dict, others go via extra_body.
    if thinking and thinking in ("enabled", "disabled", "auto"):
        if model_provider in _ANTHROPIC_PROVIDERS:
            params["thinking"] = {"type": thinking, "budget_tokens": 8000}
        else:
            params["extra_body"] = {"thinking": {"type": thinking}}
        logger.info(f"Model thinking mode: {thinking} (provider={model_provider})")

    # Reasoning effort — low/medium/high/max. Anthropic-family skips it
    # (uses thinking budget instead); DeepSeek/OpenAI/o-series accept it
    # directly; other OpenAI-compatible providers get it via extra_body.
    effort = reasoning_effort
    if effort:
        if effort not in ("low", "medium", "high", "max"):
            raise ValueError(
                f"reasoning_effort must be one of: low, medium, high, max (got {effort!r})"
            )
        if model_provider not in _ANTHROPIC_PROVIDERS:
            if model_provider in ("deepseek", "openai", "openrouter"):
                params["reasoning_effort"] = effort
            else:
                extra = params.get("extra_body") or {}
                extra["reasoning_effort"] = effort
                params["extra_body"] = extra
            logger.info(f"Reasoning effort: {effort} (provider={model_provider})")

    # Delegate instantiation to the central registry (single source of truth
    # for the provider dispatch table — avoids drift with a parallel copy).
    from agentica.provider_registry import create_provider, get_provider_factory, list_providers
    if get_provider_factory(model_provider) is None:
        raise ValueError(
            f"Unknown model_provider '{model_provider}'. "
            f"Supported providers: {list_providers()}"
        )
    model = create_provider(model_provider, **params)

    # context_window is NOT an API param — set on the instance for budget /
    # compression / status display. Overrides the catalog auto-detected value.
    if context_window:
        try:
            model.context_window = context_window
        except (AttributeError, TypeError):
            logger.debug(f"Could not set context_window on {type(model).__name__}")

    return model


def get_cron_tools() -> List[Any]:
    """Load cron tools from the SDK cron module.

    Returns:
        A list containing the cronjob tool function, or an empty list
        if the cron module is not available.
    """
    try:
        from agentica.tools.cron_tool import cronjob
        logger.debug("Loaded cron tool")
        return [cronjob]
    except Exception as e:
        logger.warning(f"Failed to load cron tools: {e}")
        return []


def get_cron_instructions() -> str:
    """Return the system prompt instructions for cron job management."""
    return """
# Cron Job Management

You can help users create and manage scheduled tasks. Use the `cronjob` tool:

- `cronjob(action="create", prompt="...", schedule="30 7 * * *")` — Create a cron job
- `cronjob(action="list")` — List all cron jobs
- `cronjob(action="pause", job_id="...")` — Pause a job
- `cronjob(action="resume", job_id="...")` — Resume a paused job
- `cronjob(action="remove", job_id="...")` — Delete a job
- `cronjob(action="run", job_id="...")` — Trigger a job immediately

Schedule formats: cron expression ("30 7 * * *"), interval ("30m", "every 2h"), or ISO datetime.
"""
