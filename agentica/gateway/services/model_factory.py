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

from ..config import settings


def create_model(
    model_provider: str,
    model_name: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Any:
    """Instantiate the configured LLM model.

    Provider instantiation is delegated to the central ``provider_registry``
    (core classes openai/claude/kimi/azure + every OpenAI-compatible factory in
    ``PROVIDER_FACTORIES`` are seeded there). This factory only owns the
    gateway-specific request tuning (thinking mode, reasoning effort).

    Args:
        model_provider: Provider identifier (e.g. "openai", "zhipuai").
        model_name: Model identifier (e.g. "gpt-4o", "glm-4.7-flash").
        base_url: Optional override for the provider's base URL. Empty string
            is treated as "no override".
        api_key: Optional override for the provider's API key. Empty string
            is treated as "no override".

    Returns:
        A Model instance ready for use by an Agent.

    Raises:
        ValueError: If the provider is not recognized.
    """
    _ANTHROPIC_PROVIDERS = {"kimi", "anthropic", "claude"}

    params: dict[str, Any] = {"id": model_name, "timeout": 300}
    if base_url:
        params["base_url"] = base_url
    if api_key:
        params["api_key"] = api_key

    if settings.model_thinking and settings.model_thinking in ("enabled", "disabled", "auto"):
        if model_provider in _ANTHROPIC_PROVIDERS:
            params["thinking"] = {"type": settings.model_thinking, "budget_tokens": 8000}
        else:
            params["extra_body"] = {"thinking": {"type": settings.model_thinking}}
        logger.info(f"Model thinking mode: {settings.model_thinking} (provider={model_provider})")

    if settings.model_reasoning_effort:
        if settings.model_reasoning_effort not in ("high", "max"):
            raise ValueError("AGENTICA_REASONING_EFFORT must be one of: high, max")
        if model_provider == "deepseek":
            params["reasoning_effort"] = settings.model_reasoning_effort
            logger.info(f"DeepSeek reasoning effort: {settings.model_reasoning_effort}")

    # Delegate instantiation to the central registry (single source of truth
    # for the provider dispatch table — avoids drift with a parallel copy).
    from agentica.provider_registry import create_provider, get_provider_factory, list_providers
    if get_provider_factory(model_provider) is None:
        raise ValueError(
            f"Unknown model_provider '{model_provider}'. "
            f"Supported providers: {list_providers()}"
        )
    return create_provider(model_provider, **params)


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
