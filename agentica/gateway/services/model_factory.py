# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Model factory: instantiates configured LLM models and loads cron tools.

Extracted from AgentService to keep model creation logic independently testable
and to reduce AgentService's responsibility count.
"""
import asyncio
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


def _gateway_cron_job_runner(job: Any) -> dict:
    """Execute a cron job immediately and return its ``_execute_job`` result.

    Called from ``CronTool``'s sync tool entrypoint, which itself runs in a
    worker thread (see ``Tool.execute``). Bridges back to the gateway's own
    asyncio event loop with ``run_coroutine_threadsafe`` instead of spinning
    up a second loop (``asyncio.run``), so the run shares the same
    AgentService session locks/state as every other request — no cross-loop
    hazard.
    """
    from .. import deps
    from agentica.cron.scheduler import _execute_job

    if deps.cron_runner is None or deps.main_loop is None:
        return {"job_id": job.id, "status": "failed", "error": "Cron runner not ready"}
    future = asyncio.run_coroutine_threadsafe(
        _execute_job(job, agent_runner=deps.cron_runner, verbose=False),
        deps.main_loop,
    )
    return future.result()


def get_cron_tools() -> List[Any]:
    """Load the cron tool, wired for real immediate execution.

    Returns a ``CronTool`` (not the bare ``cronjob`` function) with a
    ``job_runner`` so ``cronjob(action="run", ...)`` actually executes the
    job now and returns its result — same as the CLI — instead of just
    marking it due for the next scheduler tick.
    """
    try:
        from agentica.tools.cron_tool import CronTool
        logger.debug("Loaded cron tool")
        return [CronTool(job_runner=_gateway_cron_job_runner)]
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


def get_self_manage_tools() -> List[Any]:
    """Load the self-management tool (same one the CLI gives its agent).

    Lets the web agent inspect its own config.yaml/.env, know its own
    version, and adjust its own model/tuning — the "self-awareness"
    capability the CLI has always had. Gated by the same permission_mode
    tiers as write_file/edit_file (hidden in "ask" mode, available in
    "auto"/"allow-all") since it's registered as destructive.
    """
    try:
        from agentica.tools.self_manage_tool import SelfManageTool
        logger.debug("Loaded self_manage tool")
        return [SelfManageTool()]
    except Exception as e:
        logger.warning(f"Failed to load self_manage tool: {e}")
        return []


def get_self_manage_instructions() -> str:
    """Return the system prompt instructions describing the agent's own identity."""
    return """
# Self-Awareness

You are Agentica, running as a web/API agent inside the Agentica Gateway — this
may be the browser chat UI or one of its bot channels (Feishu/WeCom/WeChat/
Telegram/Discord/DingTalk/QQ). You have a `self_manage` tool to inspect and
adjust your own runtime configuration:

- `self_manage(action="show")` — view your own config.yaml profiles, settings
  (including whether the cron scheduler daemon is enabled, its tick interval,
  etc.) and .env vars (secrets masked)
- `self_manage(action="set_config", key="...", value="...")` — edit a config.yaml profile field
- `self_manage(action="set_env", key="...", value="...")` — set/delete a .env variable
- `self_manage(action="check_upgrade")` — check for a newer Agentica release on PyPI
- `self_manage(action="install_skill", value="<git url or path>")` — install a skill

Use this when the user asks what model/version you're running, or wants you to
tune your own settings (e.g. "switch yourself to gpt-4", "raise your max_tokens").
Model/profile edits take effect on the next agent rebuild, not mid-conversation.

Important: there is no terminal here. Never tell the user to run a CLI-only
slash command (e.g. `/cron daemon on`, `/model`, `/upgrade`, `/config`) — this
surface has no command line and the user cannot type them. If something needs
deployment-level access you don't have (enabling the cron daemon, restarting
the server process after an upgrade), say so plainly and tell the user to
contact whoever manages this deployment instead of handing them a command.
"""
