# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Model factory: instantiates configured LLM models and loads cron tools.

Extracted from AgentService to keep model creation logic independently testable
and to reduce AgentService's responsibility count.
"""
import asyncio
import dataclasses
from typing import Any, List, Optional

from agentica.utils.log import logger


def create_model(
    model_provider: str,
    model_name: str,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    wire_api: str = "",
    max_tokens: int = 0,
    temperature: float = 0.0,
    top_p: float = 0.0,
    context_window: int = 0,
    reasoning: str = "",
    reasoning_effort: str = "",
    thinking: str = "",
    enable_cache_control: Optional[bool] = None,
    cache_control_messages: Optional[int] = None,
    cache_control_session_header: Optional[str] = None,
    cache_keepalive: Optional[int] = None,
    extra_body: Optional[dict[str, Any]] = None,
    extra_headers: Optional[dict[str, Any]] = None,
    default_headers: Optional[dict[str, Any]] = None,
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
        wire_api: ``responses`` selects OpenAI Responses API; empty or
            ``chat_completions`` uses the provider's default model class.
        max_tokens: Output token limit (0 = leave model default).
        temperature: Sampling temperature (0 = leave default).
        top_p: Nucleus sampling probability (0 = leave default).
        context_window: Context limit; overrides the catalog auto-detected
            value. NOT sent to the API — set on the instance for budget /
            compression / status display only.
        reasoning: Responses API effort: none|minimal|low|medium|high|xhigh|max.
        reasoning_effort: passed through as-is (no enum). Ignored for
            Anthropic-family providers which use a thinking budget instead.
        thinking: enabled|disabled|auto — Anthropic-family providers use a
            ``thinking`` dict; others get it via ``extra_body``.

    Returns:
        A Model instance ready for use by an Agent.

    Raises:
        ValueError: If the provider is not recognized or tuning is invalid.
    """
    _ANTHROPIC_PROVIDERS = {"kimi", "anthropic", "claude"}
    effective_wire_api = wire_api or "chat_completions"
    if effective_wire_api not in ("chat_completions", "responses"):
        raise ValueError("wire_api must be either 'chat_completions' or 'responses'")
    if wire_api and model_provider != "openai":
        raise ValueError("wire_api requires model_provider='openai'")
    if reasoning and effective_wire_api != "responses":
        raise ValueError("reasoning requires wire_api='responses'")
    if effective_wire_api == "responses" and reasoning_effort:
        raise ValueError("Responses API uses reasoning, not reasoning_effort")
    if reasoning and reasoning not in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError(f"Invalid Responses reasoning effort: {reasoning!r}")

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
    if reasoning:
        params["reasoning"] = reasoning

    # Thinking mode — Anthropic-family gets a dict, others go via extra_body.
    if thinking and thinking in ("enabled", "disabled", "auto"):
        if model_provider in _ANTHROPIC_PROVIDERS:
            params["thinking"] = {"type": thinking, "budget_tokens": 8000}
        else:
            params["extra_body"] = {"thinking": {"type": thinking}}
        logger.info(f"Model thinking mode: {thinking} (provider={model_provider})")

    if model_provider in _ANTHROPIC_PROVIDERS:
        if default_headers:
            params["default_headers"] = dict(default_headers)
    else:
        if extra_body:
            merged_body = dict(params.get("extra_body") or {})
            merged_body.update(extra_body)
            params["extra_body"] = merged_body
        if extra_headers:
            params["extra_headers"] = dict(extra_headers)

    # Reasoning effort — any string the endpoint accepts. Anthropic-family
    # skips it (uses thinking budget instead); DeepSeek/OpenAI/o-series
    # take it as a top-level field; other OpenAI-compatible providers get
    # it via extra_body.
    effort = reasoning_effort
    if effort:
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
    from agentica.model.providers import create_provider, get_provider_factory, list_providers
    if get_provider_factory(model_provider) is None:
        raise ValueError(
            f"Unknown model_provider '{model_provider}'. "
            f"Supported providers: {list_providers()}"
        )
    if effective_wire_api == "responses":
        from agentica.model.openai import OpenAIResponses
        model = OpenAIResponses(**params)
    else:
        model = create_provider(model_provider, **params)

    fields = type(model).__dataclass_fields__ if dataclasses.is_dataclass(model) else {}
    if enable_cache_control is not None and "enable_cache_control" in fields:
        model.enable_cache_control = enable_cache_control
    if effective_wire_api == "chat_completions" and cache_control_messages is not None and "cache_control_messages" in fields:
        model.cache_control_messages = cache_control_messages
    if cache_control_session_header and "cache_control_session_header" in fields:
        model.cache_control_session_header = cache_control_session_header
    if effective_wire_api == "chat_completions" and cache_keepalive is not None and "cache_keepalive" in fields:
        model.cache_keepalive = cache_keepalive

    # context_window is NOT an API param — set on the instance for budget /
    # compression / status display. Overrides the catalog auto-detected value.
    if context_window:
        try:
            model.context_window = context_window
        except (AttributeError, TypeError):
            logger.debug(f"Could not set context_window on {type(model).__name__}")

    return model


def _gateway_cron_job_runner(job: Any) -> dict:
    """Start one run of a cron job now; return where it will land.

    Called from ``CronTool``'s sync tool entrypoint, in a worker thread. The
    run is handed to the gateway's scheduler on its own loop and not waited
    for: it is a turn of the job's conversation, which may be the very
    conversation making this call — waiting would deadlock on it.
    """
    from .. import deps

    if deps.scheduler is None or deps.main_loop is None:
        return {"job_id": job.id, "status": "failed", "error": "Scheduler not ready"}

    async def _start() -> dict:
        return deps.scheduler.run_now(job)

    return asyncio.run_coroutine_threadsafe(_start(), deps.main_loop).result(timeout=10)


def get_cron_tools(owner: Optional[str] = None) -> List[Any]:
    """Load the cron tool, wired for real immediate runs.

    Returns a ``CronTool`` (not the bare ``cronjob`` function) with a
    ``job_runner`` so ``cronjob(action="run", ...)`` starts the job now in
    its conversation, instead of just marking it due for the next tick.

    ``owner`` is the signed-in account: create/list/edit stay inside that
    account's jobs. The CLI omits it (one user on the machine).
    """
    try:
        from agentica.tools.cron_tool import CronTool
        logger.debug("Loaded cron tool")
        return [CronTool(job_runner=_gateway_cron_job_runner, owner=owner)]
    except Exception as e:
        logger.warning(f"Failed to load cron tools: {e}")
        return []


def get_cron_instructions() -> str:
    """Return the system prompt instructions for cron job management."""
    return """
# Scheduled tasks

You can create and manage the user's scheduled tasks with the `cronjob` tool
(create / list / runs / update / pause / resume / remove / run). Each run is
delivered as an automatic message into a conversation and answered there,
where the user can read and follow up. When the user wants the result here,
use target='session'; for an independent recurring job, the default
target='new_once' keeps every run in one dedicated conversation.

Schedule formats: cron expression ("30 7 * * *"), interval ("30m", "every 2h"), or ISO datetime.
The user also manages these on the Automations page (left sidebar).
"""


def get_self_manage_tools() -> List[Any]:
    """Load the self-management tool (same one the CLI gives its agent).

    Lets the web agent inspect its own config.yaml/.env, know its own
    version, and adjust its own model/tuning — the "self-awareness"
    capability the CLI has always had. Gated by the same permission_mode
    tiers as write_file/apply_patch (hidden in "ask" mode, available in
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
    from .im_bot import IM_COMMANDS

    im_commands = "\n".join(f"  - `{cmd}` — {what}" for cmd, _, _, what in IM_COMMANDS)
    return f"""
# Self-Awareness

You are Agentica, running inside the Agentica Gateway on the user's own
machine. One session can be reached from several clients: the web page, the
desktop app, an IM channel (WeChat/Feishu/WeCom/Telegram/Discord/DingTalk/QQ),
and a terminal running `agentica --resume <session_id>`. They share this
session's transcript, so the user may switch between them mid-conversation.

## Who and where you are: `self_manage(action="show")`

Its `this_session` block is the authoritative answer to "which model / profile
are you", "what is this session's id", "where is the transcript / log / work
dir", and "which client is the user on" (`channel`: `wechat`, `feishu` … for
an IM chat, null for web / desktop / CLI). Call it rather than inferring from
the prompt or from config.yaml's active profile (a session can run on a
different profile than the default). When the user is on an IM channel and
asks what you can do or how to control you, answer with the IM commands below.

- `self_manage(action="show")` — this session + config.yaml profiles, settings
  (including whether the cron scheduler daemon is enabled) and .env (masked)
- `self_manage(action="set_config", key="...", value="...")` — edit a config.yaml profile field
- `self_manage(action="set_env", key="...", value="...")` — set/delete a .env variable
- `self_manage(action="check_upgrade")` — check for a newer Agentica release on PyPI
- `self_manage(action="install_skill", value="<git url or path>")` — install a skill

Model/profile edits take effect on the next agent rebuild, not mid-conversation.

## Commands the user can type

These are handled by the client before the message reaches you; you never
see them. When the user asks how to do one of these things, point them to
the command for the client they are on.

- IM channels:
{im_commands}
- Web / desktop chat box: `/goal`, `/queue`, `/compact`, `/permission`,
  `/model`, `/export`, plus `/<skill>` for user-invocable skills.

Terminal-only commands (`/cron daemon on`, `/upgrade`, `/config`, …) exist
only in the `agentica` CLI. Do not hand them to a user on the web or in IM.
If something needs deployment-level access you don't have (enabling the cron
daemon, restarting the gateway after an upgrade), say so plainly.
"""
