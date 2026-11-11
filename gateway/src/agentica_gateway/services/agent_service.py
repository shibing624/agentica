# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Agent service - wraps the agentica SDK.

Key design decisions:
- LRU cache for Agent instances (bounded by settings.agent_max_sessions)
- Per-session work_dir stored separately from global settings
- Fail fast on initialization errors (no silent mock mode)
- cancel_session(session_id) for precise stream cancellation
- Agent build timeout to guard against SDK hangs
- Uses DeepAgent (batteries-included) instead of manual Agent + builtin tools
- Per-session stream lock prevents concurrent streams on the same session
"""
import asyncio
import os
import queue
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Any, Dict, TYPE_CHECKING, Awaitable
import inspect

from agentica.compression.manager import apply_idle_compact
from agentica.compression.new_window import notes_path_for
from agentica.utils.log import logger
from agentica.utils.string import replace_invalid_utf8
from agentica import DeepAgent
from agentica.agent.config import ExperienceConfig, ToolConfig, WorkspaceMemoryConfig
from agentica.cron.scheduler import scheduled_job_guidance
from agentica.run.display import RunDisplayEventKind, classify_run_response
from agentica.run.response import AgentCancelledError
from agentica.run.config import RunConfig
from agentica.run.context import RunSource
from agentica.workspace import Workspace
from agentica.config.profiles import (
    apply_global_config,
    get_setting,
    get_profile,
    get_profiles,
    get_project_profile,
    get_active_profile_name,
    set_active_profile,
    provider_api_key_env,
)
from agentica.memory.session_log import SessionLog, iso_timestamp, is_session_unread
from agentica.skills import get_skill_registry, load_system_skills
from agentica.tools.background_processes import (
    BackgroundProcess,
    BackgroundProcessCompleted,
    BackgroundProcessRegistry,
    background_result_for_agent,
    read_log_tail,
)
from agentica.compression.manager import (
    compact_token_limit_of,
    parse_compact_token_limit,
    working_context_window,
)

from ..config import settings
from .media_understanding import media_understanding
from .model_factory import (
    create_model, get_cron_tools, get_cron_instructions,
    get_self_manage_tools, get_self_manage_instructions,
)
from agentica.agent.approvals import (
    SessionGrants,
    apply_path_grant_on_agent,
    make_approve,
    sync_grants_from_project,
)

from .response_formatter import extract_metrics, format_tool_call_args, format_tool_result
from .session_usage import turn_usage_payload, usage_payload
from . import chat_images, live_turn

if TYPE_CHECKING:
    from ..channels.base import InboundMedia


def _profile_model_config(name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """One config.yaml profile as the flat model config the agent is built from.

    Empty base_url / api_key stay empty so the provider's own defaults (and its
    ``*_API_KEY`` env var) apply, as they do for the gateway default. The
    auxiliary block inherits the main endpoint only on the same provider.
    """
    wire_api = profile.get("wire_api") or ""
    aux = profile.get("auxiliary_model")
    aux = aux if isinstance(aux, dict) else {}
    provider = profile["model_provider"]
    base_url = profile.get("base_url") or ""
    api_key = profile.get("api_key") or ""
    aux_provider = aux.get("model_provider") or provider
    same = aux_provider == provider
    aux_wire_api = aux.get("wire_api") or ""
    raw_fallbacks = profile.get("fallback_models")
    fallbacks: List[Dict[str, Any]] = []
    if isinstance(raw_fallbacks, list):
        for fb in raw_fallbacks:
            if isinstance(fb, dict) and fb.get("model_provider") and fb.get("model_name"):
                fallbacks.append(dict(fb))
    max_api_retry = profile.get("max_api_retry")
    return {
        "profile": name,
        "model_provider": provider,
        "model_name": profile["model_name"],
        "base_url": base_url,
        "api_key": api_key,
        "wire_api": wire_api,
        "reasoning": profile.get("reasoning") or "",
        "reasoning_effort": "" if wire_api == "responses" else (profile.get("reasoning_effort") or ""),
        "max_tokens": int(profile.get("max_tokens") or 0),
        "temperature": float(profile.get("temperature") or 0),
        "top_p": float(profile.get("top_p") or 0),
        "context_window": int(profile.get("context_window") or 0),
        "enable_cache_control": profile.get("enable_cache_control"),
        "cache_control_messages": profile.get("cache_control_messages"),
        "cache_control_session_header": profile.get("cache_control_session_header") or "",
        "cache_keepalive": profile.get("cache_keepalive"),
        "extra_body": dict(profile["extra_body"]) if isinstance(profile.get("extra_body"), dict) else None,
        "extra_headers": dict(profile["extra_headers"]) if isinstance(profile.get("extra_headers"), dict) else None,
        "default_headers": dict(profile["default_headers"]) if isinstance(profile.get("default_headers"), dict) else None,
        "fallback_models": fallbacks,
        "max_api_retry": int(max_api_retry) if max_api_retry is not None else 2,
        "auxiliary_model_provider": aux_provider if aux.get("model_name") else "",
        "auxiliary_model_name": aux.get("model_name") or "",
        "auxiliary_base_url": aux.get("base_url") or (base_url if same else ""),
        "auxiliary_api_key": aux.get("api_key") or (api_key if same else ""),
        "auxiliary_wire_api": aux_wire_api,
        "auxiliary_reasoning": aux.get("reasoning") or "",
        "auxiliary_reasoning_effort": "" if aux_wire_api == "responses" else (aux.get("reasoning_effort") or ""),
        "auxiliary_extra_body": dict(aux["extra_body"]) if isinstance(aux.get("extra_body"), dict) else None,
        "auxiliary_extra_headers": dict(aux["extra_headers"]) if isinstance(aux.get("extra_headers"), dict) else None,
    }

# Timeout in seconds for building a new Agent instance (guards against SDK hangs)
_AGENT_BUILD_TIMEOUT_S = 30

# Session-id prefix reserved for scheduled (cron) job runs. These sessions are
# never surfaced by list_sessions() (the chat sidebar) — cron execution
# history lives in the dedicated TaskRun store (agentica.cron.jobs), not the
# chat session log. See AgentService.run_cron().
CRON_SESSION_PREFIX = "scheduled_"

# IM channel values stored in SessionLog sidecar ``source`` so Web and Desktop
# list IM conversations alongside web chats and badge them by channel
# (WeChat / Feishu / WeCom / Telegram / DingTalk / QQ / Discord / Slack).
IM_SESSION_SOURCES = frozenset({
    "wechat",
    "feishu",
    "wecom",
    "telegram",
    "dingtalk",
    "qq",
    "discord",
    "slack",
})


def _infer_im_source_from_session_id(session_id: str) -> Optional[str]:
    """Extract the channel slug from legacy ``agent:<id>:<channel>:<chat_id>`` ids."""
    if not session_id or not session_id.startswith("agent:"):
        return None
    parts = session_id.split(":", 3)
    if len(parts) >= 3 and parts[2] in IM_SESSION_SOURCES:
        return parts[2]
    return None

_TITLE_PROMPT = (
    "Name this conversation for a sidebar list. Reply with ONLY the title: "
    "at most 8 words (or 16 CJK characters), in the user's language, no quotes, "
    "no trailing punctuation."
)

_SYSTEM_TEMP_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/var/folders", "/private/var/folders")


def is_system_temp_path(path: str) -> bool:
    """Whether ``path`` is inside the OS temp dir (``/tmp``, ``/var/folders``, ``$TMPDIR``).

    ``$AGENTICA_HOME`` is never temp, even when it lives under one (tests, a
    throwaway home): the web's own "temporary workspace" is
    ``$AGENTICA_HOME/tmp/web-chats``, a place the user chose to chat in.
    """
    if not path:
        return False
    try:
        real = os.path.realpath(os.path.expanduser(path))
        home = os.path.realpath(os.path.expanduser(os.getenv("AGENTICA_HOME", "~/.agentica")))
        roots = {os.path.realpath(r) for r in _SYSTEM_TEMP_ROOTS}
        roots.add(os.path.realpath(tempfile.gettempdir()))
    except (OSError, ValueError):
        return False
    if real == home or real.startswith(home + os.sep):
        return False
    return any(real == r or real.startswith(r + os.sep) for r in roots)

# Agent cache, run lock, work_dir and approval mode are partitioned by
# owner. ``session_id`` is client-supplied; a bare key lets one account
# evict another's cached Agent / lock / live run.
_OWNER_SEP = "\x1f"


def _cache_key_is_pinned(key: str) -> bool:
    """True when this cache entry has a LiveTurn waiting on an approval."""
    owner, sep, session_id = key.partition(_OWNER_SEP)
    if not sep:
        owner, session_id = settings.default_user_id, key
    turn = live_turn.active(session_id, owner)
    return turn is not None and turn.approvals.size > 0


async def dispatch_stream_chunk(
    chunk,
    on_content: Optional[Callable[[str], Any]] = None,
    on_tool_call: Optional[Callable[[str, dict], Any]] = None,
    on_tool_result: Optional[Callable[..., Any]] = None,
    on_thinking: Optional[Callable[[str], Any]] = None,
) -> None:
    """Fan a ``run_stream`` chunk out to the same callbacks chat uses."""
    if chunk is None:
        return

    async def _emit(fn, *args):
        if fn is None:
            return
        maybe = fn(*args)
        if inspect.isawaitable(maybe):
            await maybe

    display_event = classify_run_response(chunk)

    if display_event.kind == RunDisplayEventKind.TOOL_STARTED:
        tool_info = chunk.tool_call
        if tool_info:
            tool_name = tool_info.tool_name or "unknown"
            display_args = format_tool_call_args(tool_name, tool_info.tool_args)
            await _emit(
                on_tool_call, tool_name, display_args, tool_info.tool_call_id or "",
            )
        return

    if display_event.kind == RunDisplayEventKind.TOOL_COMPLETED:
        if chunk.tool_call and on_tool_result:
            t_name, result_str, extra = format_tool_result(chunk.tool_call)
            await _emit(on_tool_result, t_name, result_str, extra)
        return

    if display_event.kind in (
        RunDisplayEventKind.METADATA_SKIP,
        RunDisplayEventKind.TELEMETRY_ONLY,
    ):
        return

    if display_event.kind == RunDisplayEventKind.CONTENT_DELTA:
        if chunk.reasoning_content:
            await _emit(on_thinking, chunk.reasoning_content)
        if chunk.content:
            await _emit(on_content, chunk.content)


def goal_event_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Web goal bar: tokens only (empty/unlimited budget shows the running total)."""
    tokens_used = int(payload.get("tokens_used") or 0)
    token_budget = payload.get("token_budget")
    if token_budget is not None:
        progress = f"tokens {tokens_used:,}/{int(token_budget):,}"
    else:
        progress = f"tokens {tokens_used:,}"
    return {
        "status": payload.get("status") or "active",
        "objective": payload.get("objective") or "",
        "progress": progress,
        "tokens_used": tokens_used,
        "token_budget": token_budget,
        "message": payload.get("message") or "",
    }

# Web sessions default to "auto" approval mode. Baked in once as a standing
# agent instruction (not per-message) so it stays in the cache-friendly
# static prompt zone. Tools are always in schema; the mode only decides
# whether execute waits for a human.
_APPROVAL_MODE_INSTRUCTION = (
    "This session's approval mode can pause a tool call until the user "
    "confirms it. In \"ask\" mode, file reads (including outside the work "
    "directory), read-only shell commands (including wrappers such as "
    "cd && git diff | head), web_search/fetch_url, memory, task/delegate, "
    "skills, and other builtins run immediately; write_file / apply_patch "
    "and mutating execute wait for approval. In \"auto\" mode, reads "
    "(including outside the work directory), in-workspace writes, every "
    "execute, network tools, and every builtin/skill/third-party tool "
    "run; only file writes outside the work directory or to sensitive "
    "paths wait. "
    "\"allow-all\" never asks. If a tool result is "
    "\"Tool call denied by user.\", do not retry the same call — tell the "
    "user what you needed and wait for guidance or a mode change."
)


def _log_report_failure(future) -> None:
    """A background report that failed is logged, not lost in the loop."""
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:
        logger.warning(f"Background report was not delivered: {exc!r}")


@dataclass
class ChatResult:
    """Chat response from the agent."""
    content: str
    tool_calls: int = 0
    session_id: str = ""
    user_id: str = ""
    tools_used: List[str] = field(default_factory=list)
    reasoning: str = ""
    metrics: Optional[Dict[str, Any]] = None
    # User-facing one-liners about how inbound media was handled (non-base
    # model used, media skipped, …); the channel layer prefixes these to the
    # reply so IM users see them.
    media_notes: List[str] = field(default_factory=list)
    # Session-level /usage snapshot (context breakdown + billed totals).
    usage: Optional[Dict[str, Any]] = None
    # This run's CostTracker split (input / cache read / hit / output / cost).
    turn_usage: Optional[Dict[str, Any]] = None


class LRUAgentCache:
    """Thread-unsafe but asyncio-safe LRU cache for DeepAgent instances."""

    def __init__(self, max_size: int = 50):
        self._cache: OrderedDict[str, DeepAgent] = OrderedDict()
        self.max_size = max_size

    def get(self, session_id: str) -> Optional[DeepAgent]:
        if session_id not in self._cache:
            return None
        self._cache.move_to_end(session_id)
        return self._cache[session_id]

    def put(self, session_id: str, agent: DeepAgent) -> None:
        if session_id in self._cache:
            self._cache.move_to_end(session_id)
            self._cache[session_id] = agent
            return
        self._cache[session_id] = agent
        while len(self._cache) > self.max_size:
            evicted = False
            for key in list(self._cache.keys()):
                if _cache_key_is_pinned(key):
                    continue
                del self._cache[key]
                logger.debug(f"LRU evicted agent for session: {key}")
                evicted = True
                break
            if not evicted:
                break

    def delete(self, session_id: str) -> bool:
        if session_id in self._cache:
            del self._cache[session_id]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def contains(self, session_id: str) -> bool:
        """Membership without the LRU touch ``get`` performs.

        Liveness probes (``AgentService.has_cached_session``) must not reorder
        the cache: a polling caller would otherwise keep whatever it asks about
        alive and evict the session the user is actually talking to.
        """
        return session_id in self._cache

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)


# The notes go out in a JSON response and onto the chat page; a runaway file
# should not turn one /compact into megabytes.
_COMPACT_NOTES_LIMIT = 200_000


def _read_compact_notes(agent: Any) -> tuple[str, str]:
    """``(<session>.notes.md text, path shown with ~)``; empty strings if none."""
    path = notes_path_for(getattr(agent, "_session_log", None))
    if not path:
        return "", ""
    p = Path(path)
    shown = str(p)
    home = str(Path.home())
    if shown == home or shown.startswith(home + os.sep):
        shown = "~" + shown[len(home):]
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return "", shown
    return text[:_COMPACT_NOTES_LIMIT], shown


class AgentService:
    """Agent service wrapping the agentica SDK.

    Features:
    - Workspace config layer (AGENTS.md, MEMORY.md, etc.)
    - Session history management (per session_id)
    - LRU-bounded Agent instance cache (evicts on overflow)
    - Per-session working directory
    - Per-session run lock (prevents concurrent chat/stream on same session)
    - Scheduler tool integration
    """

    def __init__(
        self,
        workspace_path: Optional[str] = None,
        extra_tools: Optional[List[Any]] = None,
        extra_instructions: Optional[List[str]] = None,
    ):
        self.workspace_path = Path(workspace_path or settings.workspace_path).expanduser()
        self.extra_tools = extra_tools or []
        self.extra_instructions = extra_instructions or []

        self._cache = LRUAgentCache(max_size=settings.agent_max_sessions)
        # Per-session work_dir overrides; falls back to settings.base_dir
        self._session_work_dirs: Dict[str, str] = {}
        self._session_approval_modes: Dict[str, str] = {}
        self._session_grants: Dict[str, SessionGrants] = {}
        self._session_preview_cache: OrderedDict[str, tuple[tuple[int, int], Dict[str, Any]]] = OrderedDict()
        self._session_preview_lock = threading.Lock()
        # Sessions already handed to the background titler, and its live tasks.
        self._titled: set[tuple[str, str]] = set()
        self._title_tasks: set[asyncio.Task] = set()
        # Per-session run locks: prevents concurrent runs (chat or stream) on the same session.
        # The underlying Agent instance is NOT thread-safe for concurrent reuse.
        self._session_locks: Dict[str, asyncio.Lock] = {}
        # Cached agents built under a model/profile that has since changed but
        # were mid-run then: kept so cancel/steer still reach the run, rebuilt
        # on the session's next request.
        self._stale_agents: set[str] = set()
        self._workspace: Optional[Workspace] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._session_ask_callbacks: Dict[str, Callable] = {}
        # Per session, outside the agent: a model switch or LRU eviction
        # rebuilds the agent, and its running commands must stay reachable.
        self._session_background: Dict[str, BackgroundProcessRegistry] = {}
        # Runs a finished command's report as a new automated turn when the
        # session is idle (set by main: the IM bot for an IM-bound session,
        # else a live web turn). Without one the report waits for `wait`.
        self.background_report_handler: Optional[
            Callable[[str, str, str], Awaitable[None]]
        ] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ============== Model config (single source of truth: `settings`) ==============
    # These proxy directly to the gateway's global `settings` singleton instead
    # of keeping a duplicate copy, so routes only ever need to write to one
    # place (settings.xxx) and every reader (here, routes/settings.py, etc.)
    # sees the same value with no risk of drift.

    @property
    def model_provider(self) -> str:
        return settings.model_provider

    @model_provider.setter
    def model_provider(self, value: str) -> None:
        settings.model_provider = value

    @property
    def model_name(self) -> str:
        return settings.model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        settings.model_name = value

    @property
    def model_base_url(self) -> str:
        return settings.model_base_url

    @property
    def model_api_key(self) -> str:
        return settings.model_api_key

    @property
    def model_wire_api(self) -> str:
        return settings.model_wire_api

    @property
    def model_reasoning(self) -> str:
        return settings.model_reasoning

    @property
    def model_reasoning_effort(self) -> str:
        return settings.model_reasoning_effort

    @property
    def max_tokens(self) -> int:
        return settings.max_tokens

    @property
    def temperature(self) -> float:
        return settings.temperature

    @property
    def top_p(self) -> float:
        return settings.top_p

    @property
    def context_window(self) -> int:
        return settings.context_window

    @property
    def auxiliary_model_provider(self) -> str:
        return settings.auxiliary_model_provider

    @property
    def auxiliary_model_name(self) -> str:
        return settings.auxiliary_model_name

    @property
    def auxiliary_base_url(self) -> str:
        return settings.auxiliary_base_url

    @property
    def auxiliary_api_key(self) -> str:
        return settings.auxiliary_api_key

    @property
    def auxiliary_wire_api(self) -> str:
        return settings.auxiliary_wire_api

    @property
    def auxiliary_reasoning(self) -> str:
        return settings.auxiliary_reasoning

    @property
    def auxiliary_reasoning_effort(self) -> str:
        return settings.auxiliary_reasoning_effort

    def _default_model_config(self) -> Dict[str, Any]:
        """The gateway default (``settings``): config.yaml's active profile, or
        the env vars when no profile is configured."""
        active_name = get_active_profile_name()
        active = get_profile(active_name) if active_name else {}
        aux = active.get("auxiliary_model") if isinstance(active.get("auxiliary_model"), dict) else {}
        matches_active = (
            bool(active)
            and active.get("model_provider") == self.model_provider
            and active.get("model_name") == self.model_name
        )
        raw_fallbacks = active.get("fallback_models") if matches_active else None
        fallbacks: List[Dict[str, Any]] = []
        if isinstance(raw_fallbacks, list):
            for fb in raw_fallbacks:
                if isinstance(fb, dict) and fb.get("model_provider") and fb.get("model_name"):
                    fallbacks.append(dict(fb))
        max_api_retry = active.get("max_api_retry") if matches_active else None
        return {
            "profile": active_name,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "base_url": self.model_base_url,
            "api_key": self.model_api_key,
            "wire_api": self.model_wire_api,
            "reasoning": self.model_reasoning,
            "reasoning_effort": self.model_reasoning_effort,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "enable_cache_control": active.get("enable_cache_control") if matches_active else None,
            "cache_control_messages": active.get("cache_control_messages") if matches_active else None,
            "cache_control_session_header": (active.get("cache_control_session_header") or "") if matches_active else "",
            "cache_keepalive": active.get("cache_keepalive") if matches_active else None,
            "extra_body": (dict(active["extra_body"]) if isinstance(active.get("extra_body"), dict) else None) if matches_active else None,
            "extra_headers": (dict(active["extra_headers"]) if isinstance(active.get("extra_headers"), dict) else None) if matches_active else None,
            "default_headers": (dict(active["default_headers"]) if isinstance(active.get("default_headers"), dict) else None) if matches_active else None,
            "fallback_models": fallbacks,
            "max_api_retry": int(max_api_retry) if max_api_retry is not None else 2,
            "auxiliary_model_provider": self.auxiliary_model_provider,
            "auxiliary_model_name": self.auxiliary_model_name,
            "auxiliary_base_url": self.auxiliary_base_url,
            "auxiliary_api_key": self.auxiliary_api_key,
            "auxiliary_wire_api": self.auxiliary_wire_api,
            "auxiliary_reasoning": self.auxiliary_reasoning,
            "auxiliary_reasoning_effort": self.auxiliary_reasoning_effort,
            "auxiliary_extra_body": (dict(aux["extra_body"]) if isinstance(aux.get("extra_body"), dict) else None) if matches_active else None,
            "auxiliary_extra_headers": (dict(aux["extra_headers"]) if isinstance(aux.get("extra_headers"), dict) else None) if matches_active else None,
        }

    def _session_profile(self, session_id: str, owner: Optional[str] = None) -> tuple[str, str]:
        """``(profile, source)`` a session runs on; ``("", "default")`` is the
        gateway default.

        A session the user picked a model for keeps it (``<id>.meta.json``
        ``profile_name``, the same sidecar field the CLI resumes from). A
        session without a choice follows its project's default
        (``project.json`` ``active_profile``), then config.yaml's. A name that
        is no longer in config.yaml falls through to the next layer.
        """
        profiles = get_profiles()
        picked = self.session_log_for(session_id, owner).get_profile_name()
        if picked and picked in profiles:
            return picked, "session"
        project = get_project_profile(self.get_session_work_dir(session_id, owner))
        if project and project in profiles:
            return project, "project"
        return "", "default"

    def session_profile_name(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._session_profile(session_id, owner)[0]

    def session_profile_source(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._session_profile(session_id, owner)[1]

    def session_model_config(self, session_id: str, owner: Optional[str] = None) -> Dict[str, Any]:
        """The flat model config a session's agent is built from."""
        name = self.session_profile_name(session_id, owner)
        if name:
            profile = get_profile(name)
            if profile.get("model_provider") and profile.get("model_name"):
                return _profile_model_config(name, profile)
        return self._default_model_config()

    async def set_session_profile(self, session_id: str, name: str, owner: Optional[str] = None) -> Dict[str, Any]:
        """Pin a session to a profile; other sessions keep theirs.

        A run in flight finishes on its current model; the session rebuilds on
        its next request.
        """
        self.session_log_for(session_id, owner).set_profile(name, "session")
        key = self._sk(session_id, owner)
        lock = self._session_locks.get(key)
        if lock is not None and lock.locked():
            self._stale_agents.add(key)
        else:
            self._cache.delete(key)
        return self.session_model_config(session_id, owner)

    def _session_provider(self, session_id: str, owner: Optional[str] = None) -> str:
        """Provider slug of the session's model, for the ``provider/model`` label."""
        if session_id.startswith(CRON_SESSION_PREFIX) and self.auxiliary_model_name:
            return self.auxiliary_model_provider or self.model_provider
        return self.session_model_config(session_id, owner)["model_provider"]

    @staticmethod
    def _sibling_model(cfg: Dict[str, Any], prefix: str) -> Optional[Any]:
        """Build a sibling (auxiliary) model if a model name is configured.

        Returns None when no sibling name is set — DeepAgent will reuse
        the main model. The auxiliary model also serves as the task subagent
        model (CLI unified them).
        """
        sibling_name = cfg[f"{prefix}_model_name"]
        if not sibling_name:
            return None
        return create_model(
            cfg[f"{prefix}_model_provider"] or cfg["model_provider"],
            sibling_name,
            base_url=cfg[f"{prefix}_base_url"] or None,
            api_key=cfg[f"{prefix}_api_key"] or None,
            wire_api=cfg[f"{prefix}_wire_api"],
            max_tokens=cfg.get("max_tokens") or 0,
            temperature=cfg.get("temperature") or 0.0,
            top_p=cfg.get("top_p") or 0.0,
            context_window=cfg.get("context_window") or 0,
            reasoning=cfg[f"{prefix}_reasoning"],
            reasoning_effort=cfg[f"{prefix}_reasoning_effort"],
            thinking=settings.model_thinking,
            enable_cache_control=cfg.get("enable_cache_control"),
            cache_control_messages=cfg.get("cache_control_messages"),
            cache_control_session_header=cfg.get("cache_control_session_header") or None,
            cache_keepalive=cfg.get("cache_keepalive"),
            extra_body=cfg.get(f"{prefix}_extra_body"),
            extra_headers=cfg.get(f"{prefix}_extra_headers"),
        )

    @staticmethod
    def _fallback_models(cfg: Dict[str, Any]) -> List[Any]:
        """Build fallback Model instances from resolved profile fallback entries."""
        models: List[Any] = []
        for fb in cfg.get("fallback_models") or []:
            if not isinstance(fb, dict) or not fb.get("model_provider") or not fb.get("model_name"):
                continue
            models.append(
                create_model(
                    fb["model_provider"],
                    fb["model_name"],
                    base_url=fb.get("base_url") or None,
                    api_key=fb.get("api_key") or None,
                    wire_api=fb.get("wire_api") or "",
                    reasoning=fb.get("reasoning") or "",
                    reasoning_effort="" if fb.get("wire_api") == "responses" else (fb.get("reasoning_effort") or ""),
                    extra_body=fb.get("extra_body") if isinstance(fb.get("extra_body"), dict) else None,
                    extra_headers=fb.get("extra_headers") if isinstance(fb.get("extra_headers"), dict) else None,
                    default_headers=fb.get("default_headers") if isinstance(fb.get("default_headers"), dict) else None,
                )
            )
        return models

    def _build_sibling_model(self, prefix: str) -> Optional[Any]:
        return self._sibling_model(self._default_model_config(), prefix)

    def side_task_model(self) -> Any:
        """The model for gateway side tasks (titles, commit messages, prompt
        polish): the auxiliary model when configured, else the main model."""
        aux = self._build_sibling_model("auxiliary")
        if aux is not None:
            return aux
        return create_model(
            self.model_provider,
            self.model_name,
            base_url=self.model_base_url or None,
            api_key=self.model_api_key or None,
            wire_api=self.model_wire_api,
            reasoning=self.model_reasoning,
            reasoning_effort=self.model_reasoning_effort,
        )

    async def side_task(self, system: str, user: str) -> str:
        """One system+user call on :meth:`side_task_model`; the reply text."""
        from agentica.model.message import Message

        model = self.side_task_model()
        response = await model.response([
            Message(role="system", content=system),
            Message(role="user", content=user),
        ])
        return ((response.content if response else "") or "").strip()

    # ============== Initialization ==============

    async def _ensure_initialized(self) -> None:
        """Ensure the workspace is initialized (idempotent, Lock-protected)."""
        self._loop = asyncio.get_running_loop()
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._do_initialize)

    def _do_initialize(self) -> None:
        """Initialize the shared Workspace (sync, runs in thread).

        Raises RuntimeError on failure — callers must handle this explicitly.
        No silent mock mode.
        """
        try:
            self._workspace = Workspace(self.workspace_path)
            if not self._workspace.exists():
                self._workspace.initialize()
                logger.info(f"Workspace initialized at {self.workspace_path}")

            self._initialized = True
            logger.info("AgentService initialized successfully")
            logger.info(f"Model: {self.model_provider}/{self.model_name}")
            logger.info(f"Workspace: {self.workspace_path}")

        except Exception as e:
            logger.error(
                f"AgentService initialization failed: {e}\n"
                f"Check your API key, model provider, and agentica version."
            )
            raise RuntimeError(f"AgentService init failed: {e}") from e

    def _build_agent(self, session_id: str, owner: Optional[str] = None) -> DeepAgent:
        """Build a new DeepAgent instance (sync, runs in thread).

        DeepAgent auto-includes: builtin tools, skills, agentic prompt,
        two-layer compression (eviction then LLM summarisation),
        workspace memory (MEMORY.md index +
        auto_extract_memory), experience capture (tool
        errors / user corrections / success patterns), memory tools.

        auxiliary_model / task_model default to the main model (DeepAgent
        default). Pass AGENTICA_AUXILIARY_* / AGENTICA_TASK_* env vars (or build
        AgentService with the matching kwargs) to route them to a
        different provider / model / base_url / api_key.

        Scheduled (cron) jobs are unattended, don't need flagship-model
        quality, and can run frequently — so when an auxiliary model is
        configured, cron sessions use it as their *main* model to cut cost.
        Interactive chat sessions are unaffected.

        An interactive session builds from its own model config
        (:meth:`session_model_config`): the profile picked for it, else its
        project's, else the gateway default.
        """
        if session_id.startswith(CRON_SESSION_PREFIX) and self.auxiliary_model_name:
            cfg = self._default_model_config()
            model = self._build_sibling_model("auxiliary")
            auxiliary_model = None
            task_model = None
            fallback_models: List[Any] = []
            max_api_retry = 1
        else:
            cfg = self.session_model_config(session_id, owner)
            model = create_model(
                cfg["model_provider"],
                cfg["model_name"],
                base_url=cfg["base_url"] or None,
                api_key=cfg["api_key"] or None,
                wire_api=cfg["wire_api"],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                context_window=cfg["context_window"],
                reasoning=cfg["reasoning"],
                reasoning_effort=cfg["reasoning_effort"],
                thinking="enabled",
                enable_cache_control=cfg.get("enable_cache_control"),
                cache_control_messages=cfg.get("cache_control_messages"),
                cache_control_session_header=cfg.get("cache_control_session_header") or None,
                cache_keepalive=cfg.get("cache_keepalive"),
                extra_body=cfg.get("extra_body"),
                extra_headers=cfg.get("extra_headers"),
                default_headers=cfg.get("default_headers"),
            )
            auxiliary_model = self._sibling_model(cfg, "auxiliary")
            # The auxiliary model also serves as the task subagent model (CLI
            # unified them — no separate task_model config anymore).
            task_model = auxiliary_model
            fallback_models = self._fallback_models(cfg)
            max_api_retry = int(cfg.get("max_api_retry") or 2)
        # Per-session project dir (set via routes/chat.py::_apply_session_work_dir
        # from the frontend's session.dir) takes precedence — falls back to the
        # global settings.base_dir only when the session has none set.
        work_dir = self.get_session_work_dir(session_id, owner)

        # Extra tools: user-provided + cron + self-management (self-awareness)
        extra = list(self.extra_tools)
        cron_tools = get_cron_tools(owner=self._owner(owner))
        extra.extend(cron_tools)
        self_manage_tools = get_self_manage_tools()
        extra.extend(self_manage_tools)

        ask_cb = self._session_ask_callbacks.get(self._sk(session_id, owner))
        if ask_cb is not None and not session_id.startswith(CRON_SESSION_PREFIX):
            from agentica.tools.ask_user_question_tool import AskUserQuestionTool

            extra.append(AskUserQuestionTool(input_callback=ask_cb))

        instructions = list(self.extra_instructions) if self.extra_instructions else None
        if cron_tools:
            if instructions is None:
                instructions = []
            instructions.append(get_cron_instructions())
        if self_manage_tools:
            if instructions is None:
                instructions = []
            instructions.append(get_self_manage_instructions())
        if instructions is None:
            instructions = []
        instructions.append(_APPROVAL_MODE_INSTRUCTION)

        permission_mode = self.get_session_approval_mode(session_id, owner)
        enable_evict = get_setting("enable_evict", True)
        enable_auto_compact = get_setting("enable_auto_compact", True)
        profile = get_profile(self.session_profile_name(session_id, owner) or get_active_profile_name()) or {}
        compact_token_limit = parse_compact_token_limit(profile.get("compact_token_limit"))
        if compact_token_limit is None:
            compact_token_limit = parse_compact_token_limit(
                get_setting("compact_token_limit", None)
            )
        auto_extract = self._auto_extract_for(owner)
        load_system_skills()
        agent = DeepAgent(
            session_id=session_id,
            model=model,
            auxiliary_model=auxiliary_model,
            task_model=task_model,
            session_profile=cfg.get("profile") or None,
            fallback_models=fallback_models,
            max_api_retry=max_api_retry,
            description=(
                "You are DeepAgent, an AI coding agent served over the Agentica "
                "gateway (web UI and chat channels). You help users with software "
                "engineering and general tasks through a conversational interface."
            ),
            tools=extra if extra else None,
            workspace=self._workspace,
            work_dir=work_dir,
            user_id=self._owner(owner),
            num_history_turns=settings.num_history_turns,
            instructions=instructions,
            debug=settings.debug,
            # memory, skills, experience capture, workspace memory all on by
            # DeepAgent default — no explicit overrides needed. ask_user_question
            # is intentionally OFF: the gateway is headless (web/chat channels)
            # with no stdin, so the tool would block on a bare input() call.
            include_ask_user_question=False,
            # Scheduled runs have nobody to see or stop a detached command.
            background_process_registry=(
                None if session_id.startswith(CRON_SESSION_PREFIX)
                else self._background_registry(session_id, owner)
            ),
            permission_mode=permission_mode,
            tool_config=ToolConfig(
                auto_load_mcp=True,
                permission_mode=permission_mode,
                enable_evict=bool(enable_evict),
                enable_auto_compact=bool(enable_auto_compact),
                compact_token_limit=compact_token_limit,
            ),
            long_term_memory_config=WorkspaceMemoryConfig(
                auto_archive=True,
                auto_extract_memory=auto_extract,
                extract_every_n_turns=10,
                extract_min_seconds_between=60,
                load_workspace_context=True,
                load_workspace_memory=True,
                recall_memories=True,
                recall_sessions=True,
                track_intents=True,
                max_memory_entries=5,
            ),
            experience_config=ExperienceConfig(
                capture_tool_errors=True,
                capture_user_corrections=True,
                capture_success_patterns=False,
                judge_every_n_turns=10,
                judge_min_seconds_between=60,
            ),
        )

        agent.approve = self._make_session_approve(session_id, owner)
        self._restore_path_grants(agent, session_id, owner)
        if (
            agent._session_log is not None
            and agent._session_log.exists()
            and len(agent.working_memory.runs) == 0
            and len(agent.working_memory.messages) == 0
        ):
            if not self.is_session_active(session_id, owner):
                agent._session_log.seal_incomplete_turn()
            resumed_messages = agent._session_log.load(
                model=model.id if model is not None else None
            )
            if resumed_messages:
                agent.working_memory.hydrate_runs_from_history(resumed_messages)

        tool_count = len(agent.tools) if agent.tools else 0
        logger.info(
            f"DeepAgent built: {tool_count} tools "
            f"(extra={len(extra)}, cron={len(cron_tools)}, self_manage={len(self_manage_tools)})"
        )
        return agent

    async def _get_agent(self, session_id: str, owner: Optional[str] = None) -> DeepAgent:
        """Return the cached Agent for a session, creating one if absent.

        Raises RuntimeError if the agent cannot be built (e.g. SDK error).
        Times out after _AGENT_BUILD_TIMEOUT_S seconds.

        A cached agent is only reused for the owner it was built for. The cache
        is keyed by ``(owner, session_id)``; ``session_id`` comes from the
        browser, so a bare key would let one account run against (or evict)
        another's Agent.
        """
        key = self._sk(session_id, owner)
        agent = self._cache.get(key)
        if agent is not None and agent.user_id == self._owner(owner):
            if key not in self._stale_agents or self.is_session_active(session_id, owner):
                return agent
        self._stale_agents.discard(key)

        try:
            agent = await asyncio.wait_for(
                asyncio.to_thread(self._build_agent, session_id, owner),
                timeout=_AGENT_BUILD_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Agent build timed out after {_AGENT_BUILD_TIMEOUT_S}s "
                f"for session {session_id}. Check MCP server connectivity."
            )

        self._cache.put(key, agent)
        logger.info(f"Agent created for session: {session_id} (cache size: {len(self._cache)})")
        return agent

    def _get_session_lock(self, session_id: str, owner: Optional[str] = None) -> asyncio.Lock:
        """Return (or create) the per-session run lock.

        Both chat() and chat_stream() acquire this lock to prevent concurrent
        runs on the same Agent instance, which is not thread-safe.
        """
        key = self._sk(session_id, owner)
        if key not in self._session_locks:
            self._session_locks[key] = asyncio.Lock()
        return self._session_locks[key]

    def has_cached_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Whether a built Agent is still cached for this session.

        ``owner=None`` (peer liveness probe) matches any account holding
        this session id. Named lookups stay inside that account's partition.
        """
        return any(self._cache.contains(k) for k in self._map_keys(session_id, owner))

    def is_session_active(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Whether a run is in flight on this session (its lock is held)."""
        for key in self._map_keys(session_id, owner):
            lock = self._session_locks.get(key)
            if lock is not None and lock.locked():
                return True
        return False

    def set_session_ask_callback(
        self,
        session_id: str,
        callback: Optional[Callable[[str, Optional[List[str]]], str]],
        owner: Optional[str] = None,
    ) -> None:
        """Register or clear the ``ask_user_question`` callback for a session."""
        key = self._sk(session_id, owner)
        if callback is None:
            self._session_ask_callbacks.pop(key, None)
        else:
            self._session_ask_callbacks[key] = callback
            cached = self._cache.get(key)
            if cached is not None:
                from agentica.tools.ask_user_question_tool import AskUserQuestionTool

                found = False
                for t in cached.tools or []:
                    if isinstance(t, AskUserQuestionTool):
                        t.input_callback = callback
                        found = True
                        break
                if not found:
                    self._stale_agents.add(key)

    # Default approval mode for web sessions: "auto" — file edits and commands
    # are allowed (writes restricted to work_dir). "ask" is opt-in via the
    # approval selector next to the send button. See agentica.agent.permissions
    # for the exact semantics of each of the 3 tiers ("ask"/"auto"/"allow-all").
    _DEFAULT_APPROVAL_MODE = "auto"

    def set_session_approval_mode(
        self, session_id: str, mode: str, owner: Optional[str] = None,
    ) -> None:
        """Persist the selected approval mode for a session.

        If an Agent is already cached for this session, its permission mode
        is switched in place (no rebuild) via ``Agent.set_permission_mode``.
        """
        from agentica.agent.permissions import PERMISSION_MODES

        normalized = (mode or self._DEFAULT_APPROVAL_MODE).strip().lower()
        if normalized not in PERMISSION_MODES:
            normalized = self._DEFAULT_APPROVAL_MODE
        self._session_approval_modes[self._sk(session_id, owner)] = normalized

        cached_agent = self._cache.get(self._sk(session_id, owner))
        if cached_agent is not None:
            cached_agent.set_permission_mode(normalized)

    def get_session_approval_mode(self, session_id: str, owner: Optional[str] = None) -> str:
        return self._session_approval_modes.get(
            self._sk(session_id, owner), self._DEFAULT_APPROVAL_MODE,
        )

    def _grants_for(self, session_id: str, owner: Optional[str] = None) -> SessionGrants:
        key = self._sk(session_id, owner)
        grants = self._session_grants.get(key)
        if grants is None:
            grants = SessionGrants()
            self._session_grants[key] = grants
            sync_grants_from_project(
                grants,
                work_dir=self.get_session_work_dir(session_id, owner),
                user_id=self._owner(owner),
            )
        return grants

    def _restore_path_grants(
        self, agent: DeepAgent, session_id: str, owner: Optional[str] = None,
    ) -> None:
        """Re-apply in-session path grants after an Agent rebuild."""
        grants = self._grants_for(session_id, owner)
        for path in grants.path_exact:
            apply_path_grant_on_agent(agent, path, prefix=False)
        for path in grants.path_prefixes:
            apply_path_grant_on_agent(agent, path, prefix=True)

    def _make_session_approve(self, session_id: str, owner: Optional[str] = None):
        """``Agent.approve``: look up the current LiveTurn on every wait.

        No LiveTurn (IM, cron, non-streaming POST /chat) is an immediate deny.
        Subscriber count is not a gate — a refresh with 0 SSE clients still parks.
        """
        owner_id = self._owner(owner)

        def get_registry():
            turn = live_turn.active(session_id, owner_id)
            return None if turn is None else turn.approvals

        def publish(pending):
            turn = live_turn.active(session_id, owner_id)
            if turn is None:
                return
            turn.publish(live_turn.approval_event(pending))

        def apply_path_grant(path: str, prefix: bool) -> None:
            agent = self._cache.get(self._sk(session_id, owner))
            if agent is not None:
                apply_path_grant_on_agent(agent, path, prefix=prefix)

        return make_approve(
            get_mode=lambda: self.get_session_approval_mode(session_id, owner),
            get_grants=lambda: self._grants_for(session_id, owner),
            get_registry=get_registry,
            get_work_dir=lambda: self.get_session_work_dir(session_id, owner),
            publish=publish,
            apply_path_grant=apply_path_grant,
            get_user_id=lambda: owner_id,
        )

    def _run_config_for_session(
        self,
        session_id: str,
        source: RunSource,
        *,
        channel: Optional[str] = None,
        stream_intermediate_steps: bool = False,
        automated: bool = False,
    ) -> RunConfig:
        # Tool permission enforcement lives on the Agent itself now
        # (tool_config.permission_mode + sandbox_config, kept in sync with
        # the session's approval mode by set_session_approval_mode above).
        return RunConfig(
            stream_intermediate_steps=stream_intermediate_steps, source=source, channel=channel,
            automated_input=automated,
        )


    # ============== Public API ==============

    async def session_usage(
        self,
        session_id: str,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """CLI ``/usage`` for one web session: occupancy + session billing.

        Builds the Agent if it is not cached yet, because the window
        breakdown is measured from the live prompt (system / skills / tools
        / history), not from the last SSE ``done`` event.
        """
        await self._ensure_initialized()
        agent = await self._get_agent(session_id, owner)
        payload = await usage_payload(agent, model_provider=self._session_provider(session_id, owner))
        # Timing lives only in the log: the cached Agent forgets it on restart,
        # and these are the same numbers the Trace page shows.
        log = self.session_log_for(session_id, owner)
        if log.path.exists():
            totals = (await asyncio.to_thread(log.analyze, lite=True))["totals"]
            tokens = totals["tokens"]
            if not payload["api_calls"] and totals["requests"]:
                payload["api_calls"] = totals["requests"]
                payload["input_tokens"] = tokens["prompt"]
                payload["output_tokens"] = tokens["output"]
                payload["cache_read_tokens"] = tokens["cacheRead"]
                payload["cache_write_tokens"] = tokens["cacheWrite"]
                payload["cache_hit_percent"] = tokens["cacheHitPercent"]
                if totals["costUsd"] is not None:
                    payload["cost_usd"] = round(float(totals["costUsd"]), 6)
            payload["stats"] = {
                "rounds": totals["rounds"],
                "steps": totals["requests"],
                "llm_ms": totals["llmMs"],
                "tool_ms": totals["toolMs"],
                "ttft_ms": totals["ttftMs"],
                "tps": totals["tps"],
                "uncached_input_tokens": tokens["input"],
                "cache_read_tokens": tokens["cacheRead"],
                "cache_write_tokens": tokens["cacheWrite"],
                "output_tokens": tokens["output"],
                "cache_hit_percent": tokens["cacheHitPercent"],
                "cost_usd": totals["costUsd"],
            }
        return payload

    def get_context_window(
        self, session_id: Optional[str] = None, owner: Optional[str] = None,
    ) -> int:
        """Return the context window size for the model used by a session.

        When ``session_id`` is omitted, returns the context window of an
        arbitrary cached agent (useful for a general status check before any
        specific session is known). Falls back to 128000 if no matching
        agent is cached yet (``context_window`` is a declared field on every
        Model, defaulting to 128000, so no per-call fallback is needed once
        an agent exists).
        """
        agent = None
        if session_id is None:
            keys = self._cache.keys()
            agent = self._cache.get(keys[0]) if keys else None
        else:
            agent = self._cache.get(self._sk(session_id, owner))
            if agent is None and owner is None:
                for key in self._map_keys(session_id, None):
                    agent = self._cache.get(key)
                    if agent is not None:
                        break
        if agent and agent.model:
            return working_context_window(
                agent.model.context_window,
                compact_token_limit_of(agent.tool_config) if agent.tool_config else None,
            )
        return 128000

    @staticmethod
    def expand_skill_invocation(message: str) -> str:
        """Turn a ``/skill-trigger [arguments]`` message into the skill prompt.

        The web input box inserts skill references as plain ``/trigger`` text,
        so without this the model only ever saw the raw slash line. Uses the
        same renderer as the CLI so both surfaces frame arguments identically.
        """
        expanded = get_skill_registry().expand_invocation(message)
        return expanded if expanded is not None else message

    @staticmethod
    async def _prepare_run_media(agent: Any, message: str, media: Optional[List["InboundMedia"]]):
        """Route inbound images/audio/video the same way IM channels do.

        Capable base models get the payload on ``agent.run(images=)`` /
        ``audio=``. Everything else is described by ``settings.media_model``
        (Gemini) and appended to the user text.
        """
        if not media:
            return message, None, None, []
        plan = await media_understanding.prepare(
            media,
            base_model_id=agent.model.id or "",
            base_supports_images=bool(agent.model.supports_images),
        )
        if plan.text_parts:
            message = message + "\n\n" + "\n\n".join(plan.text_parts)
        return message, (plan.images or None), plan.audio, plan.notes

    async def chat(
        self,
        message: str,
        session_id: str,
        user_id: str = "default",
        source: RunSource = RunSource.gateway,
        media: Optional[List["InboundMedia"]] = None,
        owner: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> ChatResult:
        """Send a message and return the full response (non-streaming).

        Acquires a per-session lock to prevent concurrent runs on the same
        Agent instance (which is not thread-safe).

        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier (for workspace memory isolation)
            owner: Which ``users/<id>/`` partition this session is stored in.
                Web routes pass the signed-in account; unset means the machine's
                own partition (cron, IM channels, the CLI).
            media: Downloaded inbound media (image/voice/video payloads) from
                the channel. Images attach to the run when the base model can
                see them; audio/video attach when the base is Gemini. Anything
                else is described/transcribed by ``settings.media_model``.
            channel: The IM channel this turn came from (``"wechat"`` …);
                recorded as the session's latest entry point. None = web.

        Returns:
            ChatResult with content, tool_calls, metrics

        Raises:
            RuntimeError: If another run is already active on this session.
        """
        await self._ensure_initialized()

        lock = self._get_session_lock(session_id, owner)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            agent = await self._get_agent(session_id, owner)
            message = replace_invalid_utf8(message)
            message, run_images, run_audio, media_notes = await self._prepare_run_media(
                agent, message, media,
            )

            try:
                if self._workspace:
                    await asyncio.to_thread(
                        self._workspace.set_user, self._owner(owner or user_id)
                    )

                response = await agent.run(
                    self.expand_skill_invocation(message),
                    config=self._run_config_for_session(session_id, source, channel=channel),
                    images=run_images,
                    audio=run_audio,
                )

                content = (response.content or "").strip()
                tools_used: List[str] = []
                tool_calls = 0

                if response.tools:
                    tool_calls = len(response.tools)
                    for tool in response.tools:
                        if isinstance(tool, dict):
                            tools_used.append(tool.get("tool_name", tool.get("name", "unknown")))
                        else:
                            tools_used.append(str(tool))

                return ChatResult(
                    content=content,
                    tool_calls=tool_calls,
                    session_id=session_id,
                    user_id=user_id,
                    tools_used=tools_used,
                    metrics=extract_metrics(agent),
                    media_notes=media_notes,
                    usage=await usage_payload(agent, model_provider=self._session_provider(session_id, owner)),
                    turn_usage=turn_usage_payload(agent),
                )

            except Exception as e:
                logger.error(f"AgentService.chat error (session={session_id}): {e}")
                return ChatResult(
                    content=f"Error: {e}",
                    tool_calls=0,
                    session_id=session_id,
                    user_id=user_id,
                    media_notes=media_notes,
                )

    async def run_cron(
        self,
        message: str,
        job_id: str,
        user_id: str = "default",
        permission_mode: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> ChatResult:
        """Run a scheduled job's prompt on a brand-new, independent Agent.

        Unlike chat()/chat_stream(), this never reads from or writes to the
        interactive-session Agent cache (self._cache): every cron run builds
        and discards its own Agent, so job run N+1 never inherits any
        conversation state from run N or from any interactive chat session.
        The per-job session id keeps run-lock / work_dir / approval-mode
        bookkeeping scoped to the job (one entry per job, not one per run),
        but is CRON_SESSION_PREFIX-tagged so it's excluded from
        list_sessions() (the chat sidebar).

        ``permission_mode`` / ``work_dir`` are the job's mandate (see
        ``CronJob``). Empty / None clears any earlier per-job override so the
        job runs on the gateway defaults.
        """
        await self._ensure_initialized()
        session_id = f"{CRON_SESSION_PREFIX}{job_id}"
        if permission_mode:
            self.set_session_approval_mode(session_id, permission_mode, user_id)
        else:
            self._session_approval_modes.pop(self._sk(session_id, user_id), None)
        if work_dir:
            self.set_session_work_dir(session_id, work_dir, user_id)
        else:
            self._session_work_dirs.pop(self._sk(session_id, user_id), None)

        lock = self._get_session_lock(session_id, user_id)
        if lock.locked():
            raise RuntimeError(f"Job '{job_id}' already has an active run.")

        async with lock:
            agent = await asyncio.wait_for(
                asyncio.to_thread(self._build_agent, session_id, user_id),
                timeout=_AGENT_BUILD_TIMEOUT_S,
            )
            agent.add_session_guidance(scheduled_job_guidance(job_id))
            try:
                if self._workspace:
                    await asyncio.to_thread(self._workspace.set_user, user_id)

                response = await agent.run(
                    message,
                    config=self._run_config_for_session(session_id, RunSource.cron),
                )
                return ChatResult(
                    content=(response.content or "").strip(),
                    tool_calls=len(response.tools) if response.tools else 0,
                    session_id=session_id,
                    user_id=user_id,
                    metrics=extract_metrics(agent),
                    turn_usage=turn_usage_payload(agent),
                )
            except Exception as e:
                logger.error(f"AgentService.run_cron error (job={job_id}): {e}")
                return ChatResult(
                    content=f"Error: {e}",
                    tool_calls=0,
                    session_id=session_id,
                    user_id=user_id,
                )

    async def run_goal(
        self,
        objective: str,
        session_id: str,
        user_id: str = "default",
        owner: Optional[str] = None,
        token_budget: int = -1,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
        on_content: Optional[Callable[[str], Any]] = None,
        on_tool_call: Optional[Callable[[str, dict], Any]] = None,
        on_tool_result: Optional[Callable[..., Any]] = None,
        on_thinking: Optional[Callable[[str], Any]] = None,
    ) -> Dict[str, Any]:
        """Drive a standing-goal loop (Agent.run_goal) for the web UI's ``/goal``.

        Web exposes one knob: ``token_budget`` (``-1`` = unlimited, the default).
        Turn and wall-clock caps stay off here; CLI ``/goal --turns/--wall`` still
        has them. ``on_event`` receives a compact status dict after each
        ``goal.*`` event so the web bar can tick. Content / tool / thinking
        callbacks match ``chat_stream`` so the UI paints each turn as it happens.
        Runs on the cached session agent (``isolate=False``), not a clone, so
        steer / cancel / the next chat turn see the same instance.
        """
        await self._ensure_initialized()

        lock = self._get_session_lock(session_id, owner)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        def _cb(event, payload: Dict[str, Any]) -> None:
            if on_event is None:
                return
            data = goal_event_payload(payload)
            data["event"] = getattr(event, "value", str(event))
            on_event(data)

        async def _on_chunk(chunk) -> None:
            await dispatch_stream_chunk(
                chunk, on_content, on_tool_call, on_tool_result, on_thinking,
            )

        async with lock:
            agent = await self._get_agent(session_id, owner)
            if self._workspace:
                await asyncio.to_thread(
                    self._workspace.set_user, self._owner(owner or user_id)
                )
            result = await agent.run_goal(
                objective,
                token_budget=token_budget,
                isolate=False,
                event_callback=_cb if on_event else None,
                stream_chunks=_on_chunk,
            )
            return {
                "status": result.status,
                "reason": result.reason,
                "content": result.response_content,
                "turns_used": result.turns_used,
            }

    async def compact_session(
        self,
        session_id: str,
        owner: Optional[str] = None,
        instructions: str = "",
    ) -> Dict[str, Any]:
        """Start a new context window — the web counterpart of CLI ``/compact``.

        Same cut as the CLI: no native summarizer, no LLM summary. A failed
        compact leaves the transcript unchanged.
        """
        await self._ensure_initialized()
        lock = self._get_session_lock(session_id, owner)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            agent = await self._get_agent(session_id, owner)
            wm = agent.working_memory
            messages = wm.messages
            msg_count = len(messages)
            if msg_count == 0:
                return {"ok": False, "error": "No messages to compact."}

            # Idle: _run_hooks is cleared after every run, so the agent's own
            # memory/experience hooks are only reachable through the defaults.
            hooks = agent._default_run_hooks
            if hooks is not None:
                await hooks.on_pre_compact(agent=agent, messages=messages)

            cm = agent.tool_config.compression_manager if agent.tool_config else None
            if cm is None:
                return {"ok": False, "error": "No compression manager on this agent; nothing to compact with."}
            compacted = await apply_idle_compact(agent)
            if not compacted:
                return {"ok": False, "error": "New context window failed; conversation left unchanged."}

            if hooks is not None:
                await hooks.on_post_compact(agent=agent, messages=messages)

            notes, notes_path = _read_compact_notes(agent)
            return {
                "ok": True,
                "native": False,
                "window_id": cm.window_id,
                "messages_before": msg_count,
                "messages_after": len(messages),
                # What the new window starts from; the Web shows it under the
                # compact result, as the CLI prints its path and diff.
                "notes": notes,
                "notes_path": notes_path,
                "usage": await usage_payload(agent, model_provider=self._session_provider(session_id, owner)),
            }

    async def chat_stream(
        self,
        message: str,
        session_id: str,
        user_id: str = "default",
        source: RunSource = RunSource.gateway,
        on_content: Optional[Callable[[str], Any]] = None,
        on_tool_call: Optional[Callable[[str, dict], Any]] = None,
        on_tool_result: Optional[Callable[..., Any]] = None,
        on_thinking: Optional[Callable[[str], Any]] = None,
        owner: Optional[str] = None,
        media: Optional[List["InboundMedia"]] = None,
        include_usage: bool = True,
        channel: Optional[str] = None,
        automated: bool = False,
    ) -> ChatResult:
        """Send a message and stream the response via callbacks.

        Acquires the per-session run lock to prevent concurrent runs
        (both chat and chat_stream) on the same Agent instance.

        Args:
            message: User message
            session_id: Session identifier
            user_id: User identifier
            owner: Which ``users/<id>/`` partition this session is stored in.
                Web routes pass the signed-in account; unset means the machine's
                own partition (cron, IM channels, the CLI).
            on_content: Called with each content delta
            on_tool_call: Called when a tool call starts (name, args)
            on_tool_result: Called when a tool call completes (name, result, extra)
            on_thinking: Called with each reasoning delta
            media: Pasted/attached images from the web UI (same routing as IM).
            channel: The IM channel this turn came from; None = web.

        Returns:
            ChatResult with accumulated content + metrics

        Raises:
            RuntimeError: If another run is already active on this session.
        """
        lock = self._get_session_lock(session_id, owner)
        if lock.locked():
            raise RuntimeError(
                f"Session '{session_id}' already has an active run. "
                "Wait for it to complete or cancel it first."
            )

        async with lock:
            return await self._chat_stream_impl(
                message, session_id, user_id,
                source, on_content, on_tool_call, on_tool_result, on_thinking,
                owner, media, include_usage=include_usage, channel=channel,
                automated=automated,
            )

    async def _chat_stream_impl(
        self,
        message: str,
        session_id: str,
        user_id: str,
        source: RunSource,
        on_content: Optional[Callable[[str], Any]],
        on_tool_call: Optional[Callable[[str, dict], Any]],
        on_tool_result: Optional[Callable[..., Any]],
        on_thinking: Optional[Callable[[str], Any]],
        owner: Optional[str],
        media: Optional[List["InboundMedia"]],
        *,
        include_usage: bool = True,
        channel: Optional[str] = None,
        automated: bool = False,
    ) -> ChatResult:
        """Internal stream implementation (called under per-session lock)."""
        await self._ensure_initialized()
        agent = await self._get_agent(session_id, owner)
        message = replace_invalid_utf8(message)
        message, run_images, run_audio, media_notes = await self._prepare_run_media(
            agent, message, media,
        )

        try:
            if self._workspace:
                await asyncio.to_thread(
                    self._workspace.set_user, self._owner(owner or user_id)
                )

            full_content = ""
            reasoning_content = ""
            tools_used: List[str] = []
            tool_calls = 0

            async for chunk in agent.run_stream(
                self.expand_skill_invocation(message),
                config=self._run_config_for_session(
                    session_id,
                    source,
                    channel=channel,
                    stream_intermediate_steps=True,
                    automated=automated,
                ),
                images=run_images,
                audio=run_audio,
            ):
                if chunk is None:
                    continue

                display_event = classify_run_response(chunk)
                if display_event.kind == RunDisplayEventKind.TOOL_STARTED:
                    tool_info = chunk.tool_call
                    if tool_info:
                        tools_used.append(tool_info.tool_name or "unknown")
                        tool_calls += 1
                elif display_event.kind == RunDisplayEventKind.CONTENT_DELTA:
                    if chunk.reasoning_content:
                        reasoning_content += chunk.reasoning_content
                    if chunk.content:
                        full_content += chunk.content

                await dispatch_stream_chunk(
                    chunk, on_content, on_tool_call, on_tool_result, on_thinking,
                )

            # A report nobody typed is no name for the chat.
            if not automated:
                self._schedule_title(session_id, owner, message, full_content)
            usage = (
                await usage_payload(agent, model_provider=self._session_provider(session_id, owner))
                if include_usage
                else None
            )
            return ChatResult(
                content=full_content.strip(),
                tool_calls=tool_calls,
                session_id=session_id,
                user_id=user_id,
                tools_used=tools_used,
                reasoning=reasoning_content,
                metrics=extract_metrics(agent),
                media_notes=media_notes,
                usage=usage,
                turn_usage=turn_usage_payload(agent),
            )

        except (asyncio.CancelledError, AgentCancelledError, KeyboardInterrupt):
            logger.info(f"AgentService stream cancelled (session={session_id})")
            raise

        except Exception as e:
            logger.error(f"AgentService.chat_stream error (session={session_id}): {e}")
            return ChatResult(
                content=f"Error: {e}",
                tool_calls=0,
                session_id=session_id,
                user_id=user_id,
            )

    # ============== Session management ==============

    def _session_preview(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        path = str(entry["path"])
        version = (
            int(entry.get("size_bytes") or 0),
            int(entry.get("mtime_ns") or 0),
        )
        with self._session_preview_lock:
            cached = self._session_preview_cache.get(path)
            if cached is not None and cached[0] == version:
                self._session_preview_cache.move_to_end(path)
                return cached[1]
        preview = SessionLog.session_preview(path)
        with self._session_preview_lock:
            self._session_preview_cache[path] = (version, preview)
            self._session_preview_cache.move_to_end(path)
            while len(self._session_preview_cache) > 10000:
                self._session_preview_cache.popitem(last=False)
        return preview

    def list_sessions(
        self,
        owner: Optional[str] = None,
        scope: str = "recent",
        recent_days: int = 30,
        include_cli: bool = False,
        hide_temp_dirs: bool = False,
    ) -> List[Dict[str, Any]]:
        """List lightweight session summaries across the owner's projects.

        ``recent`` keeps every gateway session, archived session and active run,
        but only CLI sessions modified within ``recent_days``. Filtering
        happens before preview generation so old logs are never parsed.

        Only the three product surfaces share history: gateway sessions (web,
        desktop, IM) are always listed and CLI sessions are added when
        ``include_cli`` is set. SDK scripts and unlabelled legacy logs write to
        the same project tree but are never listed. ``hide_temp_dirs`` drops
        *empty* residue under the system temp dir (``/tmp``, ``/var/folders``):
        zero-byte stubs and logs with no real user turn, left behind by test
        runs and demos. A conversation someone actually had in a temp working
        directory is a real session and stays reachable — hiding it would make
        it disappear from the web with no way to bring it back.
        Running sessions are always listed.
        """
        if scope not in ("all", "recent"):
            raise ValueError(f"Unknown session scope: {scope}")
        if recent_days < 1:
            raise ValueError("recent_days must be positive")
        uid = self._owner(owner)
        projects = SessionLog.list_projects(user_id=uid)
        if not projects:
            projects = [{
                "base_dir": self._session_base_dir(str(settings.base_dir), owner),
                "work_dir": str(settings.base_dir),
            }]
        candidates: Dict[str, tuple[Dict[str, Any], str]] = {}
        for project in projects:
            base_dir = project["base_dir"]
            work_dir = project.get("work_dir") or ""
            for entry in SessionLog.list_sessions(base_dir=base_dir, work_dir=work_dir or None):
                sid = entry["session_id"]
                if sid.startswith(CRON_SESSION_PREFIX):
                    continue
                previous = candidates.get(sid)
                if previous is None or int(entry.get("size_bytes") or 0) > int(previous[0].get("size_bytes") or 0):
                    candidates[sid] = (entry, work_dir)

        cutoff = time.time() - recent_days * 86400
        # The gateway's own default directory is where it was told to work,
        # even when that is under /tmp.
        gateway_dir = os.path.realpath(str(settings.base_dir))

        def _hidden_temp(work_dir: str) -> bool:
            if not is_system_temp_path(work_dir):
                return False
            real = os.path.realpath(work_dir)
            return not (real == gateway_dir or real.startswith(gateway_dir + os.sep))

        out: List[Dict[str, Any]] = []
        for entry, project_work_dir in candidates.values():
            sid = entry["session_id"]
            running = (
                self.is_session_active(sid, owner=uid)
                or live_turn.active(sid, owner=uid) is not None
            )
            source = entry.get("source")
            im_from_sid = _infer_im_source_from_session_id(sid)
            if not source:
                # A CLI flag is authoritative. Otherwise an old Web session
                # (written before `source` existed) only has `last_read_at` to
                # say the gateway showed it; dropping that hint would hide the
                # user's own web history after an upgrade with no way back.
                if im_from_sid:
                    source = im_from_sid
                elif entry.get("cli"):
                    source = "cli"
                elif entry.get("last_read_at"):
                    source = "gateway"
                else:
                    source = "legacy"
            # ``source`` is where the latest turn came in, ``origin`` where the
            # session started. A conversation begun in WeChat and continued in
            # the CLI is still one the gateway owns, so either one counts.
            # An ``agent:<id>:<channel>:<chat>`` id was minted by that channel,
            # which settles where it started even for sidecars written before
            # ``origin`` existed.
            origin = im_from_sid or entry.get("origin") or source
            is_gateway_or_im = any(
                s == "gateway" or s in IM_SESSION_SOURCES for s in (source, origin)
            )
            archived = bool(entry.get("archived"))
            pinned = bool(entry.get("pinned"))
            if not running and not (is_gateway_or_im or (include_cli and source == "cli")):
                continue
            if (
                not running
                and hide_temp_dirs
                and _hidden_temp(entry.get("work_dir") or project_work_dir)
                # Only residue. Parsing the log for user turns is cheap here
                # because it is one line at a time and preview is cached; the
                # alternative is losing a real conversation with no UI to
                # restore it.
                and not self._session_preview(entry).get("user_count")
            ):
                continue
            if (
                scope == "recent"
                and not is_gateway_or_im
                and not running
                and not archived
                and not pinned
                and float(entry.get("mtime") or 0) < cutoff
            ):
                continue
            preview = self._session_preview(entry)
            first_user = chat_images.strip_markers(preview.get("first_user", "") or "")
            first_ts = entry.get("first_timestamp")
            if not first_ts and entry.get("mtime"):
                first_ts = iso_timestamp(entry["mtime"])
            last_ts = entry.get("last_timestamp")
            last_read_at = entry.get("last_read_at")
            out.append({
                "session_id": sid,
                "name": entry.get("name") or (first_user[:40] if first_user else "Chat"),
                "preview": first_user,
                "user_count": preview.get("user_count", 0),
                "first_timestamp": first_ts,
                "last_timestamp": last_ts,
                "last_read_at": last_read_at,
                "unread": (not running) and is_session_unread(last_ts, last_read_at),
                "size_bytes": entry.get("size_bytes", 0),
                "archived": archived,
                "pinned": pinned,
                "work_dir": entry.get("work_dir") or project_work_dir,
                "running": running,
                "source": source,
                "origin": origin,
            })
        seen = {row["session_id"] for row in out}
        for turn in live_turn.iter_owner(uid):
            if turn.session_id.startswith(CRON_SESSION_PREFIX) or turn.session_id in seen:
                continue
            live_source = (
                self.session_log_for(turn.session_id, owner=uid).get_source()
                or _infer_im_source_from_session_id(turn.session_id)
                or "gateway"
            )
            out.append({
                "session_id": turn.session_id,
                "name": "Chat",
                "preview": "",
                "user_count": 0,
                "first_timestamp": iso_timestamp(),
                "last_timestamp": None,
                "last_read_at": None,
                "unread": False,
                "size_bytes": 0,
                "archived": False,
                "work_dir": self.get_session_work_dir(turn.session_id, owner=uid),
                "running": not turn.done,
                "source": live_source,
            })
        out.sort(key=lambda row: str(row.get("first_timestamp") or ""), reverse=True)
        return out

    def session_log_for(self, session_id: str, owner: Optional[str] = None) -> SessionLog:
        """Open the on-disk SessionLog for a session (may not exist yet)."""
        base_dir, work_dir = self._locate_session(session_id, owner)
        return SessionLog(session_id=session_id, base_dir=base_dir, work_dir=work_dir or None)

    @staticmethod
    def _jsonl_has_content(path: Path) -> bool:
        """True when ``path`` is a real transcript, not a 0-byte touch stub."""
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def _locate_session(self, session_id: str, owner: Optional[str] = None) -> tuple[str, str]:
        """Find the project directory that actually holds this session's jsonl.

        ``_session_work_dirs`` is process memory. After a gateway restart it is
        empty, and guessing ``settings.base_dir`` is how View trace 404'd a log
        that had been written under another project. Search the owner's tree.

        A 0-byte ``.jsonl`` is only the pre-append touch stub. Treating it as
        a hit under ``settings.base_dir`` hid the real transcript (title
        became ``Chat``, trace empty) whenever the session's work_dir was a
        different project.
        """
        remembered = self._session_work_dirs.get(self._sk(session_id, owner))
        if remembered:
            # Brand-new chats touch an empty file here before the first
            # append. Do not fall through to settings.base_dir just because
            # the file is still missing or 0 bytes.
            return self._session_base_dir(remembered, owner), remembered

        fallback_work = str(settings.base_dir)
        fallback = self._session_base_dir(fallback_work, owner)
        hits = [
            h for h in SessionLog.find_sessions(session_id, user_id=self._owner(owner))
            if h.get("session_id") == session_id
        ]
        contentful = [h for h in hits if (h.get("size_bytes") or 0) > 0]
        chosen = contentful or hits
        if chosen:
            hit = max(chosen, key=lambda h: int(h.get("size_bytes") or 0))
            work_dir = hit.get("work_dir") or fallback_work
            if work_dir:
                self._session_work_dirs[self._sk(session_id, owner)] = work_dir
            keep = Path(hit["path"])
            for shadow in hits:
                path = Path(shadow["path"])
                if path != keep and not self._jsonl_has_content(path):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            return str(hit["base_dir"]), work_dir
        return fallback, fallback_work

    def _drop_cached_agents(self) -> None:
        """Make every session rebuild its agent with the current settings.

        A session with a run in flight keeps its agent until the run ends (it
        finishes on the model it started with; evicting it would also cut off
        cancel and steer), then rebuilds. A model/profile switch therefore
        never has to wait for, or refuse because of, another session's run.
        """
        running = {key for key, lock in self._session_locks.items() if lock.locked()}
        for key in self._cache.keys():
            if key in running:
                self._stale_agents.add(key)
            else:
                self._cache.delete(key)

    def delete_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Delete a session: cached Agent + persistent SessionLog JSONL + meta.

        Removes the on-disk JSONL and sidecar meta so the session does not
        reappear after restart (SessionLog is the single source of truth).
        Returns True if either the cache or the on-disk log existed.
        """
        base_dir, _work_dir = self._locate_session(session_id, owner)
        key = self._sk(session_id, owner)
        removed = self._cache.delete(key)
        live_turn.drop(session_id, owner=self._owner(owner))
        self._session_work_dirs.pop(key, None)
        self._session_approval_modes.pop(key, None)
        self._session_grants.pop(key, None)
        self._session_ask_callbacks.pop(key, None)
        background = self._session_background.pop(key, None)
        if background is not None:
            # stop() can take seconds on a command that ignores SIGTERM.
            threading.Thread(target=background.stop, daemon=True).start()
        self._session_locks.pop(key, None)
        self._stale_agents.discard(key)
        log_existed = False
        try:
            log = SessionLog(session_id=session_id, base_dir=base_dir)
            with self._session_preview_lock:
                self._session_preview_cache.pop(str(log.path), None)
            if log.path.exists():
                log.path.unlink()
                log_existed = True
            meta = log.base_dir / f"{session_id}.meta.json"
            if meta.exists():
                meta.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove SessionLog for {session_id}: {e}")
        logger.debug(f"Session deleted: {session_id}")
        return removed or log_existed

    # ============== Background commands (the Web's /ps and /stop) ==============

    def _background_registry(self, session_id: str, owner: Optional[str]) -> BackgroundProcessRegistry:
        key = self._sk(session_id, owner)
        registry = self._session_background.get(key)
        if registry is None:
            registry = BackgroundProcessRegistry(
                user_id=self._owner(owner),
                delivers_results=self.background_report_handler is not None,
            )
            self._session_background[key] = registry
            threading.Thread(
                target=self._deliver_background_reports,
                args=(registry, session_id, owner),
                daemon=True,
                name=f"bg-report-{session_id}",
            ).start()
        return registry

    def _deliver_background_reports(
        self, registry: BackgroundProcessRegistry, session_id: str, owner: Optional[str],
    ) -> None:
        """Hand each finished command back to the agent, like the CLI does.

        A user's stop is not news. Ends when the session is deleted (its
        registry is dropped) — the queue then never fills again.
        """
        while self._background_registry_is_current(session_id, owner, registry):
            try:
                event = registry.wait_completed(timeout=1.0)
            except queue.Empty:
                continue
            if event.stop_requested:
                continue
            loop = self._loop
            if loop is None or loop.is_closed():
                return
            future = asyncio.run_coroutine_threadsafe(
                self._report_background(session_id, owner, registry, event), loop,
            )
            future.add_done_callback(_log_report_failure)

    def _background_registry_is_current(
        self, session_id: str, owner: Optional[str], registry: BackgroundProcessRegistry,
    ) -> bool:
        """False once the session is deleted: its pending reports are void."""
        return self._session_background.get(self._sk(session_id, owner)) is registry

    async def _report_background(
        self,
        session_id: str,
        owner: Optional[str],
        registry: BackgroundProcessRegistry,
        event: BackgroundProcessCompleted,
    ) -> None:
        report = background_result_for_agent(event)
        # A report is its own automated turn once the session is idle. Not a
        # steer into a running turn: one accepted after the last tool batch is
        # parked, and the gateway re-queues parked text as the user's words.
        while self._background_registry_is_current(session_id, owner, registry) and (
            self.is_session_active(session_id, owner) or live_turn.active(session_id, self._owner(owner))
        ):
            await asyncio.sleep(1.0)
        # Deleting the chat while the report waited must not bring it back.
        handler = self.background_report_handler
        if handler is None or not self._background_registry_is_current(session_id, owner, registry):
            return
        await handler(session_id, self._owner(owner), report)

    @staticmethod
    def _background_job_view(item: BackgroundProcess) -> Dict[str, Any]:
        if not item.finished.is_set():
            status = "running"
        elif item.stop_requested:
            status = "stopped"
        else:
            status = "completed" if item.returncode == 0 else "failed"
        return {
            "id": item.id,
            "num": item.num,
            "kind": item.kind,
            # A delegated worker's command line is `agentica --query <whole task>`.
            "label": item.label if item.kind == "delegate" else item.command,
            "status": status,
            "exit_code": item.returncode,
            "started_at_ms": int(item.started_at * 1000),
            "finished_at_ms": int(item.finished_at * 1000) if item.finished_at else None,
        }

    def _background_job(
        self, session_id: str, job_id: str, owner: Optional[str],
    ) -> Optional[BackgroundProcess]:
        registry = self._session_background.get(self._sk(session_id, owner))
        return registry.get(job_id) if registry is not None else None

    def list_background_jobs(self, session_id: str, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        registry = self._session_background.get(self._sk(session_id, owner))
        if registry is None:
            return []
        return [self._background_job_view(item) for item in registry.list(include_finished=True)]

    def background_job_output(
        self, session_id: str, job_id: str, owner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        item = self._background_job(session_id, job_id, owner)
        if item is None:
            return None
        return {
            **self._background_job_view(item),
            "output": read_log_tail(item.log_path, max_lines=400, max_chars=20000),
            "log_path": item.log_path,
        }

    async def stop_background_job(
        self, session_id: str, job_id: str, owner: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        item = self._background_job(session_id, job_id, owner)
        if item is None:
            return None
        registry = self._session_background[self._sk(session_id, owner)]
        # By the resolved id: stop() reads "all" as every command in the session.
        await asyncio.to_thread(registry.stop, item.id)
        return self._background_job_view(item)

    def stop_all_background_jobs(self) -> None:
        for registry in list(self._session_background.values()):
            registry.stop()

    def fork_session(self, session_id: str, at_uuid: Optional[str] = None, owner: Optional[str] = None) -> str:
        """Copy a session up to ``at_uuid`` (inclusive) into a new one; return its id.

        The branch lives in the same project and is marked as a gateway session,
        so it shows in the web list even when CLI sessions are hidden. Its name
        says where it came from; the original is untouched.
        """
        log = self.session_log_for(session_id, owner=owner)
        if not log.path.exists() or log.path.stat().st_size == 0:
            raise FileNotFoundError(f"Session not found: {session_id}")
        if at_uuid and not log.has_entry(at_uuid):
            raise ValueError(f"No entry {at_uuid} in session {session_id}")
        new_id = str(uuid.uuid4())
        branch = log.fork(new_id, at_uuid=at_uuid)
        branch.set_source(RunSource.gateway.value)
        meta = SessionLog._read_meta(log.meta_path)
        title = SessionLog._meta_name(meta) or (chat_images.strip_markers(log.first_user_message() or "") or "Chat")[:40]
        SessionLog.rename_session(new_id, f"{title} (fork)", base_dir=str(log.base_dir))
        if log._cwd:
            self._session_work_dirs[self._sk(new_id, owner)] = log._cwd
        return new_id

    def session_feedback(self, session_id: str, owner: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        return self.session_log_for(session_id, owner=owner).read_feedback()

    def set_session_feedback(
        self, session_id: str, entry_uuid: str, rating: Optional[str], comment: str = "",
        owner: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        log = self.session_log_for(session_id, owner=owner)
        if not log.path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        if not log.has_entry(entry_uuid):
            raise ValueError(f"No entry {entry_uuid} in session {session_id}")
        return log.set_feedback(entry_uuid, rating, comment)

    def rename_session(self, session_id: str, name: str, owner: Optional[str] = None) -> None:
        """Rename a session by writing the sidecar .meta.json (SessionLog)."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.rename_session(session_id, name, base_dir=base_dir)

    def _schedule_title(self, session_id: str, owner: Optional[str], question: str, answer: str) -> None:
        """Name an unnamed session from its first exchange, in the background.

        One attempt per session per process: a failed call leaves the
        first-message fallback the sidebar already shows.
        """
        if session_id.startswith(CRON_SESSION_PREFIX) or not answer.strip():
            return
        key = (owner or "", session_id)
        if key in self._titled:
            return
        self._titled.add(key)
        log = self.session_log_for(session_id, owner=owner)
        if log.get_name():
            return
        task = asyncio.create_task(self._write_title(log, question, answer))
        self._title_tasks.add(task)
        task.add_done_callback(self._title_tasks.discard)

    async def _write_title(self, log: SessionLog, question: str, answer: str) -> None:
        prompt = (
            f"User:\n{chat_images.strip_markers(question)[:2000]}\n\n"
            f"Assistant:\n{answer[:2000]}"
        )
        try:
            title = await self.side_task(_TITLE_PROMPT, prompt)
        except Exception as e:
            logger.warning(f"Session title generation failed ({log.session_id}): {e}")
            return
        title = title.splitlines()[0].strip().strip("\"'“”「」").strip() if title else ""
        # The user may have renamed it while the call ran.
        if title and not log.get_name():
            log.set_name(title[:60])

    def archive_session(
        self, session_id: str, archived: bool = True, owner: Optional[str] = None,
    ) -> None:
        """Archive/unarchive a session by writing SessionLog sidecar metadata."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.archive_session(session_id, archived=archived, base_dir=base_dir)

    def pin_session(
        self, session_id: str, pinned: bool = True, owner: Optional[str] = None,
    ) -> None:
        """Pin/unpin a session (sidebar order) by writing sidecar metadata."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        SessionLog.pin_session(session_id, pinned=pinned, base_dir=base_dir)

    def mark_session_read(self, session_id: str, owner: Optional[str] = None) -> str:
        """Stamp ``last_read_at`` so the sidebar drops the unread dot."""
        base_dir, _work_dir = self._locate_session(session_id, owner)
        return SessionLog.mark_session_read(session_id, base_dir=base_dir)

    def clear_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Alias for delete_session (for compatibility)."""
        return self.delete_session(session_id, owner)

    def cancel_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Cancel the in-flight run for a specific session.

        Returns True if the session has an agent to cancel, False otherwise.
        """
        agent = None
        for key in self._map_keys(session_id, owner):
            agent = self._cache.get(key)
            if agent is not None:
                break
        if agent is None:
            return False
        try:
            agent.cancel()
            logger.debug(f"Cancelled agent for session: {session_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel session {session_id}: {e}")
            return False

    def _cached_agent_for_owner(self, session_id: str, owner: Optional[str] = None):
        """Cached agent for this session, only if it belongs to ``owner``.

        Does not build a new agent: steering a session that is not running
        (or not this account's) must return False so the caller can queue.
        """
        agent = self._cache.get(self._sk(session_id, owner))
        if agent is None or agent.user_id != self._owner(owner):
            return None
        return agent

    def steer_session(self, session_id: str, guidance: str, owner: Optional[str] = None) -> bool:
        """Inject mid-run guidance (CLI ``agent.steer``) for a web session.

        Returns True when the live run accepted it (delivered at the next
        tool-batch boundary). False means the caller must queue a fresh turn:
        no cached agent, wrong owner, or the run ended in the TOCTOU window.
        """
        agent = self._cached_agent_for_owner(session_id, owner)
        if agent is None:
            return False
        return bool(agent.steer(self.expand_skill_invocation(guidance)))

    def take_undelivered_steer(self, session_id: str, owner: Optional[str] = None) -> List[str]:
        """Pop steering accepted too late for this run, so the web UI can queue it.

        Same contract as CLI ``promote_late_steer``: a True from ``steer()``
        during the final inference is parked on the agent and must be taken
        after the run ends, or it sits forever.
        """
        agent = self._cached_agent_for_owner(session_id, owner)
        if agent is None:
            return []
        return [item[0] for item in agent.pop_undelivered_steer()]

    # ============== Work directory ==============

    def set_session_work_dir(
        self, session_id: str, work_dir: str, owner: Optional[str] = None,
    ) -> None:
        """Set the working directory for a specific session.

        Per-session work_dirs override the global settings.base_dir.
        Does NOT clear other sessions' agents.
        """
        self._session_work_dirs[self._sk(session_id, owner)] = work_dir

    def get_session_work_dir(self, session_id: str, owner: Optional[str] = None) -> str:
        """Get the working directory for a session (falls back to global base_dir)."""
        key = self._sk(session_id, owner)
        if key not in self._session_work_dirs:
            self._locate_session(session_id, owner)
        return self._session_work_dirs.get(
            key, str(settings.base_dir),
        )

    # ============== Session storage scoping ==============

    def _session_base_dir(self, work_dir: str, owner: Optional[str] = None) -> str:
        """Resolve the SessionLog storage dir for a given project work_dir.

        Sessions are scoped by project (work_dir) + owner, mirroring exactly how
        the Agent writes them (see ``SessionLog`` construction in the agent).
        This is what makes the Web sidebar and the CLI ``/resume`` list a
        consistent set of sessions for the same project + user.
        """
        from agentica.config.project import project_base_dir
        return project_base_dir(work_dir, user_id=self._owner(owner))

    @staticmethod
    def _owner(owner: Optional[str]) -> str:
        """Which ``users/<id>/`` partition a call reads and writes.

        Unset means the machine's own partition (``settings.default_user_id``).
        IM channels pass the home account (whoever scanned WeChat, else
        default) so web / desktop / IM share sessions and memory. Cron jobs
        carry their own ``user_id``. Only the web surface has accounts, so
        only its routes pass an owner — and they take it from the session
        cookie, never from the request body.
        """
        return owner or settings.default_user_id

    @classmethod
    def _sk(cls, session_id: str, owner: Optional[str] = None) -> str:
        return f"{cls._owner(owner)}{_OWNER_SEP}{session_id}"

    def _map_keys(self, session_id: str, owner: Optional[str] = None) -> List[str]:
        """Keys in the owner-partitioned maps that refer to ``session_id``.

        A named owner is one key. Peer probes pass ``owner=None`` and match
        every account that has this id (and a leftover unpartitioned key).
        """
        if owner is not None:
            return [self._sk(session_id, owner)]
        suffix = f"{_OWNER_SEP}{session_id}"
        keys = {self._sk(session_id, None)}
        for k in self._cache.keys():
            if k == session_id or k.endswith(suffix):
                keys.add(k)
        for k in self._session_locks:
            if k == session_id or k.endswith(suffix):
                keys.add(k)
        return list(keys)

    def update_work_dir(self, new_dir: str) -> None:
        """Update the global work_dir and clear ALL cached agents.

        Called when the user changes the global working directory via the UI.
        All agents must be rebuilt to pick up the new directory.
        """
        self._cache.clear()
        self._session_work_dirs.clear()
        logger.info(f"Global work_dir updated to: {new_dir}, all agent instances cleared")

    def _auto_extract_for(self, owner: Optional[str]) -> bool:
        """Per-account toggle for conversation memory extraction. Default on."""
        from agentica_gateway.routes.settings import _clean_prefs, _read_prefs
        prefs = _clean_prefs(_read_prefs(self._owner(owner)))
        return bool(prefs.get("auto_extract_memory", True))

    def _switch_workspace_user(self, owner: str) -> Workspace:
        if self._workspace is None:
            raise RuntimeError("Workspace is not initialized")
        self._workspace.set_user(self._owner(owner))
        return self._workspace

    # ============== User AGENTS.md / MEMORY.md ==============

    async def read_user_agents_md(self, owner: str) -> dict:
        """Return this account's user-level AGENTS.md for the settings page."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        content = await asyncio.to_thread(ws.read_user_agents_md)
        return {
            "content": content,
            "path": str(ws.user_agent_md_path()),
            "empty_template": Workspace._is_empty_template(content),
            "user_id": self._owner(owner),
        }

    async def write_user_agents_md(self, owner: str, content: str) -> dict:
        """Replace this account's user-level AGENTS.md."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        try:
            await asyncio.to_thread(ws.write_user_agents_md, content)
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return await self.read_user_agents_md(owner)

    async def read_memory_index(self, owner: str) -> dict:
        """Return this account's MEMORY.md index entries for the settings page."""
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        entries = await ws.list_memory_index_entries()
        return {
            "auto_extract": self._auto_extract_for(owner),
            "index_path": str(ws.memory_index_path()),
            "entries": entries,
            "user_id": self._owner(owner),
        }

    async def read_memory_entry(self, owner: str, filename: str) -> dict | None:
        """One MEMORY.md topic file for the settings page.

        ``filename`` comes straight from the browser, so it must be a bare name
        that is actually listed in this account's MEMORY.md index. That keeps
        the endpoint from reading AGENTS.md or anything outside the index.
        """
        await self._ensure_initialized()
        ws = await asyncio.to_thread(self._switch_workspace_user, owner)
        entries = await ws.list_memory_index_entries()
        meta = next((e for e in entries if e.get("filename") == filename), None)
        if not meta:
            return None
        body, origin = await ws.read_memory_entry(filename)
        return {
            "filename": filename,
            "title": meta.get("title", ""),
            "hook": meta.get("hook", ""),
            "content": body,
            "origin_work_dir": origin,
            "path": str(ws.get_user_memory_dir() / filename),
            "user_id": self._owner(owner),
        }

    async def get_workspace_context(self, user_id: str = "default") -> str:
        """Retrieve workspace context prompt for a user."""
        await self._ensure_initialized()
        if self._workspace and self._workspace.exists():
            await asyncio.to_thread(self._workspace.set_user, user_id)
            return await self._workspace.get_context_prompt() or ""
        return ""

    async def list_users(self) -> List[str]:
        """List all known users from Workspace."""
        await self._ensure_initialized()
        if self._workspace:
            return await asyncio.to_thread(self._workspace.list_users)
        return []

    async def get_user_info(self, user_id: str) -> dict:
        """Get workspace user info."""
        await self._ensure_initialized()
        if self._workspace:
            return await asyncio.to_thread(self._workspace.get_user_info, user_id=user_id)
        return {"user_id": user_id}

    # ============== Hot reload ==============

    async def reload_profile(self, profile_name: Optional[str] = None) -> None:
        """Switch to a different config.yaml profile at runtime.

        Reloads main + auxiliary model config from the active profile (or
        ``profile_name`` if given), projects the profile's api_key/env into
        os.environ, then clears the agent cache so agents rebuild on next
        request with the new model.
        """
        if profile_name:
            set_active_profile(profile_name)
        profile = apply_global_config() or {}
        aux_profile = profile.get("auxiliary_model") or {}
        if not isinstance(aux_profile, dict):
            aux_profile = {}

        # apply_global_config uses setdefault semantics, so switching between
        # two profiles on the SAME provider would leave the old api_key in
        # place. Force-overwrite the provider env vars so the new profile's
        # key actually takes effect for SDK code paths that read the env.
        new_provider = profile.get("model_provider")
        new_api_key = profile.get("api_key")
        if new_provider and new_api_key:
            os.environ[provider_api_key_env(new_provider)] = new_api_key
        if aux_profile:
            aux_provider = aux_profile.get("model_provider")
            aux_key = aux_profile.get("api_key")
            if aux_provider and aux_key:
                os.environ[provider_api_key_env(aux_provider)] = aux_key

        async with self._init_lock:
            if profile.get("model_provider"):
                settings.model_provider = profile["model_provider"]
            if profile.get("model_name"):
                settings.model_name = profile["model_name"]
            settings.model_base_url = profile.get("base_url") or settings.model_base_url
            settings.model_api_key = profile.get("api_key") or settings.model_api_key
            settings.model_wire_api = profile.get("wire_api") or ""
            settings.model_reasoning = profile.get("reasoning") or ""
            settings.model_reasoning_effort = profile.get("reasoning_effort") or ""
            settings.max_tokens = int(profile.get("max_tokens") or 0)
            settings.temperature = float(profile.get("temperature") or 0)
            settings.top_p = float(profile.get("top_p") or 0)
            settings.context_window = int(profile.get("context_window") or 0)
            settings.auxiliary_model_provider = aux_profile.get("model_provider") or ""
            settings.auxiliary_model_name = aux_profile.get("model_name") or ""
            settings.auxiliary_base_url = aux_profile.get("base_url") or ""
            settings.auxiliary_api_key = aux_profile.get("api_key") or ""
            settings.auxiliary_wire_api = aux_profile.get("wire_api") or ""
            settings.auxiliary_reasoning = aux_profile.get("reasoning") or ""
            settings.auxiliary_reasoning_effort = aux_profile.get("reasoning_effort") or ""
            self._initialized = False
            self._drop_cached_agents()
            logger.info(
                f"Profile reloaded: {profile_name or 'active'} -> "
                f"{self.model_provider}/{self.model_name}"
            )

    async def _invalidate_cache(self) -> None:
        """Clear the agent cache so agents rebuild on next request.

        Used when a runtime-only setting changes (model switch, work dir)
        that does not require re-reading the profile.
        """
        async with self._init_lock:
            self._initialized = False
            self._drop_cached_agents()
            logger.info("Agent cache invalidated (will rebuild on next request)")

    async def add_tool(self, tool: Any) -> None:
        """Dynamically add a tool; clears agent cache to force rebuild."""
        async with self._init_lock:
            self.extra_tools.append(tool)
            self._initialized = False
            self._cache.clear()

    def add_instruction(self, instruction: str) -> None:
        """Append an instruction to all existing agents."""
        self.extra_instructions.append(instruction)
        for session_id in self._cache.keys():
            agent = self._cache.get(session_id)
            if agent:
                agent.add_instruction(instruction)

    # ============== Properties ==============

    @property
    def workspace(self) -> Optional[Workspace]:
        """Shared Workspace instance (synchronous; call after initialization)."""
        return self._workspace

    @property
    def agent(self) -> Optional[DeepAgent]:
        """Deprecated: returns an arbitrary cached DeepAgent.

        Prefer cancel_session(session_id) for targeted cancellation.
        """
        sessions = self._cache.keys()
        if sessions:
            return self._cache.get(sessions[0])
        return None
