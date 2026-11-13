# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared types, constants, LRU cache, and pure helpers for AgentService.
"""
from __future__ import annotations

import inspect
import os
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentica import DeepAgent
from agentica.compression.new_window import notes_path_for
from agentica.run.display import RunDisplayEventKind, classify_run_response
from agentica.utils.log import logger

from ...config import settings
from ..response_formatter import format_tool_call_args, format_tool_result


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


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

# Session-id prefix of the per-job sessions scheduled runs used to write before
# runs were delivered into conversations. Their logs are still on disk; they
# were never conversations, so the sidebar keeps leaving them out.
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

# How long an ask_user_question waits on the Web card / IM card before the
# model is told nobody answered (same 5 minutes as the IM pending action).
QUESTION_WAIT_S = 300.0


def _cache_key_is_pinned(key: str) -> bool:
    """True while this cache entry's Agent is running a LiveTurn.

    Evicting it would leave the run going while steer, cancel, approvals and
    questions look up a different (or missing) Agent.
    """
    owner, sep, session_id = key.partition(_OWNER_SEP)
    if not sep:
        owner, session_id = settings.default_user_id, key
    return _pkg().live_turn.active(session_id, owner) is not None


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
    # Set when the run failed; ``content`` then carries a readable form of it.
    error: str = ""


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
    path = notes_path_for(agent._session_log)
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
