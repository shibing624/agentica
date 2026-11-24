# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Shared types, constants, LRU cache, and pure helpers for AgentService.
"""
from __future__ import annotations

import inspect
import os
import sys
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


def _sibling_model_fields(
    block: Any, prefix: str, provider: str, base_url: str, api_key: str,
) -> Dict[str, Any]:
    """Flat ``{prefix}_*`` fields for one profile sibling-model block.

    Same provider as main inherits the main endpoint/key for omitted fields; a
    different provider keeps its own (empty means its preset / env var), never
    the main key.
    """
    block = block if isinstance(block, dict) else {}
    named = bool(block.get("model_name"))
    sibling_provider = block.get("model_provider") or provider
    same = sibling_provider == provider
    wire_api = block.get("wire_api") or ""
    return {
        f"{prefix}_model_provider": sibling_provider if named else "",
        f"{prefix}_model_name": block.get("model_name") or "",
        f"{prefix}_base_url": block.get("base_url") or (base_url if same else ""),
        f"{prefix}_api_key": block.get("api_key") or (api_key if same else ""),
        f"{prefix}_wire_api": wire_api,
        f"{prefix}_reasoning": block.get("reasoning") or "",
        f"{prefix}_reasoning_effort": "" if wire_api == "responses" else (block.get("reasoning_effort") or ""),
        f"{prefix}_extra_body": dict(block["extra_body"]) if isinstance(block.get("extra_body"), dict) else None,
        f"{prefix}_extra_headers": dict(block["extra_headers"]) if isinstance(block.get("extra_headers"), dict) else None,
    }


def _profile_model_config(name: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """One config.yaml profile as the flat model config the agent is built from.

    Empty base_url / api_key stay empty so the provider's own defaults (and its
    ``*_API_KEY`` env var) apply, as they do for the gateway default. The
    auxiliary block inherits the main endpoint only on the same provider.
    """
    wire_api = profile.get("wire_api") or ""
    provider = profile["model_provider"]
    base_url = profile.get("base_url") or ""
    api_key = profile.get("api_key") or ""
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
        "temperature": profile.get("temperature"),
        "top_p": profile.get("top_p"),
        "context_window": int(profile.get("context_window") or 0),
        "enable_cache_control": profile.get("enable_cache_control"),
        "cache_control_messages": profile.get("cache_control_messages"),
        "cache_control_session_header": profile.get("cache_control_session_header") or "",
        "extra_body": dict(profile["extra_body"]) if isinstance(profile.get("extra_body"), dict) else None,
        "extra_headers": dict(profile["extra_headers"]) if isinstance(profile.get("extra_headers"), dict) else None,
        "default_headers": dict(profile["default_headers"]) if isinstance(profile.get("default_headers"), dict) else None,
        "fallback_models": fallbacks,
        "max_api_retry": int(max_api_retry) if max_api_retry is not None else 2,
        **_sibling_model_fields(
            profile.get("auxiliary_model"), "auxiliary", provider, base_url, api_key,
        ),
        **_sibling_model_fields(
            profile.get("auto_review_model"), "auto_review", provider, base_url, api_key,
        ),
    }


# Timeout in seconds for building a new Agent instance (guards against SDK hangs)
_AGENT_BUILD_TIMEOUT_S = 30

_TITLE_PROMPT = (
    "Name this conversation for a sidebar list. Reply with ONLY the title: "
    "at most 8 words (or 16 CJK characters), in the user's language, no quotes, "
    "no trailing punctuation."
)

# Agent cache, run lock, work_dir and approval mode are partitioned by
# owner. ``session_id`` is client-supplied; a bare key lets one account
# evict another's cached Agent / lock / live run.
_OWNER_SEP = "\x1f"


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

    def __init__(self, max_size: int = 50, on_evict: Optional[Callable[[str], None]] = None):
        self._cache: OrderedDict[str, DeepAgent] = OrderedDict()
        self.max_size = max_size
        self.on_evict = on_evict

    def _release(self, key: str) -> None:
        if self.on_evict is not None:
            self.on_evict(key)

    def get(self, session_id: str) -> Optional[DeepAgent]:
        if session_id not in self._cache:
            return None
        self._cache.move_to_end(session_id)
        return self._cache[session_id]

    def put(self, session_id: str, agent: DeepAgent) -> None:
        if session_id in self._cache:
            # A rebuild reuses the same daemon session name, so its browser
            # tool must stay registered instead of being closed here.
            self._cache.move_to_end(session_id)
            self._cache[session_id] = agent
            return
        self._cache[session_id] = agent
        while len(self._cache) > self.max_size:
            evicted = False
            for key in list(self._cache.keys()):
                if _cache_key_is_pinned(key):
                    continue
                self._cache.pop(key)
                self._release(key)
                logger.debug(f"LRU evicted agent for session: {key}")
                evicted = True
                break
            if not evicted:
                break

    def delete(self, session_id: str) -> bool:
        if self._cache.pop(session_id, None) is None:
            return False
        self._release(session_id)
        return True

    def clear(self) -> None:
        keys = list(self._cache.keys())
        self._cache.clear()
        for key in keys:
            self._release(key)

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
