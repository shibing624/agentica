# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Chat, streaming, goal runs, compaction, usage, and steering on AgentService.
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from agentica.compression.manager import (
    apply_idle_compact,
    compact_token_limit_of,
    working_context_window,
)
from agentica.run.context import RunSource
from agentica.run.display import RunDisplayEventKind, classify_run_response
from agentica.run.response import AgentCancelledError
from agentica.utils.log import logger
from agentica.utils.string import replace_invalid_utf8

from .types import (
    ChatResult,
    _read_compact_notes,
    dispatch_stream_chunk,
    goal_event_payload,
)

if TYPE_CHECKING:
    from ...channels.base import InboundMedia


def _pkg():
    return sys.modules["agentica_gateway.services.agent_service"]


class AgentExecutionMixin:
    """chat, chat_stream, run_goal, compact_session, session_usage, cancel, and steer."""

    async def session_usage(
        self,
        session_id: str,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """CLI ``/usage`` for one web session: occupancy + session billing."""
        await self._ensure_initialized()
        agent = await self._get_agent(session_id, owner)
        payload = await _pkg().usage_payload(agent, model_provider=self._session_provider(session_id, owner))
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
        """Return the context window size for the model used by a session."""
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
        """Turn a ``/skill-trigger [arguments]`` message into the skill prompt."""
        expanded = _pkg().get_skill_registry().expand_invocation(message)
        return expanded if expanded is not None else message

    @staticmethod
    async def _prepare_run_media(agent: Any, message: str, media: Optional[List["InboundMedia"]]):
        """Route inbound images/audio/video the same way IM channels do."""
        if not media:
            return message, None, None, []
        plan = await _pkg().media_understanding.prepare(
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
        """Send a message and return the full response (non-streaming)."""
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

                pkg = _pkg()
                return ChatResult(
                    content=content,
                    tool_calls=tool_calls,
                    session_id=session_id,
                    user_id=user_id,
                    tools_used=tools_used,
                    metrics=pkg.extract_metrics(agent),
                    media_notes=media_notes,
                    usage=await pkg.usage_payload(agent, model_provider=self._session_provider(session_id, owner)),
                    turn_usage=pkg.turn_usage_payload(agent),
                )

            except Exception as e:
                logger.error(f"AgentService.chat error (session={session_id}): {e}")
                return ChatResult(
                    content=f"Error: {e}",
                    tool_calls=0,
                    session_id=session_id,
                    user_id=user_id,
                    media_notes=media_notes,
                    error=str(e),
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
        """Drive a standing-goal loop (Agent.run_goal) for the web UI's ``/goal``."""
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
            data["event"] = event.value
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
        """Start a new context window — the web counterpart of CLI ``/compact``."""
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
                "notes": notes,
                "notes_path": notes_path,
                "usage": await _pkg().usage_payload(agent, model_provider=self._session_provider(session_id, owner)),
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
        input_meta: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Send a message and stream the response via callbacks."""
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
                automated=automated, input_meta=input_meta,
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
        input_meta: Optional[Dict[str, Any]] = None,
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
                    input_meta=input_meta,
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

            if not automated:
                self._schedule_title(session_id, owner, message, full_content)
            pkg = _pkg()
            usage = (
                await pkg.usage_payload(agent, model_provider=self._session_provider(session_id, owner))
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
                metrics=pkg.extract_metrics(agent),
                media_notes=media_notes,
                usage=usage,
                turn_usage=pkg.turn_usage_payload(agent),
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
                error=str(e),
            )

    def cancel_session(self, session_id: str, owner: Optional[str] = None) -> bool:
        """Cancel the in-flight run for a specific session."""
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
        """Cached agent for this session, only if it belongs to ``owner``."""
        agent = self._cache.get(self._sk(session_id, owner))
        if agent is None or agent.user_id != self._owner(owner):
            return None
        return agent

    def steer_session(self, session_id: str, guidance: str, owner: Optional[str] = None) -> bool:
        """Inject mid-run guidance (CLI ``agent.steer``) for a web session."""
        agent = self._cached_agent_for_owner(session_id, owner)
        if agent is None:
            return False
        accepted = bool(agent.steer(self.expand_skill_invocation(guidance)))
        if accepted:
            turn = _pkg().live_turn.active(session_id, self._owner(owner))
            if turn is not None:
                turn.publish({"event": "steer", "data": {"text": guidance, "ts": int(time.time() * 1000)}})
        return accepted

    def take_undelivered_steer(self, session_id: str, owner: Optional[str] = None) -> List[str]:
        """Pop steering accepted too late for this run, so the web UI can queue it."""
        agent = self._cached_agent_for_owner(session_id, owner)
        if agent is None:
            return []
        return [item[0] for item in agent.pop_undelivered_steer()]
