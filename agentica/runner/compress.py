# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner compression pipeline (tool-budget, evict, auto-compact)
"""

# This file was mechanically extracted from /tmp/runner_backup.py for compress.py.

import json
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)


from agentica.utils.log import logger
from agentica.compression.evict import evict_context
from agentica.model.base import Model
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.utils.tokens import count_tokens

if TYPE_CHECKING:
    from agentica.agent import Agent

from agentica.runner.persist import PersistMixin


class CompressMixin:
    """Extracted Runner methods."""

    @staticmethod
    def _compact_fallback_transaction(
        messages: List[Message],
        marker: Message,
        model_response: ModelResponse,
        used_model: "Model",
    ) -> List[Dict[str, Any]]:
        """Hide provider-specific fallback tool transcript from future model replay.

        ``marker`` is the first message the fallback produced. We resolve its
        live index by object identity so a compression pass that dropped/moved
        messages between transaction start and now can't make us fold the wrong
        segment: if the marker is gone we skip compaction rather than corrupt
        the transcript.
        """
        start = next((i for i, m in enumerate(messages) if m is marker), -1)
        if start < 0:
            return []
        segment = messages[start:]
        audit_tools = PersistMixin._tool_records_from_messages(
            segment,
            fallback_compacted=True,
            fallback_model=used_model.id,
        )
        final_content = model_response.content
        if final_content is None:
            for msg in reversed(segment):
                if msg.role == "assistant" and not msg.tool_calls and msg.content is not None:
                    final_content = msg.content
                    break
        compacted = Message(
            role="assistant",
            content=final_content or "",
            reasoning_content=model_response.reasoning_content,
            finish_reason=model_response.finish_reason,
            provider_data={
                "fallback_compacted": True,
                "fallback_model": used_model.id,
            },
        )
        messages[start:] = [compacted]
        return audit_tools

    @staticmethod
    async def _maybe_compress_messages(
        messages: List[Message],
        agent: "Agent",
        model: "Model",
        loop_state: "LoopState",
    ) -> None:
        """Run the compression pipeline before each LLM call.

        There are only two ways to make an oversized request fit, and they are
        tried cheapest-first:

          Layer 1 - Evict (free): drop the bulk of old tool results and shrink
                    oversized tool-call arguments, oldest-first, down to a
                    target. Nothing is dropped while the window has room.
          Layer 2 - Summarise (costly, irreversible): replace the history with
                    an LLM summary. Provider-native compaction is the same
                    layer done server-side, so it is tried first when the model
                    supports it; the reactive path in ``_call_with_retry``
                    forces this layer after a ``prompt_too_long`` rejection.

        Ahead of both is Layer 0, which is not compression but an output
        policy and does not run here: it bounds a result the moment it is
        produced (``Model.run_function_calls``), so nothing enters the context
        at full size in the first place. It used to run a second sweep from
        here over the whole transcript against a fixed char budget — a second
        threshold governing the same decision Layer 1 owns, firing on history
        that the window had ample room for.

        Sets ``loop_state.context_collapsed`` whenever a stage drops messages,
        so the caller knows the ``num_input_messages`` prefix boundary is gone.
        """
        cb = agent._event_callback
        agent_name = agent.name or "Agent"
        is_main_agent = agent._parent_run_id is None

        cm = agent.tool_config.compression_manager

        # Layer 2, provider-native variant. Tried before the local layers
        # because a successful checkpoint leaves the portable transcript
        # untouched, so cross-provider fallback remains possible while
        # subsequent Responses calls use the smaller context.
        if cm is not None and cm.should_native_compact(
            messages, model, tools=model.tools
        ):
            before_tokens = model.estimate_native_compaction_tokens(messages, model.tools)
            t0 = time.monotonic()
            await CompressMixin._fire_pre_compact(agent, messages)
            try:
                result = await model.compact_context(messages)
                if result is None:
                    raise RuntimeError("model advertised native compaction but returned no checkpoint")
            except Exception as error:
                logger.warning(
                    f"Native compact failed for {model.id}; falling back to local compression: {error}"
                )
                if cb is not None:
                    cb(
                        {
                            "type": "compact.native_failed",
                            "agent_name": agent_name,
                            "is_main_agent": is_main_agent,
                            "model": model.id,
                            "error": str(error),
                            "elapsed": time.monotonic() - t0,
                        }
                    )
            else:
                messages[-1].provider_checkpoint = result.checkpoint
                logger.info(f"Native compact complete for {model.id}")
                CompressMixin._note_compaction(agent)
                if agent.run_response is not None:
                    agent.run_response.metrics = agent.run_response.metrics or {}
                    compression_metrics = agent.run_response.metrics.setdefault("compression", {})
                    compression_metrics["native"] = {
                        "model": model.id,
                        "input_tokens_before": before_tokens,
                        "usage": result.usage,
                    }
                await CompressMixin._fire_post_compact(agent, messages)
                if cb is not None:
                    cb(
                        {
                            "type": "compact.native",
                            "agent_name": agent_name,
                            "is_main_agent": is_main_agent,
                            "model": model.id,
                            "input_tokens_before": before_tokens,
                            "usage": result.usage,
                            "elapsed": time.monotonic() - t0,
                        }
                    )
                return

        # Layer 1: evict (free). Gated on real context pressure: below the
        # threshold there is nothing to buy by dropping a result the window had
        # room for, and the model pays for it by re-running the tool.
        _window = model.context_window if isinstance(model.context_window, int) else 0
        context_tokens = count_tokens(messages, model.tools, model.id) if _window else 0
        reclaimed = evict_context(
            messages,
            context_tokens=context_tokens,
            context_window=_window,
            model_id=model.id,
        )
        if reclaimed.total:
            logger.debug(
                f"Layer 1 (evict): {reclaimed.tool_results} tool result(s), "
                f"{reclaimed.tool_call_args} tool-call argument(s)"
            )
            if cb is not None:
                cb(
                    {
                        "type": "compact.evict",
                        "agent_name": agent_name,
                        "is_main_agent": is_main_agent,
                        "evicted": reclaimed.tool_results,
                        "shrunk": reclaimed.tool_call_args,
                    }
                )
            # Eviction rewrote content, so the count taken above is stale — and
            # too high. Re-measure before deciding on Layer 2: an LLM summary
            # bought for a request eviction already made fit is pure waste.
            context_tokens = count_tokens(messages, model.tools, model.id) if _window else 0

        if cm is None:
            return

        # Layer 2: LLM summarisation. The threshold is checked *here* rather
        # than inside auto_compact() because on_pre_compact flushes memory and
        # experience buffers through an auxiliary LLM: firing it on every turn
        # would turn a once-per-many-rounds boundary into a per-turn cost.
        # Deciding once and forcing keeps the gate and the compaction from
        # disagreeing about the same number.
        if not cm.should_auto_compact(
            messages, model, context_tokens=context_tokens if _window else None
        ):
            return
        await CompressMixin._fire_pre_compact(agent, messages)
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(messages, model=model, force=True)
        if compacted:
            loop_state.context_collapsed = True
            CompressMixin._note_compaction(agent)
            logger.debug("Layer 2 (auto-compact): conversation summarised by LLM")
            await CompressMixin._fire_post_compact(agent, messages)
            if cb is not None:
                cb(
                    {
                        "type": "compact.auto",
                        "agent_name": agent_name,
                        "is_main_agent": is_main_agent,
                        "before": before,
                        "after": len(messages),
                        "elapsed": time.monotonic() - t0,
                    }
                )

    @staticmethod
    def _note_compaction(agent: "Agent") -> None:
        """Record on the run that history was summarised.

        Without this the only witness is the CLI event callback, so an SDK
        caller sees a turn that was slow, cost extra, and quietly lost the
        early transcript, with nothing to attribute it to.
        """
        if agent.run_response is not None:
            agent.run_response.context_compactions += 1

    @staticmethod
    async def _fire_pre_compact(agent: "Agent", messages: List[Message]) -> None:
        """Data-loss boundary: buffered memories/experience must be extracted
        from the transcript before a summary replaces it."""
        if agent._run_hooks is not None:
            await agent._run_hooks.on_pre_compact(agent=agent, messages=messages)

    @staticmethod
    async def _fire_post_compact(agent: "Agent", messages: List[Message]) -> None:
        """Fired by every path that actually compacts — native, Layer 2 and
        reactive — so a hook cannot see a compaction start and never finish."""
        if agent._run_hooks is not None:
            await agent._run_hooks.on_post_compact(agent=agent, messages=messages)

    @staticmethod
    async def _try_reactive_compact(
        messages: List[Message],
        agent: "Agent",
        model: "Model",
    ) -> bool:
        """Force Layer 2 after a prompt_too_long rejection. True if compacted."""
        cm = agent.tool_config.compression_manager if agent is not None else None
        if cm is None:
            return False
        await CompressMixin._fire_pre_compact(agent, messages)
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(messages, model=model, force=True)
        if compacted:
            CompressMixin._note_compaction(agent)
            logger.info("Reactive compact triggered (prompt_too_long) -- retrying")
            await CompressMixin._fire_post_compact(agent, messages)
            cb = agent._event_callback
            if cb is not None:
                cb(
                    {
                        "type": "compact.reactive",
                        "agent_name": agent.name or "Agent",
                        "is_main_agent": agent._parent_run_id is None,
                        "before": before,
                        "after": len(messages),
                        "elapsed": time.monotonic() - t0,
                    }
                )
            return True
        return False

    @staticmethod
    def _emit_context_usage(agent: "Agent", model: "Model", messages: List[Message]) -> None:
        """Expose the context carried by an actual model request.

        Context occupancy is runtime state, not cost accounting. Emitting it at
        the request boundary keeps summarisation and other auxiliary LLM calls
        from polluting the main session's status-bar value.

        Also the per-request cache observability point: attaches the previous
        request's hit ratio (current request has no numbers yet) and the
        prefix-break index against the previous request's message digests —
        the two signals needed to spot cache pollution as it happens.
        """
        cb = agent._event_callback
        if cb is None:
            return
        tools = model.tools if isinstance(model.tools, list) else []
        window = model.context_window if isinstance(model.context_window, int) else 0

        prev_digests = getattr(model, "_last_prefix_digests", None)
        digests = _prefix_digests(messages)
        break_index = _first_prefix_break(prev_digests, digests) if prev_digests else None
        model._last_prefix_digests = digests

        hit_ratio = None
        usage = getattr(model, "usage", None)
        if usage is not None and usage.request_usage_entries:
            hit_ratio = usage.request_usage_entries[-1].cache_hit_ratio()

        try:
            cb(
                {
                    "type": "context.usage",
                    "agent_name": agent.name or "Agent",
                    "is_main_agent": agent._parent_run_id is None,
                    "context_tokens": count_tokens(messages, tools, model.id),
                    "context_window": window,
                    "cache_hit_ratio": hit_ratio,
                    "prefix_break_index": break_index,
                }
            )
        except Exception as e:
            logger.warning(f"event callback failed for context.usage: {e}")


def _prefix_digests(messages: List[Message]) -> List[str]:
    """One digest per request-bound message (role + content + tool_calls).

    Local-only fields are intentionally ignored: this is a prefix-stability
    probe for the provider cache, not a transcript checksum, so fields the
    model never sees must not count as differences.
    """
    import hashlib

    def _norm(value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
        except Exception:
            return str(value)

    digests: List[str] = []
    for m in messages:
        raw = f"{m.role}\x00{_norm(m.content)}\x00{_norm(m.tool_calls or '')}"
        digests.append(hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:10])
    return digests


def _first_prefix_break(prev: List[str], curr: List[str]) -> Optional[int]:
    """Index of the oldest message whose digest changed; None when curr is
    prev plus appended tail (the append-only fast path the provider cache is
    built around). A shrink counts as a break at the cut point."""
    for i, d in enumerate(prev):
        if i >= len(curr):
            return i
        if curr[i] != d:
            return i
    return None if len(curr) >= len(prev) else len(prev)

