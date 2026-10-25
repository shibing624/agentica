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
from agentica.compression.calibration import ContextCalibration
from agentica.compression.evict import evict_context, under_pressure
from agentica.compression.manager import compact_token_limit_of, working_context_window
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
          Layer 2 - New window (no LLM summary): replace the activity window
                    with initial context + notes. ``/compact`` uses the same
                    cut. Prior turns stay in JSONL.

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
        enable_evict = agent.tool_config.enable_evict
        enable_auto_compact = agent.tool_config.enable_auto_compact

        # Layer 1: evict (free). Gated on real context pressure: below the
        # threshold there is nothing to buy by dropping a result the window had
        # room for, and the model pays for it by re-running the tool.
        # compact_token_limit is a working cap on the same layers, not a lie
        # about model.context_window — Layer 0's batch budget reads the same
        # working window, so all three layers measure pressure against one number.
        _window = model.context_window if isinstance(model.context_window, int) else 0
        _cap = None
        if cm is not None:
            _cap = cm.compact_token_limit
        elif agent is not None:
            _cap = agent.tool_config.compact_token_limit
        _working = working_context_window(_window, _cap)
        calibration = CompressMixin._calibration(model)
        calibration.observe(messages)
        # Raw = local tiktoken estimate; context_tokens = the same scaled to
        # what the provider will count. Thresholds read the scaled figure.
        raw_tokens = count_tokens(messages, model.tools, model.id) if _working else 0
        context_tokens = calibration.scale(raw_tokens)
        if enable_evict:
            # Eviction measures what each placeholder saves with the local
            # tokenizer, so it works in raw units against a window shrunk by
            # the same ratio.
            evict_window = int(_working / calibration.ratio)
            # A placeholder may only point at the session log when the model
            # can both have one written and read it back.
            recoverable = (
                agent._session_log is not None
                and "search_session" in (model.functions or {})
            )
            digests_before = (
                _prefix_digests(messages) if under_pressure(raw_tokens, evict_window) else None
            )
            reclaimed = evict_context(
                messages,
                context_tokens=raw_tokens,
                context_window=evict_window,
                model_id=model.id,
                recoverable=recoverable,
            )
            if reclaimed.total:
                # Eviction rewrote content, so the count taken above is stale — and
                # too high. Re-measure before deciding on Layer 2: an LLM summary
                # bought for a request eviction already made fit is pure waste.
                tokens_before = raw_tokens
                raw_tokens = count_tokens(messages, model.tools, model.id) if _working else 0
                context_tokens = calibration.scale(raw_tokens)
                # What this eviction cost the provider cache: every message from
                # the first rewritten one onward has to be written again.
                break_index = _first_prefix_break(digests_before, _prefix_digests(messages))
                invalidated_tokens = (
                    count_tokens(messages[break_index:], None, model.id)
                    if break_index is not None else 0
                )
                logger.debug(
                    f"Layer 1 (evict): {reclaimed.tool_results} tool result(s), "
                    f"{reclaimed.tool_call_args} tool-call argument(s), "
                    f"freed {tokens_before - raw_tokens} tokens, "
                    f"prefix break at message {break_index} "
                    f"({invalidated_tokens} tokens to re-cache)"
                )
                if cb is not None:
                    cb(
                        {
                            "type": "compact.evict",
                            "agent_name": agent_name,
                            "is_main_agent": is_main_agent,
                            "evicted": reclaimed.tool_results,
                            "shrunk": reclaimed.tool_call_args,
                            "freed_tokens": tokens_before - raw_tokens,
                            "break_index": break_index,
                            "invalidated_tokens": invalidated_tokens,
                            "tokens_after": raw_tokens,
                        }
                    )

        if not enable_auto_compact or cm is None:
            calibration.record_send(messages, raw_tokens)
            return

        CompressMixin._maybe_inject_token_budget_reminder(
            messages, agent, cm, context_tokens, _working,
        )

        # Layer 2: new window. Threshold checked here so on_pre_compact
        # (memory flush) only runs on turns that actually roll over.
        if not cm.should_auto_compact(
            messages, model, context_tokens=context_tokens if _window else None
        ):
            calibration.record_send(messages, raw_tokens)
            return
        await CompressMixin._install_new_window(
            messages, agent, model, loop_state, keep_trailing_turn=True,
        )
        # The window was replaced: the estimate for the request now going out
        # has to be taken again.
        calibration.record_send(messages, count_tokens(messages, model.tools, model.id))

    @staticmethod
    def _calibration(model: "Model") -> ContextCalibration:
        if model._context_calibration is None:
            model._context_calibration = ContextCalibration()
        return model._context_calibration

    @staticmethod
    async def _install_new_window(
        messages: List[Message],
        agent: "Agent",
        model: "Model",
        loop_state: "LoopState",
        *,
        keep_trailing_turn: bool,
    ) -> bool:
        """Layer 2 new-window cut, session notes written by the model first."""
        cm = agent.tool_config.compression_manager
        if cm is None:
            return False
        cb = agent._event_callback
        before = len(messages)
        t0 = time.monotonic()
        await CompressMixin._fire_pre_compact(agent, messages)
        # The notes pass is a full model request; the spinner says
        # "compacting" for it instead of a turn that looks hung.
        if cb is not None:
            cb({"type": "compact.start", "agent_name": agent.name or "Agent"})
        try:
            compacted = await cm.auto_compact(
                messages,
                model=model,
                force=True,
                keep_trailing_turn=keep_trailing_turn,
                notes_request=messages,
            )
        finally:
            if cb is not None:
                cb({"type": "compact.end", "agent_name": agent.name or "Agent"})
        if not compacted:
            return False
        loop_state.context_collapsed = True
        CompressMixin._note_compaction(agent)
        await CompressMixin._fire_post_compact(agent, messages)
        if cb is not None:
            cb(
                {
                    "type": "compact.auto",
                    "agent_name": agent.name or "Agent",
                    "is_main_agent": agent._parent_run_id is None,
                    "before": before,
                    "after": len(messages),
                    "elapsed": time.monotonic() - t0,
                    "window_id": cm.window_id,
                    "new_window": True,
                    "notes_authored": cm.last_notes_authored,
                }
            )
        return True

    @staticmethod
    def _maybe_inject_token_budget_reminder(
        messages: List[Message],
        agent: "Agent",
        cm: Any,
        context_tokens: int,
        working_window: int,
    ) -> None:
        """Once-per-window remaining-token notice (Codex post-#27438 reminder)."""
        from agentica.compression.token_budget import (
            remaining_text,
            reminder_threshold,
            tokens_remaining,
        )

        if cm.reminder_claimed or working_window <= 0:
            return
        left = tokens_remaining(context_tokens, working_window)
        if left > reminder_threshold(working_window):
            return
        last = messages[-1] if messages else None
        if last is None:
            return
        if last.role == "assistant" and last.tool_calls:
            return
        text = remaining_text(left)
        if CompressMixin._fold_budget_fragment(messages, text):
            cm.reminder_claimed = True

    @staticmethod
    def _fold_budget_fragment(messages: List[Message], text: str) -> bool:
        from agentica.compression.evict import carries_tool_results, tool_result_blocks
        from agentica.compression.token_budget import is_pending_window_preamble

        if not messages:
            return False
        m = messages[-1]
        if carries_tool_results(m):
            if m.role == "tool" and isinstance(m.content, str):
                m.content = f"{m.content}\n\n{text}" if m.content else text
                return True
            blocks = tool_result_blocks(m)
            if blocks:
                last_block = blocks[-1]
                prev = last_block.get("content", "")
                if isinstance(prev, str):
                    last_block["content"] = f"{prev}\n\n{text}" if prev else text
                    return True
            return False
        if m.role == "user" and isinstance(m.content, str):
            if is_pending_window_preamble(m):
                return False
            m.content = f"{m.content}\n\n{text}" if m.content else text
            return True
        return False

    @staticmethod
    def _note_compaction(agent: "Agent") -> None:
        """Record on the run that the activity window was reset.

        Without this the only witness is the CLI event callback, so an SDK
        caller sees a turn that quietly dropped early transcript from the
        prompt, with nothing to attribute it to.
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
        if cm is None or not agent.tool_config.enable_auto_compact:
            return False
        await CompressMixin._fire_pre_compact(agent, messages)
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(
            messages, model=model, force=True, keep_trailing_turn=True,
        )
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
        provider_window = model.context_window if isinstance(model.context_window, int) else 0
        # The bar's denominator is the working window the compression policy
        # acts on. Emitting only the provider limit made the bar divide by 1M
        # while the runner evicted against a 512k cap on the same session.
        cap = compact_token_limit_of(agent.tool_config) if agent.tool_config else None
        window = working_context_window(provider_window, cap)

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
                    "context_tokens": CompressMixin._calibration(model).scale(
                        count_tokens(messages, tools, model.id)
                    ),
                    "context_window": window,
                    "provider_window": provider_window,
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

