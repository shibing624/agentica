# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner compression stages (tool-budget, micro, rule, auto-compact)
"""

# This file was mechanically extracted from /tmp/runner_backup.py for compress.py.

import time
from typing import (
    Any,
    Dict,
    List,
    TYPE_CHECKING,
)


from agentica.utils.log import logger
from agentica.compression.evict import evict_tool_results
from agentica.compression.tool_result_storage import enforce_tool_result_budget
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
        """Run the multi-stage compression pipeline before each LLM call.

        Stages ordered to preserve provider-native state when available:
          Stage 1 - Tool result budget (free, O(n))
          Stage 2 - Provider-native compact (Responses API)
          Stage 3 - Tool-result eviction (free, O(n))
          Stage 4 - Rule-based compress (free, O(n))
          Stage 5 - Auto-compact (costly, portable LLM summarisation)
          Stage 6 (reactive compact) is handled in _call_with_retry on API error.

        Sets ``loop_state.context_collapsed`` whenever a stage drops messages,
        so the caller knows the ``num_input_messages`` prefix boundary is gone.
        """
        cb = agent._event_callback
        agent_name = agent.name or "Agent"
        is_main_agent = agent._parent_run_id is None
        # Stage 1: tool result budget (persist oversized results to disk)
        _sid = agent.run_id or "default"
        _uid = agent.workspace.user_id if agent.workspace is not None else None
        _recent_tools = [m for m in messages if m.role == "tool" and not m.compressed_content]
        if _recent_tools:
            enforce_tool_result_budget(
                tool_results=_recent_tools,
                session_id=_sid,
                user_id=_uid,
            )

        compression_enabled = agent.tool_config.compress_tool_results
        cm = agent.tool_config.compression_manager

        async def _fire_compact_hooks(event: str) -> None:
            if agent._run_hooks is not None:
                fn = getattr(agent._run_hooks, event, None)
                if fn is not None:
                    await fn(agent=agent, messages=messages)

        # Stage 2: provider-native compact. A successful checkpoint leaves the
        # portable transcript untouched, so cross-provider fallback remains
        # possible while subsequent Responses calls use the smaller context.
        if compression_enabled and cm is not None and cm.should_native_compact(
            messages, model, tools=model.tools
        ):
            before_tokens = model.estimate_native_compaction_tokens(messages, model.tools)
            t0 = time.monotonic()
            await _fire_compact_hooks("on_pre_compact")
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
                if agent.run_response is not None:
                    agent.run_response.metrics = agent.run_response.metrics or {}
                    compression_metrics = agent.run_response.metrics.setdefault("compression", {})
                    compression_metrics["native"] = {
                        "model": model.id,
                        "input_tokens_before": before_tokens,
                        "usage": result.usage,
                    }
                await _fire_compact_hooks("on_post_compact")
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

        # Stage 3: evict old tool results (free). Gated on real context
        # pressure: below the threshold there is nothing to buy by dropping a
        # result the window had room for, and the model pays for it by
        # re-running the tool.
        _window = model.context_window if isinstance(model.context_window, int) else 0
        n = evict_tool_results(
            messages,
            context_tokens=count_tokens(messages, model.tools, model.id) if _window else 0,
            context_window=_window,
            model_id=model.id,
        )
        if n:
            logger.debug(f"Stage 3 (evict): evicted {n} old tool result(s)")
            if cb is not None:
                cb(
                    {
                        "type": "compact.evict",
                        "agent_name": agent_name,
                        "is_main_agent": is_main_agent,
                        "evicted": n,
                    }
                )

        # Remaining stages require CompressionManager.
        if not compression_enabled:
            return
        if cm is None:
            return

        # Stage 4: rule-based compress (truncate + drop old rounds, free)
        if cm.should_compress(messages, tools=model.tools, model=model):
            await _fire_compact_hooks("on_pre_compact")
            logger.debug("Stage 4 (rule-based compress): truncating + dropping old messages")
            before = len(messages)
            t0 = time.monotonic()
            await cm.compress(
                messages,
                tools=model.tools,
                model=model,
                trigger="threshold",
                task_anchor=agent.task_anchor,
                user_id=_uid,
            )
            if len(messages) < before:
                loop_state.context_collapsed = True
            compression_report = cm.get_stats().get("last_report")
            if compression_report and agent.run_response is not None:
                agent.run_response.metrics = agent.run_response.metrics or {}
                agent.run_response.metrics["compression"] = {"last_report": compression_report}
            await _fire_compact_hooks("on_post_compact")
            if cb is not None:
                cb(
                    {
                        "type": "compact.rule_based",
                        "agent_name": agent_name,
                        "is_main_agent": is_main_agent,
                        "before": before,
                        "after": len(messages),
                        "elapsed": time.monotonic() - t0,
                        "report": compression_report,
                    }
                )

        # Stage 5: auto-compact via LLM summarisation.
        # auto_compact() returns False fast when threshold not met; only fire
        # events when it actually compresses (avoids per-turn spam).
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(messages, model=model)
        if compacted:
            loop_state.context_collapsed = True
            logger.debug("Stage 5 (auto-compact): conversation summarised by LLM")
            await _fire_compact_hooks("on_post_compact")
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
    async def _try_reactive_compact(
        messages: List[Message],
        agent: "Agent",
        model: "Model",
    ) -> bool:
        """Attempt emergency compression on prompt_too_long. Returns True if compacted.

        Stage 5 (reactive compact) is handled in _call_with_retry on API error.
        """
        cm = agent.tool_config.compression_manager if agent is not None else None
        if cm is None:
            return False
        before = len(messages)
        t0 = time.monotonic()
        compacted = await cm.auto_compact(messages, model=model, force=True)
        if compacted:
            logger.info("Reactive compact triggered (prompt_too_long) -- retrying")
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
        """
        cb = agent._event_callback
        if cb is None:
            return
        tools = model.tools if isinstance(model.tools, list) else []
        window = model.context_window if isinstance(model.context_window, int) else 0
        try:
            cb(
                {
                    "type": "context.usage",
                    "agent_name": agent.name or "Agent",
                    "is_main_agent": agent._parent_run_id is None,
                    "context_tokens": count_tokens(messages, tools, model.id),
                    "context_window": window,
                }
            )
        except Exception as e:
            logger.warning(f"event callback failed for context.usage: {e}")

