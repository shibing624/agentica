# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner model-call retry and fallback recovery
"""

import asyncio
import copy
import random
from typing import (
    List,
    Optional,
    TYPE_CHECKING,
)


from agentica.compression.evict import is_irreducible_prompt_too_long
from agentica.utils.log import logger
from agentica.model.base import Model
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.model.usage import Usage
from agentica.run_config import RunConfig

if TYPE_CHECKING:
    from agentica.agent import Agent

from agentica.runner.types import ModelCallResult
from agentica.runner.compress import CompressMixin
from agentica.runner.persist import PersistMixin


class RetryMixin:
    """Extracted Runner methods."""

    async def _recover_with_fallback(
        self,
        messages: List[Message],
        loop_state: "LoopState",
        agent: "Agent",
        break_reason: str,
    ) -> Optional[ModelResponse]:
        """After a loop break, do ONE tool-free inference with the fallback chain.

        Opt-in via ``Agent.fallback_on_break`` / ``RunConfig.fallback_on_break``.
        The full message history (including the failed tool calls) is replayed so
        the fallback model can see what went wrong — no synthetic hint is injected.
        Fallback models carry no bound tools (``update_model`` only binds tools to
        the primary), so the reply is naturally tool-free.

        Returns the recovery ModelResponse (its assistant message is appended to
        ``messages``), or None when break-recovery is disabled, no fallback is
        configured, or every fallback failed.
        """
        if not agent._run_fallback_on_break or not agent._run_fallback_models:
            return None

        fb_chain = list(agent._run_fallback_models)
        primary_id = agent.model.id if agent.model else "?"

        # Reuse the retry/fallback machinery: treat the first fallback as the
        # "primary" of this recovery call and the rest as its own fallbacks.
        saved_chain = agent._run_fallback_models
        agent._run_fallback_models = fb_chain[1:]
        try:
            recovery_call = await self._call_with_retry(
                fb_chain[0],
                messages,
                loop_state,
                agent,
                stream=False,
            )
            recovery = recovery_call.response
        except Exception as exc:
            logger.error(
                f"[fallback.on_break] recovery inference failed (reason={break_reason}, primary={primary_id}): {exc}"
            )
            return None
        finally:
            agent._run_fallback_models = saved_chain

        used_id = loop_state.last_used_model_id or fb_chain[0].id
        logger.warning(f"[fallback.on_break] recovered: primary={primary_id} -> used={used_id} (reason={break_reason})")
        cb = agent._event_callback
        if cb is not None:
            try:
                cb(
                    {
                        "type": "fallback.on_break",
                        "agent_name": agent.name or "Agent",
                        "primary_model": primary_id,
                        "used_model": used_id,
                        "break_reason": break_reason,
                    }
                )
            except Exception as e:
                logger.warning(f"event callback failed for fallback.on_break: {e}")
        return recovery

    @staticmethod
    def _prepare_model_for_runner_call(current: "Model", primary: "Model") -> None:
        """Bind runner-owned tool context to the concrete model used for this call."""
        current.run_tools = False
        if current is primary:
            return
        current.tools = primary.tools
        current.functions = primary.functions
        current.tool_choice = primary.tool_choice
        current.tool_call_limit = primary.tool_call_limit
        current.max_concurrent_tools = primary.max_concurrent_tools
        current._cost_tracker = primary._cost_tracker

    @staticmethod
    def _isolate_fallback_models(models: List["Model"]) -> List["Model"]:
        """Give this run its own fallback model instances.

        A fallback model is commonly shared across agents/runs (one backup model
        handed to many agents). Since ``_prepare_model_for_runner_call`` rebinds
        tools/cost_tracker onto the concrete model used for a call, two concurrent
        runs sharing the same fallback object would stomp on each other. Shallow-
        clone per run (the canonical Model clone), reset runtime state, and drop
        client refs so each run owns an isolated instance bound to its own loop.
        """
        isolated: List["Model"] = []
        for source in models:
            # Only real Model instances are clonable. Anything else (e.g. test
            # doubles) is passed through untouched so it keeps its identity.
            if not isinstance(source, Model):
                isolated.append(source)
                continue
            cloned = copy.copy(source)
            cloned.metrics = {}
            cloned.usage = Usage()
            for attr in ("client", "http_client", "async_client"):
                if hasattr(cloned, attr):
                    setattr(cloned, attr, None)
            isolated.append(cloned)
        return isolated

    @staticmethod
    def _resolve_max_api_retry(agent: "Agent", config: RunConfig) -> int:
        """Resolve and validate Runner-level API call attempts for one run."""
        value = config.max_api_retry if config.max_api_retry is not None else agent.max_api_retry
        if value < 1:
            raise ValueError("max_api_retry must be >= 1; use 1 to disable Runner-level same-model retry")
        return value

    @staticmethod
    async def _call_with_retry(
        model: "Model",
        messages: List[Message],
        state: "LoopState",
        agent: "Agent",
        *,
        stream: bool = False,
    ):
        """Call model.response()/response_stream() with retry, reactive compact,
        and cross-provider fallback chain.

        Returns:
            ModelCallResult(response=ModelResponse, used_model=...) for non-stream.
            ModelCallResult(response=async iterator, used_model=...) for stream.

        Per-call fallback (not per-run): each invocation starts from the primary
        ``model`` argument; ``agent._run_fallback_models`` are tried in order
        when triggered by:

          1. ``finish_reason in CONTENT_FILTER_FINISH_REASONS`` (non-stream only;
             stream cannot detect this until consumed).
          2. Exception whose text matches ``CONTENT_FILTER_HINTS`` (any provider
             that raises instead of flagging a finish reason).
          3. Fallback-only API error (``FALLBACK_ONLY_SUBSTRINGS``), such as
             connection failure / 502 / 503 / bad gateway.
          4. Retryable API error (``RETRYABLE_SUBSTRINGS``) that exhausted local
             exponential backoff on the current model.

        Reactive compact (``prompt_too_long``) is NOT a fallback trigger —
        switching providers does not solve a too-long context.
        """
        # Dedup by object identity, not ``model.id``: two distinct instances may
        # share an id but differ in api_key/base_url (a legitimate second
        # fallback). Identity only drops the same object listed twice (e.g. the
        # primary also present in the fallback chain).
        candidates: List["Model"] = []
        seen_candidate_objs: set[int] = set()
        for candidate in [model, *(agent._run_fallback_models or [])]:
            if id(candidate) in seen_candidate_objs:
                continue
            candidates.append(candidate)
            seen_candidate_objs.add(id(candidate))
        last_exc: Optional[BaseException] = None
        # Reset per-call bookkeeping. last_used_* reflects the model that
        # actually produced the response returned by THIS call invocation.
        state.last_used_model_id = None
        state.last_used_model_idx = -1
        primary_id = model.id
        trigger: Optional[str] = None  # "content_filter" | "fallback_only" | "exhausted_retry"

        def _emit_fallback_recovery(used_model_id: str, used_idx: int) -> None:
            """Audit-log + event-bus a successful fallback recovery.

            Only fires when the answer came from a fallback (idx > 0). Gives
            ops a single grep-able marker for "this run was rescued by a
            fallback model" without scraping retry/switch warnings.
            """
            logger.warning(
                f"[fallback.recovered] primary={primary_id} -> used={used_model_id} (idx={used_idx}, trigger={trigger})"
            )
            cb = agent._event_callback
            if cb is not None:
                try:
                    cb(
                        {
                            "type": "fallback.recovered",
                            "agent_name": agent.name or "Agent",
                            "primary_model": primary_id,
                            "used_model": used_model_id,
                            "fallback_index": used_idx,
                            "trigger": trigger,
                        }
                    )
                except Exception as e:
                    logger.warning(f"event callback failed for fallback.recovered: {e}")

        for model_idx, current in enumerate(candidates):
            is_fallback = model_idx > 0
            if is_fallback:
                logger.warning(
                    f"[fallback] switching to {current.id} ({model_idx}/{len(candidates) - 1}) trigger={trigger}"
                )

            for attempt in range(state.max_api_retry):
                RetryMixin._prepare_model_for_runner_call(current, model)
                if (
                    is_fallback
                    and not state.portable_fallback_compacted
                    and any(message.provider_checkpoint is not None for message in messages)
                    and not current.has_compatible_native_checkpoint(messages)
                ):
                    state.portable_fallback_compacted = True
                    cm = agent.tool_config.compression_manager
                    if cm is not None:
                        compacted = await cm.auto_compact(messages, model=current, force=True)
                        if compacted:
                            state.context_collapsed = True
                            logger.info(
                                f"Compacted portable transcript before cross-provider fallback to {current.id}"
                            )
                            cb = agent._event_callback
                            if cb is not None:
                                cb(
                                    {
                                        "type": "compact.fallback_portable",
                                        "agent_name": agent.name or "Agent",
                                        "is_main_agent": agent._parent_run_id is None,
                                        "model": current.id,
                                    }
                                )
                message_checkpoint = len(messages)
                try:
                    if stream:
                        # Stream: defer content_filter detection to the consumer.
                        # Exception-based fallbacks (timeout/5xx/content_filter
                        # raised at connect time) are still handled below.
                        # Record optimistically; consumer may flip later if
                        # finish_reason turns out to be content_filter.
                        state.last_used_model_id = current.id
                        state.last_used_model_idx = model_idx
                        if is_fallback:
                            _emit_fallback_recovery(current.id, model_idx)
                        CompressMixin._emit_context_usage(agent, current, messages)
                        return ModelCallResult(
                            response=current.response_stream(messages=messages),
                            used_model=current,
                            used_fallback=is_fallback,
                        )

                    CompressMixin._emit_context_usage(agent, current, messages)
                    resp = await current.response(messages=messages)

                    # Non-stream: content_filter is a normal-return finish_reason.
                    _fr = (resp.finish_reason or "").lower()
                    if _fr in state.CONTENT_FILTER_FINISH_REASONS:
                        logger.warning(
                            f"[content_filter] {current.id} returned "
                            f"finish_reason={resp.finish_reason!r}; "
                            f"trying next fallback"
                        )
                        # Model.response() may already have appended the blocked
                        # assistant message. Fallback must retry the same clean
                        # prompt, not continue after the primary model's refusal.
                        del messages[message_checkpoint:]
                        trigger = "content_filter"
                        last_exc = RuntimeError(f"content_filter on {current.id} (finish_reason={resp.finish_reason})")
                        break  # exit retry loop, go to next model

                    # Success: stamp who actually answered.
                    state.last_used_model_id = current.id
                    state.last_used_model_idx = model_idx
                    if is_fallback:
                        _emit_fallback_recovery(current.id, model_idx)
                    return ModelCallResult(
                        response=resp,
                        used_model=current,
                        used_fallback=is_fallback,
                    )

                except Exception as exc:
                    del messages[message_checkpoint:]
                    last_exc = exc
                    err = str(exc).lower()

                    # One-shot recovery: cross-provider tool-call/tool-result
                    # format mismatch (e.g. resuming a session recorded under
                    # a different model provider). Strip tool artifacts and
                    # retry immediately — independent of max_api_retry /
                    # fallback-candidate bookkeeping, since this is a
                    # guaranteed-safe structural fix, not a flaky transient.
                    is_tool_history_error = any(h in err for h in state.TOOL_HISTORY_HINTS)
                    if is_tool_history_error and not state.tool_history_sanitized_done:
                        state.tool_history_sanitized_done = True
                        logger.warning(
                            f"[tool_history] {current.id} rejected tool-call "
                            f"history (likely cross-model resume); stripping "
                            f"tool messages and retrying once: {exc}"
                        )
                        PersistMixin._sanitize_tool_history_after_error(agent, messages)
                        try:
                            if stream:
                                state.last_used_model_id = current.id
                                state.last_used_model_idx = model_idx
                                if is_fallback:
                                    _emit_fallback_recovery(current.id, model_idx)
                                CompressMixin._emit_context_usage(agent, current, messages)
                                return ModelCallResult(
                                    response=current.response_stream(messages=messages),
                                    used_model=current,
                                    used_fallback=is_fallback,
                                )
                            CompressMixin._emit_context_usage(agent, current, messages)
                            resp = await current.response(messages=messages)
                            state.last_used_model_id = current.id
                            state.last_used_model_idx = model_idx
                            if is_fallback:
                                _emit_fallback_recovery(current.id, model_idx)
                            return ModelCallResult(response=resp, used_model=current, used_fallback=is_fallback)
                        except Exception as exc2:
                            del messages[message_checkpoint:]
                            last_exc = exc2
                            err = str(exc2).lower()
                            # Fall through to the normal classification below
                            # with the post-sanitize exception.

                    # Reactive compact: prompt_too_long -> emergency compress.
                    # Only attempted on the primary model; fallbacks inherit the
                    # already-compacted message list. An oversized *trailing*
                    # user turn cannot be rescued (Layer 2 keeps it), so surface
                    # the provider error immediately instead of hiding it behind
                    # a summary that preserves the same text.
                    is_too_long = any(h in err for h in state.PROMPT_TOO_LONG_HINTS)
                    if is_too_long:
                        _window = current.context_window if isinstance(current.context_window, int) else 0
                        if is_irreducible_prompt_too_long(
                            messages, context_window=_window, model_id=current.id,
                        ):
                            logger.warning(
                                f"[prompt_too_long] {current.id}: trailing user turn "
                                f"already fills the context window; surfacing provider error"
                            )
                            raise
                        if not state.reactive_compact_done and not is_fallback:
                            state.reactive_compact_done = True
                            if await CompressMixin._try_reactive_compact(messages, agent, current):
                                state.context_collapsed = True
                                continue
                        # Compact already tried (or refused) and the retry still
                        # does not fit — do not wrap/fallback; raise the original
                        # BadRequest so the CLI shows the model's limit text.
                        raise

                    # Content filter raised as exception (some providers do this
                    # instead of setting finish_reason). No point retrying same
                    # model — moderation is deterministic.
                    is_content_filter = any(h in err for h in state.CONTENT_FILTER_HINTS)
                    if is_content_filter:
                        logger.warning(f"[content_filter] {current.id} raised content_filter exception: {exc}")
                        trigger = "content_filter"
                        break  # next model

                    # Hard outage errors: do not retry the same model. Switch
                    # directly to the next fallback model when configured.
                    is_fallback_only = any(r in err for r in state.FALLBACK_ONLY_SUBSTRINGS)
                    if is_fallback_only:
                        logger.warning(f"[fallback] {current.id} hit non-retry outage: {exc}; trying next fallback")
                        trigger = "fallback_only"
                        break

                    # Retryable transient errors: backoff within current model.
                    # Merge SDK defaults with model-level + env-level user
                    # extensions, so deployment-specific proxy markers
                    # (e.g. a private "gateway_error") become retryable
                    # without touching SDK source.
                    _retryable = current.get_retryable_substrings(state.RETRYABLE_SUBSTRINGS)
                    is_retryable = any(r in err for r in _retryable)
                    if is_retryable and attempt < state.max_api_retry - 1:
                        wait = (2**attempt) + random.uniform(0.0, 1.0)
                        logger.warning(
                            f"[APIRetry] {current.id} attempt "
                            f"{attempt + 1}/{state.max_api_retry}, "
                            f"retrying same model in {wait:.1f}s: {exc}"
                        )
                        await asyncio.sleep(wait)
                        continue

                    if is_retryable:
                        # Exhausted local attempts → fall through to next model.
                        logger.warning(
                            f"[APIRetry] {current.id} exhausted {state.max_api_retry} attempts; trying next fallback"
                        )
                        trigger = "exhausted_retry"
                        break

                    # Truly non-retryable (auth, malformed request, etc.).
                    # Fallback would not help — propagate immediately.
                    raise

        # All models in the chain failed.
        logger.error(f"[fallback] All {len(candidates)} models exhausted. Last error: {last_exc}")
        raise RuntimeError(
            f"LLM call failed across {len(candidates)} model(s) "
            f"(primary + {len(candidates) - 1} fallback). Last error: {last_exc}"
        ) from last_exc

