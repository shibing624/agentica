# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Runner model-call retry and fallback recovery
"""

import asyncio
import copy
import random

import httpx

from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    List,
    Optional,
    TYPE_CHECKING,
)

from agentica.agent.history_filter import (
    make_history_portable,
    to_portable_history_with_input_ids,
)
from agentica.compression.evict import is_irreducible_prompt_too_long
from agentica.utils.log import logger
from agentica.model.base import Model
from agentica.model.loop_state import LoopState
from agentica.model.message import Message
from agentica.model.response import ContentFilterError, ModelResponse
from agentica.model.usage import Usage
from agentica.run.config import RunConfig
from agentica.utils.langfuse_integration import langfuse_observation_context

if TYPE_CHECKING:
    from agentica.agent import Agent

from agentica.runner.types import ModelCallResult
from agentica.runner.compress import CompressMixin
from agentica.runner.persist import PersistMixin


async def _open_stream(stream: AsyncGenerator[Any, None]) -> AsyncIterator[Any]:
    """Pull the first chunk now, then hand back an iterator over the whole stream.

    ``response_stream()`` is a lazy async generator: calling it sends nothing,
    and the HTTP request (and its 503 / connection error / 429) happens on the
    first ``__anext__``. Pulling that chunk here makes an open failure raise
    inside ``_call_with_retry``, where it gets the same retry and fallback
    treatment as a non-streaming call. Errors after the first chunk still
    surface to the consumer: output is already on the wire by then.
    """
    try:
        first = await stream.__anext__()
    except StopAsyncIteration:
        return _replay_stream(None, stream, empty=True)
    return _replay_stream(first, stream, empty=False)


async def _replay_stream(first: Any, stream: AsyncGenerator[Any, None], *, empty: bool) -> AsyncIterator[Any]:
    try:
        if empty:
            return
        yield first
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


def _generation_output(
        messages: List[Message],
        checkpoint: int,
        response: Optional[ModelResponse] = None,
) -> Any:
    new_messages = [message.to_model_dict() for message in messages[checkpoint:]]
    if new_messages:
        return new_messages[-1] if len(new_messages) == 1 else new_messages
    if response is None:
        return None
    return {
        key: value
        for key, value in {
            "content": response.content,
            "reasoning_content": response.reasoning_content,
            "tool_call": response.tool_call,
            "finish_reason": response.finish_reason,
        }.items()
        if value is not None
    }


def _finalize_generation(
        observation: Any,
        messages: List[Message],
        checkpoint: int,
        response: Optional[ModelResponse] = None,
) -> None:
    observation.set_output(_generation_output(messages, checkpoint, response))
    new_messages = messages[checkpoint:]
    if new_messages:
        metrics = new_messages[-1].metrics
        finish_reason = new_messages[-1].finish_reason
    else:
        metrics = {}
        finish_reason = response.finish_reason if response is not None else None
    input_tokens = metrics.get("input_tokens", metrics.get("prompt_tokens"))
    output_tokens = metrics.get("output_tokens", metrics.get("completion_tokens"))
    total_tokens = metrics.get("total_tokens")
    if (
        total_tokens is None
        and isinstance(input_tokens, int)
        and isinstance(output_tokens, int)
    ):
        total_tokens = input_tokens + output_tokens
    usage_details = {
        key: value
        for key, value in {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens,
        }.items()
        if isinstance(value, int)
    }
    if usage_details:
        observation.update(usage_details=usage_details)
    if finish_reason:
        observation.set_metadata("finish_reason", finish_reason)


async def _trace_model_stream(
        stream: AsyncIterator[ModelResponse],
        *,
        observation_context: Any,
        observation: Any,
        messages: List[Message],
        checkpoint: int,
) -> AsyncIterator[ModelResponse]:
    error: Optional[BaseException] = None
    try:
        async for chunk in stream:
            yield chunk
    except BaseException as stream_error:
        error = stream_error
        observation.update(level="ERROR", status_message=str(stream_error)[:1000])
        raise
    finally:
        _finalize_generation(observation, messages, checkpoint)
        observation_context.__exit__(
            type(error) if error is not None else None,
            error,
            error.__traceback__ if error is not None else None,
        )


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
                trace_trigger=f"fallback_on_break:{break_reason}",
                trace_is_fallback=True,
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
    def _classify_api_error(
        exc: BaseException, retryable_substrings: tuple, fallback_only_substrings: tuple,
    ) -> tuple[bool, bool]:
        """Return (retryable, fallback_only), preferring status/type to error prose."""
        # openai is a core dependency, but importing it here keeps `import agentica`
        # and CLI startup from loading the provider SDK.
        from openai import APIConnectionError, APITimeoutError

        status = getattr(exc, "status_code", None)
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
        if isinstance(status, int):
            if status in (408, 409, 429, 504):
                return True, False
            if 500 <= status < 600:
                return False, True
            if 400 <= status < 500:
                return False, False
        if isinstance(exc, (APITimeoutError, httpx.TimeoutException, TimeoutError)):
            return True, False

        err = str(exc).lower()
        retryable = any(marker in err for marker in retryable_substrings)
        fallback_only = not retryable and (
            isinstance(exc, (APIConnectionError, httpx.TransportError, ConnectionError))
            or any(marker in err for marker in fallback_only_substrings)
        )
        return retryable, fallback_only

    @staticmethod
    async def _call_with_retry(
        model: "Model",
        messages: List[Message],
        state: "LoopState",
        agent: "Agent",
        *,
        stream: bool = False,
        after_content_filter: bool = False,
        trace_trigger: Optional[str] = None,
        trace_is_fallback: bool = False,
    ):
        """Call model.response()/response_stream() with retry, reactive compact,
        and cross-provider fallback chain.

        Returns:
            ModelCallResult(response=ModelResponse, used_model=...) for non-stream.
            ModelCallResult(response=async iterator, used_model=...) for stream.

        Per-call fallback (not per-run): each invocation starts from the primary
        ``model`` argument; ``agent._run_fallback_models`` are tried in order
        when triggered by:

          1. Provider content filtering, without retrying the same model.
          2. Fallback-only API error (``FALLBACK_ONLY_SUBSTRINGS``), such as
             connection failure / 502 / 503 / bad gateway.
          3. Retryable API error (``RETRYABLE_SUBSTRINGS``) that exhausted local
             exponential backoff on the current model.

        ``after_content_filter`` is the streaming continuation path: the
        consumer has already exhausted the current model's lazy generator and
        observed a filtered finish_reason, so this call starts at the next
        fallback candidate instead of re-sending to the filtered model.

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
        if after_content_filter:
            candidates = candidates[1:]
            if not candidates:
                raise ContentFilterError(model.id, "no fallback model remains")
        last_exc: Optional[BaseException] = None
        # Reset per-call bookkeeping. last_used_* reflects the model that
        # actually produced the response returned by THIS call invocation.
        state.last_used_model_id = None
        state.last_used_model_idx = -1
        primary_id = model.id
        trigger: Optional[str] = "content_filter" if after_content_filter else None

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

        candidate_offset = 1 if after_content_filter else 0
        for local_idx, current in enumerate(candidates):
            model_idx = local_idx + candidate_offset
            is_fallback = model_idx > 0
            if is_fallback:
                logger.warning(
                    f"[fallback] switching to {current.id} "
                    f"({model_idx}/{candidate_offset + len(candidates) - 1}) trigger={trigger}"
                )

            # A reactive-compact resend is a structural fix (one-shot via
            # reactive_compact_done), not a retry: it does not spend an attempt.
            attempt = 0
            while attempt < state.max_api_retry:
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

                async def _call_current_model() -> Any:
                    observation_context = langfuse_observation_context(
                        name=current.id,
                        as_type="generation",
                        input_data=[message.to_model_dict() for message in messages],
                        model=current.id,
                        metadata={
                            "provider": current.provider or "",
                            "attempt": attempt + 1,
                            "fallback_index": model_idx + (1 if trace_is_fallback else 0),
                            "is_fallback": is_fallback or trace_is_fallback,
                            "trigger": trace_trigger or trigger or "primary",
                        },
                    )
                    observation = observation_context.__enter__()
                    try:
                        CompressMixin._emit_context_usage(agent, current, messages)
                        if stream:
                            opened_stream = await _open_stream(
                                current.response_stream(messages=messages)
                            )
                            return _trace_model_stream(
                                opened_stream,
                                observation_context=observation_context,
                                observation=observation,
                                messages=messages,
                                checkpoint=message_checkpoint,
                            )
                        response = await current.response(messages=messages)
                        _finalize_generation(
                            observation,
                            messages,
                            message_checkpoint,
                            response,
                        )
                        observation_context.__exit__(None, None, None)
                        return response
                    except BaseException as error:
                        observation.update(
                            level="ERROR",
                            status_message=str(error)[:1000],
                        )
                        observation_context.__exit__(
                            type(error),
                            error,
                            error.__traceback__,
                        )
                        raise

                try:
                    if stream:
                        # Stream finish_reason is available only after the
                        # consumer exhausts the generator. The Runner checks it
                        # there before accepting the call as complete.
                        opened = await _call_current_model()
                        state.last_used_model_id = current.id
                        state.last_used_model_idx = model_idx
                        if is_fallback:
                            _emit_fallback_recovery(current.id, model_idx)
                        return ModelCallResult(
                            response=opened,
                            used_model=current,
                            used_fallback=is_fallback,
                        )

                    resp = await _call_current_model()

                    # Content filtering is deterministic for this model: switch
                    # candidates immediately without same-model retry.
                    _fr = (resp.finish_reason or "").lower()
                    if _fr in state.CONTENT_FILTER_FINISH_REASONS:
                        logger.warning(
                            f"[content_filter] {current.id} returned "
                            f"finish_reason={resp.finish_reason!r}; trying next fallback"
                        )
                        del messages[message_checkpoint:]
                        trigger = "content_filter"
                        last_exc = ContentFilterError(
                            current.id,
                            f"finish_reason={resp.finish_reason!r}",
                        )
                        break

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

                    if isinstance(exc, ContentFilterError):
                        trigger = "content_filter"
                        break

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
                        PersistMixin._sanitize_tool_history_after_error(agent, messages, state)
                        # The list shrank; a failed retry must be trimmed back
                        # to the stripped end, not to the pre-strip length.
                        message_checkpoint = len(messages)
                        try:
                            if stream:
                                opened = await _call_current_model()
                                state.last_used_model_id = current.id
                                state.last_used_model_idx = model_idx
                                if is_fallback:
                                    _emit_fallback_recovery(current.id, model_idx)
                                return ModelCallResult(
                                    response=opened,
                                    used_model=current,
                                    used_fallback=is_fallback,
                                )
                            resp = await _call_current_model()
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

                    # A signed Anthropic thinking block can occasionally become
                    # unverifiable on the very next tool-result request (observed
                    # through the Venus native Anthropic endpoint). Re-sending the
                    # same payload is deterministic, while dropping the whole tool
                    # round loses the work already completed. Convert both stores
                    # to the shared shape: model-bound thinking/replay state goes,
                    # but assistant.tool_calls and their results survive.
                    invalid_thinking_signature = (
                        "thinking blocks cannot be modified" in err
                        or (
                            "invalid" in err
                            and "signature" in err
                            and "thinking" in err
                        )
                    )
                    if (
                        invalid_thinking_signature
                        and not state.thinking_history_sanitized_done
                    ):
                        state.thinking_history_sanitized_done = True
                        logger.warning(
                            "[thinking_signature] %s rejected signed thinking; "
                            "removing model-bound reasoning and retrying once",
                            current.id,
                        )
                        make_history_portable(agent.working_memory)
                        portable, portable_input_ids = to_portable_history_with_input_ids(
                            messages,
                            state.input_message_ids,
                        )
                        messages[:] = portable
                        state.context_collapsed = True
                        state.input_message_ids = portable_input_ids
                        # A second failure must trim against the rebuilt list.
                        message_checkpoint = len(messages)
                        continue

                    # Reactive compact: prompt_too_long -> emergency compress.
                    # Only attempted on the primary model; fallbacks inherit the
                    # already-compacted message list. An oversized *trailing*
                    # user turn cannot be rescued (Layer 2 keeps it), so surface
                    # the provider error immediately instead of hiding it behind
                    # a summary that preserves the same text.
                    is_too_long = state.is_prompt_too_long(last_exc)
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

                    # Some providers raise instead of returning a finish_reason.
                    # Switch candidates immediately; do not retry this model.
                    is_content_filter = any(h in err for h in state.CONTENT_FILTER_HINTS)
                    if is_content_filter:
                        logger.warning(f"[content_filter] {current.id} raised content_filter exception: {exc}")
                        last_exc = ContentFilterError(current.id, str(exc))
                        trigger = "content_filter"
                        break

                    # Retryable transient errors: backoff within current model.
                    # Merge SDK defaults with model-level + env-level user
                    # extensions, so deployment-specific proxy markers
                    # (e.g. a private "gateway_error") become retryable
                    # without touching SDK source.
                    _retryable = current.get_retryable_substrings(state.RETRYABLE_SUBSTRINGS)
                    is_retryable, is_fallback_only = RetryMixin._classify_api_error(
                        last_exc, _retryable, state.FALLBACK_ONLY_SUBSTRINGS,
                    )
                    if is_fallback_only:
                        logger.warning(f"[fallback] {current.id} hit non-retry outage: {exc}; trying next fallback")
                        trigger = "fallback_only"
                        break

                    if is_retryable and attempt < state.max_api_retry - 1:
                        wait = (2**attempt) + random.uniform(0.0, 1.0)
                        logger.warning(
                            f"[APIRetry] {current.id} attempt "
                            f"{attempt + 1}/{state.max_api_retry}, "
                            f"retrying same model in {wait:.1f}s: {exc}"
                        )
                        await asyncio.sleep(wait)
                        attempt += 1
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
        if isinstance(last_exc, ContentFilterError):
            raise last_exc
        raise RuntimeError(
            f"LLM call failed across {len(candidates)} model(s) "
            f"(primary + {len(candidates) - 1} fallback). Last error: {last_exc}"
        ) from last_exc

