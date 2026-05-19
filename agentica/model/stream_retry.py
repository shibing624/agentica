# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Generic stream-with-retry wrapper for provider chat APIs.

Real-world streaming SSE connections occasionally surface malformed chunks
(proxy gateway hiccups, partial chunked-transfer reads, JSON decode errors
mid-stream). When the failure happens *before any chunk has been yielded
downstream*, retrying is safe and dramatically improves reliability. Once
even one chunk has been yielded to the caller, retrying would produce
duplicate tokens — at that point we must propagate the error.

Usage::

    from agentica.model.stream_retry import stream_with_retry

    def is_parser_error(exc: BaseException) -> bool:
        return isinstance(exc, (json.JSONDecodeError, ValueError))

    async def _open():
        return self.get_client().chat.completions.create(..., stream=True)

    async for chunk in stream_with_retry(_open, is_parser_error=is_parser_error):
        yield chunk

Design notes
------------
- Pre-yield-only retry: tracked via ``first_chunk_seen`` flag. Any exception
  after a yield is propagated verbatim, preserving the original provider
  exception class (do not swallow).
- Bounded retries (default 2) with exponential backoff + small jitter.
- Connection-open errors are also retryable when they pass ``is_parser_error``
  — that lets providers fold "open failed with malformed first chunk" into
  the same path as "iteration produced malformed chunk".
"""
import asyncio
import logging
import os
import random
from typing import AsyncIterator, Awaitable, Callable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# SDK-level default retryable substrings for the stream wrapper.
# Kept to protocol/transport-level transients only — vendor-specific proxy
# markers (e.g. a private "venus_error" corp gateway) must be added by the
# caller via ``extra_substrings=`` or env ``AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS``.
# Listed inline (not imported) to avoid a cycle with model.loop_state.
_DEFAULT_RETRYABLE_SUBSTRINGS = (
    "rate_limit", "rate limit", "429",
    "502", "503", "504",
    "internal server error",
    "bad gateway", "gateway timeout", "service unavailable",
    "temporarily unavailable",
    "connection", "timeout", "overloaded",
    "remote disconnected", "remotedisconnected",
    "incomplete chunked read", "chunked encoding", "premature",
)


def _merged_substrings(extra: Optional[Sequence[str]]) -> tuple:
    """Defaults + ``extra`` + ``AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS`` env var."""
    merged = {s.lower() for s in _DEFAULT_RETRYABLE_SUBSTRINGS}
    if extra:
        merged.update(s.lower() for s in extra if s)
    env_extra = os.environ.get("AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS", "")
    if env_extra:
        merged.update(s.strip().lower() for s in env_extra.split(",") if s.strip())
    return tuple(merged)


def default_is_parser_error(
    exc: BaseException,
    extra_substrings: Optional[Sequence[str]] = None,
) -> bool:
    """Default predicate: retry on JSON decode errors OR transient text matches.

    Notes
    -----
    - ``json.JSONDecodeError`` is treated as inherently retryable because it
      almost always indicates a malformed SSE chunk during streaming.
    - Bare ``ValueError`` is NOT auto-retryable (it commonly carries
      configuration errors like "model 'xxx' does not exist") — it must
      additionally match a substring to retry.
    - ``extra_substrings`` lets deployments add vendor-proxy markers without
      patching SDK defaults.
    """
    import json
    if isinstance(exc, json.JSONDecodeError):
        return True
    text = str(exc).lower()
    return any(s in text for s in _merged_substrings(extra_substrings))


async def stream_with_retry(
    open_stream: Callable[[], Awaitable[AsyncIterator[T]]],
    *,
    is_parser_error: Optional[Callable[[BaseException], bool]] = None,
    extra_substrings: Optional[Sequence[str]] = None,
    max_retries: int = 2,
    base_delay: float = 0.5,
    provider_label: str = "stream",
) -> AsyncIterator[T]:
    """Iterate a provider stream with bounded pre-yield retries.

    Args:
        open_stream: Async callable that opens and returns the stream iterator.
            Called once per attempt; must produce a fresh stream each time.
        is_parser_error: Predicate to classify an exception as retryable.
            Default catches JSON decode errors and common transient text
            (gateway / proxy / 5xx / rate-limit / connection / venus_error).
        max_retries: Max number of *additional* attempts after the first one
            (so total attempts = max_retries + 1). Default 2 → up to 3 tries.
        base_delay: Initial backoff in seconds; doubled per attempt + jitter.
        provider_label: Free-form string for log lines (e.g. ``"openai stream"``).

    Yields:
        Whatever the underlying stream yields.

    Raises:
        The most recent exception if all attempts fail, OR any exception that
        occurs after at least one chunk has been yielded (retry would
        duplicate output).
    """
    if is_parser_error is None:
        def is_parser_error(exc: BaseException) -> bool:  # type: ignore[misc]
            return default_is_parser_error(exc, extra_substrings=extra_substrings)

    attempt = 0
    last_exc: BaseException | None = None
    while True:
        first_chunk_seen = False
        try:
            stream = await open_stream()
            async for chunk in stream:
                first_chunk_seen = True
                yield chunk
            return
        except BaseException as exc:  # noqa: BLE001 — we re-raise unless retryable
            last_exc = exc
            # Never retry once we've yielded — caller has already seen partial output.
            if first_chunk_seen:
                raise
            if attempt >= max_retries or not is_parser_error(exc):
                raise
            wait = base_delay * (2 ** attempt) + random.uniform(0.0, 0.25)
            logger.warning(
                "[stream-retry] %s attempt %d/%d failed pre-yield, retrying in %.2fs: %s",
                provider_label, attempt + 1, max_retries + 1, wait, exc,
            )
            await asyncio.sleep(wait)
            attempt += 1
            continue
    # unreachable — `last_exc` retained for static analyzers
    if last_exc is not None:
        raise last_exc
