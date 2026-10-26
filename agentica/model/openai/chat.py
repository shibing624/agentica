# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description:
OpenAI chat model implementation.
"""

from os import getenv
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import json
import threading
import time
import re
import sys
from typing import Optional, List, AsyncIterator, Dict, Any, Union, Literal
from typing import Awaitable, Callable, ClassVar
from uuid import uuid4
from xml.etree import ElementTree

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import httpx
import openai
from enum import Enum, EnumMeta
from pydantic import BaseModel

from openai import AsyncOpenAI as AsyncOpenAIClient
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from agentica.model.base import Model, require_first_choice
from agentica.model.cache_routing import (
    persistent_cache_session_id,
    resolve_cache_session_id,
)
from agentica.model.cost import get_model_max_output_tokens
from agentica.model.message import Message, VOLATILE_SYSTEM_MARKER
from agentica.model.metrics import Metrics, StreamData
from agentica.model.response import ModelResponse
from agentica.model.openai.sse_sanitize import TolerantSSETransport
from agentica.model.stream_retry import stream_with_retry
from agentica.utils.log import log_file_only, logger


class OpenAIImageTypeMeta(EnumMeta):
    def __contains__(cls, image_type: object) -> bool:
        try:
            cls(image_type)
        except ValueError:
            return False
        return True


class OpenAIImageType(Enum, metaclass=OpenAIImageTypeMeta):
    r"""Image types supported by OpenAI vision model."""

    # https://platform.openai.com/docs/guides/vision
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    WEBP = "webp"
    GIF = "gif"


# Gemini's signature placeholder for a tool call issued by another model
# (https://ai.google.dev/gemini-api/docs/thought-signatures). Google documents
# this exact value for replaying a trace whose signatures are not available.
_FOREIGN_CALL_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}

# Request parameter names shared between request_kwargs and to_dict
_OPENAI_REQUEST_PARAMS = [
    "store",
    "reasoning_effort",
    "verbosity",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "max_tokens",
    "max_completion_tokens",
    "modalities",
    "audio",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "user",
    "top_p",
    "extra_headers",
    "extra_body",
    "extra_query",
    "metadata",
]
_DEEPSEEK_THINKING_UNSUPPORTED_PARAMS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
)

# ── Prompt caching for OpenAI-compatible proxies that front Anthropic Claude ──
# Anthropic caps a single request at 4 cache breakpoints total, so the injector
# below budgets tools(1) + system(1) + trailing messages(remaining) ≤ 4.
_CACHE_CONTROL_BREAKPOINT_BUDGET = 4
_SYSTEM_POSITION_ERROR = "role 'system' must precede"


def _has_cacheable_content(content: Any) -> bool:
    """True when content has something worth attaching a cache breakpoint to."""
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return len(content) > 0
    return False


def _repair_claude_system_positions(
    messages: List[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    """Relocate only invalid mid-conversation system directives.

    Some OpenAI-compatible Claude gateways accept a content-bearing system
    message only immediately before an assistant message or at the end. The
    leading OpenAI system prefix is left untouched for the gateway to hoist.
    A rejected mid-conversation directive is moved to the boundary immediately
    before the next assistant reply, preserving its intended forward scope.
    """
    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end].get("role") == "system":
        prefix_end += 1

    moves = []
    for index in range(prefix_end, len(messages)):
        if messages[index].get("role") != "system":
            continue
        if index == len(messages) - 1 or messages[index + 1].get("role") == "assistant":
            continue
        target = next(
            (i for i in range(index + 1, len(messages)) if messages[i].get("role") == "assistant"),
            None,
        )
        moves.append((index, target, messages[index]))

    if not moves:
        return None

    moved_indexes = {index for index, _, _ in moves}
    repaired = [(index, message) for index, message in enumerate(messages) if index not in moved_indexes]
    for _, target, directive in moves:
        if target is None:
            repaired.append((-1, directive))
            continue
        insert_at = next(i for i, (original_index, _) in enumerate(repaired) if original_index == target)
        repaired.insert(insert_at, (-1, directive))

    result = [message for _, message in repaired]
    for index in range(prefix_end, len(result)):
        if (
            result[index].get("role") == "system"
            and index != len(result) - 1
            and result[index + 1].get("role") != "assistant"
        ):
            return None
    return result


_THINKING_BLOCK_TYPES = frozenset({"thinking", "redacted_thinking"})


def _is_unsigned_thinking_block(block: Any) -> bool:
    """True for a thinking block Claude would reject as ``Invalid signature``."""
    if not isinstance(block, dict):
        return False
    block_type = block.get("type")
    if block_type == "thinking":
        signature = block.get("signature")
        return not (isinstance(signature, str) and signature)
    if block_type == "redacted_thinking":
        data = block.get("data")
        return not (isinstance(data, str) and data)
    return False


def _tag_content_block_cache_control(content: Any) -> Any:
    """Ensure ``content`` is a list of blocks and tag the last cacheable block.

    Accepts str (wrapped into one text block) or a list of content blocks.
    None / empty content is returned unchanged (nothing to cache).
    Thinking blocks are skipped: tagging them invalidates the signature.
    """
    if content is None:
        return content
    if isinstance(content, str):
        if not content:
            return content
        return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    if isinstance(content, list) and content:
        tag_at = None
        for idx in range(len(content) - 1, -1, -1):
            block = content[idx]
            if isinstance(block, dict) and block.get("type") not in _THINKING_BLOCK_TYPES:
                tag_at = idx
                break
        if tag_at is None:
            return list(content)
        new_blocks = []
        for idx, block in enumerate(content):
            if idx == tag_at and isinstance(block, dict):
                b = dict(block)
                b["cache_control"] = {"type": "ephemeral"}
                new_blocks.append(b)
            else:
                new_blocks.append(block)
        return new_blocks
    return content


def _tag_system_content_cache_control(content: Any) -> Any:
    """Tag the system message content, splitting at the volatile marker.

    With ``VOLATILE_SYSTEM_MARKER`` present, returns two text blocks — the
    stable head WITH the breakpoint and the volatile tail WITHOUT — so a
    per-turn memory/summary update no longer busts the whole cached
    tools+system prefix. Without a marker, identical to
    :func:`_tag_content_block_cache_control`.
    """
    if isinstance(content, str):
        head, sep, tail = content.partition(VOLATILE_SYSTEM_MARKER)
        if sep and head.strip():
            blocks = [{"type": "text", "text": head.rstrip(), "cache_control": {"type": "ephemeral"}}]
            # The marker itself is a pure routing hint — never send it to the
            # model. An empty tail (marker was the last line) is dropped too,
            # since empty text blocks are rejected by the API.
            if tail.strip():
                blocks.append({"type": "text", "text": tail.lstrip("\n")})
            return blocks
        if sep:
            # Marker at offset 0: no stable prefix to cache — forward the tail
            # without the marker rather than leaking it inside a tagged block.
            content = tail
    return _tag_content_block_cache_control(content)


def _strip_volatile_marker_from_message(message: Any) -> Any:
    """Drop the volatile marker from a system message.

    The marker only exists to drive the cache split in
    :func:`_tag_system_content_cache_control`. When caching is disabled that
    split never runs, so leaving the marker in the prompt is pure noise — it
    would be sent verbatim to the model on the default (cache-off) config.
    """
    if isinstance(message, dict) and message.get("role") == "system":
        content = message.get("content")
        if isinstance(content, str) and VOLATILE_SYSTEM_MARKER in content:
            stripped = dict(message)
            stripped["content"] = content.replace(VOLATILE_SYSTEM_MARKER, "")
            return stripped
    return message


def _tag_tools_cache_control(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tag the last tool definition with an ephemeral cache breakpoint.

    Anthropic caches the tools block as a whole by marking the final tool.
    Returns a shallow copy so the caller's tool defs are not mutated.
    """
    if not tools:
        return tools
    out = list(tools)
    last = out[-1]
    if isinstance(last, dict):
        tagged = dict(last)
        tagged["cache_control"] = {"type": "ephemeral"}
        out[-1] = tagged
    return out


# The id helpers live in agentica.model.cache_routing so the Anthropic provider
# can share them without importing this module (and the OpenAI SDK with it).


def _request_has_cache_control(messages: List[Dict[str, Any]], tools: Any) -> bool:
    """True if any message content block or tool already carries cache_control."""
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and "cache_control" in t:
                return True
    return False


@dataclass
class OpenAIChat(Model):
    """
    A class for interacting with OpenAI models.

    For more information, see: https://platform.openai.com/docs/api-reference/chat/create
    """

    id: str = "gpt-4o"  # model name
    name: str = "OpenAIChat"
    provider: str = "OpenAI"
    context_window: int = 200000

    # Request parameters
    store: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    metadata: Optional[Dict[str, Any]] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Any] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    modalities: Optional[List[str]] = None  # "text" and/or "audio"
    audio: Optional[Dict[str, Any]] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    user: Optional[str] = None
    top_p: Optional[float] = None
    extra_headers: Optional[Any] = None
    extra_query: Optional[Any] = None
    extra_body: Optional[Dict[str, Any]] = None
    request_params: Optional[Dict[str, Any]] = None

    # Prompt caching for OpenAI-compatible proxies that front Anthropic Claude.
    # The proxy accepts Anthropic-style ``cache_control`` blocks
    # inside OpenAI-format system / messages / tools and reports
    # ``prompt_tokens_details.cache_read_tokens`` / ``cache_creation_tokens``.
    # Default OFF: pure OpenAI / DeepSeek / Zhipu may reject the non-standard
    # ``cache_control`` key. Enable explicitly (SDK flag, CLI --enable_cache_control,
    # or config.yaml profile) for Anthropic-via-OpenAI proxies. If a request
    # already carries cache_control blocks while this is off, a warning is logged
    # telling the user to flip it on.
    enable_cache_control: bool = False
    # Max cache breakpoints placed on the trailing non-system messages
    # (rolling cache window). Anthropic caps a request at 4 breakpoints total,
    # so the effective count is min(cache_control_messages, 4 - tools - system).
    cache_control_messages: int = 3
    # Optional sticky-routing header name (e.g. "X-Session-Id"): when set,
    # a session id is loaded-or-created once per base_url (persisted in
    # ~/.agentica/cache/cache_routing.json) and injected into ``extra_headers`` so
    # consecutive invokes — even across CLI restarts — land on the same
    # backend and actually hit the cache written by earlier requests. None = off.
    cache_control_session_header: Optional[str] = None
    # OpenAI's own cache is implicit — nothing is marked, the prefix is matched
    # automatically — but a hit also requires landing on the machine holding it,
    # and routing hashes only the first ~256 prompt tokens. ``prompt_cache_key``
    # is folded into that hash, so reusing one key across a conversation is the
    # sanctioned way to buy routing affinity (and on gpt-5.6+ it is required for
    # the more reliable matching). Orthogonal to ``enable_cache_control``, which
    # is the Anthropic breakpoint protocol: this one costs nothing and applies
    # to every OpenAI-compatible endpoint, hence default ON. Turn it off for an
    # endpoint strict enough to reject unknown request fields.
    enable_prompt_cache_key: bool = True
    # Cache keep-alive: while the session sits idle, re-read the cached
    # tools+system prefix every `cache_keepalive_interval` seconds (a cache read
    # costs ~0.1x input price vs ~1.25x for a re-write after the 5-min TTL
    # lapses; verified on a proxy that refreshes TTL on cache reads). The ping chain
    # stops after `cache_keepalive_max_pings` pings without new traffic, so idle
    # cost is bounded.
    cache_keepalive: bool = True
    cache_keepalive_interval: int = 240
    cache_keepalive_max_pings: int = 3
    _last_cache_payload: Any = field(default=None, init=False, repr=False)
    _cache_keepalive_generation: int = field(default=0, init=False, repr=False)
    _requires_system_position_repair: bool = field(default=False, init=False, repr=False)

    # Client parameters
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[Union[float, httpx.Timeout]] = None
    max_retries: Optional[int] = 2
    default_headers: Optional[Any] = None
    default_query: Optional[Any] = None
    http_client: Optional[httpx.Client] = None
    client_params: Optional[Dict[str, Any]] = None

    # OpenAI client (async-only)
    client: Optional[AsyncOpenAIClient] = None
    # Same loop-affinity rule as Anthropic: the CLI runs each turn on a fresh
    # asyncio.run loop. Reusing a client whose httpx pool is bound to a closed
    # loop prints "Event loop is closed" from aclose/__del__.
    _client_loop: Optional[Any] = field(default=None, init=False, repr=False)

    # Internal parameters. Not used for API requests
    # Whether to use the structured outputs with this Model.
    use_structured_outputs: bool = False
    # Whether the Model supports structured outputs.
    supports_structured_outputs: bool = True
    # Langfuse tags for tracing (additional tags beyond what Agent provides)
    langfuse_tags: Optional[List[str]] = None
    # True while the output cap in the request was chosen by agentica (catalog
    # ceiling or a recovery raise), not by the user. Only such a cap is ever
    # raised or dropped; a user-configured one is sent as given.
    _auto_output_cap: bool = field(default=False, init=False, repr=False)
    # Default request field that carries the output cap on this wire.
    _OUTPUT_CAP_KEY: ClassVar[str] = "max_tokens"
    # Chat Completions deployments disagree on the wire name. This is a
    # protocol capability learned from a structured API error, not inferred
    # from model-name families.
    _OUTPUT_CAP_KEYS: ClassVar[tuple[str, ...]] = ("max_tokens", "max_completion_tokens")
    _OUTPUT_CAP_KEY_BY_DEPLOYMENT: ClassVar[Dict[tuple[str, str, str], str]] = {}

    def _output_cap_deployment_key(self) -> tuple[str, str, str]:
        """Identity used to share a learned wire key across per-run clones."""
        endpoint = self.base_url or getenv("OPENAI_BASE_URL") or ""
        return self.provider, str(endpoint), self.id

    def _active_output_cap_key(self, request_kwargs: Dict[str, Any]) -> Optional[str]:
        present = [key for key in self._OUTPUT_CAP_KEYS if key in request_kwargs]
        return present[0] if len(present) == 1 else None

    @staticmethod
    def _move_output_cap(
        request_kwargs: Dict[str, Any],
        source_key: str,
        target_key: str,
    ) -> Dict[str, Any]:
        moved = {key: value for key, value in request_kwargs.items() if key != source_key}
        moved[target_key] = request_kwargs[source_key]
        return moved

    @staticmethod
    def _is_unsupported_parameter(error: openai.BadRequestError, parameter: str) -> bool:
        return error.code == "unsupported_parameter" and error.param == parameter

    def _apply_learned_output_cap_key(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        active_key = self._active_output_cap_key(request_kwargs)
        if active_key is None:
            return request_kwargs
        learned_key = self._OUTPUT_CAP_KEY_BY_DEPLOYMENT.get(self._output_cap_deployment_key())
        if learned_key is None or learned_key == active_key or learned_key not in self._OUTPUT_CAP_KEYS:
            return request_kwargs
        return self._move_output_cap(request_kwargs, active_key, learned_key)

    def __post_init__(self):
        super().__post_init__()
        # Output cap precedence: user config > catalog ceiling > gateway default.
        # Without a cap in the request the gateway picks one — 20000 on one
        # gateway for claude-opus-5-5, which a high-effort thinking turn can spend
        # entirely on reasoning. The catalog ceiling is models.dev
        # ``limit.output``. It is a ceiling, not a reservation: providers bill
        # what they generate.
        if not self._has_explicit_output_cap():
            catalog_cap = get_model_max_output_tokens(self.id)
            if catalog_cap is not None:
                self.max_tokens = catalog_cap
                self._auto_output_cap = True

    def _has_explicit_output_cap(self) -> bool:
        return self.max_tokens is not None or self.max_completion_tokens is not None

    def set_output_cap(self, max_output_tokens: int) -> None:
        if self.max_completion_tokens is not None:
            self.max_completion_tokens = max_output_tokens
        else:
            self.max_tokens = max_output_tokens

    def raise_output_cap(self, used_tokens: int) -> bool:
        """Double the cap after a turn spent ``used_tokens`` and showed nothing.

        Bounded by the catalog ceiling. A cap the user configured is left
        alone. Returns True when the next request will carry a higher cap.
        """
        if self._has_explicit_output_cap() and not self._auto_output_cap:
            return False
        target = used_tokens * 2
        ceiling = get_model_max_output_tokens(self.id)
        if ceiling is not None:
            target = min(target, ceiling)
        if self.max_tokens is not None and target <= self.max_tokens:
            return False
        logger.info(f"Raising output cap for {self.id}: {self.max_tokens} -> {target}")
        self.max_tokens = target
        self._auto_output_cap = True
        return True

    async def _create_with_auto_cap(
        self,
        create: Callable[[Dict[str, Any]], Awaitable[Any]],
        request_kwargs: Dict[str, Any],
    ) -> Any:
        """Send ``request_kwargs``; if an agentica-chosen cap is rejected, drop it.

        Chat Completions endpoints disagree between ``max_tokens`` and
        ``max_completion_tokens``. When the API identifies the sent field as an
        unsupported parameter, try the other field with the same value. A
        successful retry records that protocol capability for this exact
        deployment, including model id, so new model names require no code
        changes and per-run clones avoid another failed probe.

        Other 400 responses retain the existing recovery for agentica-chosen
        catalog caps: retry without the cap, remember a successful drop, and
        surface the retry error otherwise. User-configured cap values are never
        dropped.
        """
        request_kwargs = self._apply_learned_output_cap_key(request_kwargs)
        active_key = self._active_output_cap_key(request_kwargs)
        try:
            return await create(request_kwargs)
        except openai.BadRequestError as error:
            if (
                active_key is not None
                and len(self._OUTPUT_CAP_KEYS) == 2
                and self._is_unsupported_parameter(error, active_key)
            ):
                alternate_key = next(key for key in self._OUTPUT_CAP_KEYS if key != active_key)
                alternate_kwargs = self._move_output_cap(request_kwargs, active_key, alternate_key)
                try:
                    result = await create(alternate_kwargs)
                except openai.BadRequestError as alternate_error:
                    if (
                        not self._auto_output_cap
                        or not self._is_unsupported_parameter(alternate_error, alternate_key)
                    ):
                        raise
                    request_kwargs = alternate_kwargs
                    active_key = alternate_key
                else:
                    self._OUTPUT_CAP_KEY_BY_DEPLOYMENT[self._output_cap_deployment_key()] = alternate_key
                    logger.info(
                        f"{self.id}: endpoint rejected {active_key}; switched the output cap "
                        f"to {alternate_key} with the value preserved"
                    )
                    return result
            if not self._auto_output_cap or active_key is None:
                raise
            without_cap = {key: value for key, value in request_kwargs.items() if key != active_key}
            result = await create(without_cap)
            logger.info(
                f"{self.id}: endpoint rejected {active_key}={request_kwargs[active_key]}; "
                "using the endpoint's default output cap from now on"
            )
            self.max_tokens = None
            self._auto_output_cap = False
            return result

    def get_client_params(self) -> Dict[str, Any]:
        client_params: Dict[str, Any] = {}

        self.api_key = self.api_key or getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not set. Please set the OPENAI_API_KEY environment variable.")
        self.base_url = self.base_url or getenv("OPENAI_BASE_URL")
        if self.api_key is not None:
            client_params["api_key"] = self.api_key
        if self.organization is not None:
            client_params["organization"] = self.organization
        if self.base_url is not None:
            client_params["base_url"] = self.base_url
        if self.timeout is not None:
            client_params["timeout"] = self.timeout
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries
        if self.default_headers is not None:
            client_params["default_headers"] = self.default_headers
        if self.default_query is not None:
            client_params["default_query"] = self.default_query
        if self.client_params is not None:
            client_params.update(self.client_params)
        return client_params

    def get_client(self) -> AsyncOpenAIClient:
        """
        Returns an async OpenAI client (async-only, single implementation).

        If Langfuse is configured, uses Langfuse-wrapped AsyncOpenAI client
        for automatic tracing.

        Returns:
            AsyncOpenAIClient: An instance of the async OpenAI client.
        """
        if self.client is not None and self._client_loop is None:
            return self.client
        try:
            _loop: Optional[Any] = asyncio.get_running_loop()
        except RuntimeError:
            _loop = None
        if (
            self.client is not None
            and _loop is not None
            and self._client_loop is _loop
            and not _loop.is_closed()
        ):
            return self.client

        client_params: Dict[str, Any] = self.get_client_params()
        if self.http_client:
            client_params["http_client"] = self.http_client
        else:
            # Route the default client through the tolerant SSE transport so
            # double-wrapped `data: data: {...}` frames from OpenAI-compatible
            # proxies are repaired before the SDK's decoder sees them.
            # Limits must live on the transport: an explicitly constructed
            # transport ignores AsyncClient(limits=...).
            limits = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            transport = TolerantSSETransport(httpx.AsyncHTTPTransport(limits=limits))
            client_params["http_client"] = httpx.AsyncClient(transport=transport)

        self.client = AsyncOpenAIClient(**client_params)
        self._client_loop = _loop
        return self.client

    def _uses_claude_text_tool_call_compatibility(self) -> bool:
        """Return whether this OpenAI-compatible model is a Claude model.

        Native Anthropic models use ``agentica.model.anthropic.Claude`` and
        return structured ``tool_use`` blocks. This fallback is only for Claude
        models reached through an OpenAI-compatible endpoint that leaks
        ``<invoke>`` calls into text content.
        """
        return "claude" in self.id.lower()

    @staticmethod
    def _parse_claude_text_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
        """Convert complete Claude ``<invoke>`` XML blocks into OpenAI calls.

        Returns ``None`` when the text contains no XML tool call. If text
        starts an invoke block but the proxy emitted malformed or incomplete
        XML, raises instead of allowing the markup into assistant history.
        """
        if "<invoke" not in text:
            return None

        blocks = re.findall(r"<invoke\b[^>]*>.*?</invoke\s*>", text, flags=re.DOTALL)
        if not blocks:
            raise ValueError(
                "OpenAI-compatible provider returned a malformed Claude XML tool call. "
                "Expected a complete <invoke name=\"...\">...</invoke> block."
            )
        if "<invoke" in re.sub(r"<invoke\b[^>]*>.*?</invoke\s*>", "", text, flags=re.DOTALL):
            raise ValueError(
                "OpenAI-compatible provider returned a malformed Claude XML tool call. "
                "Expected a complete <invoke name=\"...\">...</invoke> block."
            )

        calls: List[Dict[str, Any]] = []
        for block in blocks:
            try:
                invoke = ElementTree.fromstring(block)
            except ElementTree.ParseError as exc:
                raise ValueError(
                    "OpenAI-compatible provider returned a malformed Claude XML tool call."
                ) from exc

            if invoke.tag != "invoke" or not invoke.attrib.get("name"):
                raise ValueError(
                    "OpenAI-compatible provider returned a malformed Claude XML tool call."
                )

            arguments: Dict[str, Any] = {}
            for parameter in invoke:
                name = parameter.attrib.get("name") if parameter.tag == "parameter" else None
                if not name or name in arguments:
                    raise ValueError(
                        "OpenAI-compatible provider returned a malformed Claude XML tool call."
                    )
                raw = "".join(parameter.itertext()).strip()
                # Every XML parameter is text. Array/object args therefore
                # arrive as a JSON string; parse them here so later
                # validation sees a list/dict instead of rejecting the call.
                if raw[:1] in "[{":
                    try:
                        arguments[name] = json.loads(raw)
                        continue
                    except (json.JSONDecodeError, ValueError):
                        pass
                arguments[name] = raw

            calls.append(
                {
                    "id": f"call_{uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": invoke.attrib["name"],
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        return calls

    def _collect_request_params(self) -> Dict[str, Any]:
        """Collect non-None request parameters (shared by request_kwargs and to_dict)."""
        params: Dict[str, Any] = {}
        for attr_name in _OPENAI_REQUEST_PARAMS:
            val = getattr(self, attr_name, None)
            if val is not None:
                params[attr_name] = val
        return params

    def describe_thinking_mode(self) -> str:
        """Describe thinking/reasoning configuration across OpenAI-compatible providers.

        Inspects:
        - ``reasoning_effort`` (OpenAI o-series, gpt-5.x)
        - ``extra_body.thinking`` (DeepSeek, Ark/Doubao: ``{"type": "enabled"|"disabled", ...}``)
        - ``extra_body.enable_thinking`` (Qwen / DashScope-compatible)
        - ``extra_body.thinking_enabled`` (Anthropic-via-OpenAI proxy;
          the flat thinking params ``thinking_enabled`` / ``reasoning_effort`` /
          ``thinking_display`` are forwarded as top-level request fields)
        """
        is_on: Optional[bool] = None
        effort = None
        extra_details: List[str] = []
        if self.reasoning_effort:
            if str(self.reasoning_effort).lower() in ("off", "none", "disabled"):
                is_on = False
            else:
                effort = self.reasoning_effort
                is_on = True
        if isinstance(self.extra_body, dict):
            thinking = self.extra_body.get("thinking")
            if isinstance(thinking, dict):
                t = thinking.get("type")
                budget = thinking.get("budget_tokens")
                if t == "enabled":
                    is_on = True
                elif t == "disabled":
                    is_on = False
                if budget:
                    extra_details.append(f"budget={budget}")
            enable_thinking = self.extra_body.get("enable_thinking")
            if isinstance(enable_thinking, bool):
                is_on = enable_thinking
            # Anthropic-via-OpenAI proxy flat thinking params.
            thinking_enabled = self.extra_body.get("thinking_enabled")
            if isinstance(thinking_enabled, bool):
                is_on = thinking_enabled
            eb_effort = self.extra_body.get("reasoning_effort")
            if (
                eb_effort
                and effort is None
                and str(eb_effort).lower() not in ("off", "none", "disabled")
            ):
                effort = eb_effort
        if is_on is None:
            return "default"
        if not is_on:
            return "off"
        if effort:
            return str(effort)
        return f"on({', '.join(extra_details)})" if extra_details else "on"

    def _is_deepseek_thinking_request(self, request_params: Dict[str, Any]) -> bool:
        """Return True when DeepSeek thinking mode is explicitly enabled."""
        if self.provider != "DeepSeek":
            return False
        extra_body = request_params.get("extra_body")
        if not isinstance(extra_body, dict):
            return False
        thinking = extra_body.get("thinking")
        if not isinstance(thinking, dict):
            return False
        return thinking.get("type") == "enabled"

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Returns keyword arguments for API requests."""
        request_params = self._collect_request_params()
        if self.tools is not None:
            tools_for_api = self.get_tools_for_api()
            if tools_for_api:
                request_params["tools"] = tools_for_api
                tool_choice = self.get_tool_choice()
                request_params["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if self.request_params is not None:
            request_params.update(self.request_params)
        if self._is_deepseek_thinking_request(request_params):
            for param_name in _DEEPSEEK_THINKING_UNSUPPORTED_PARAMS:
                request_params.pop(param_name, None)
        cache_key = self._prompt_cache_key()
        if cache_key is not None:
            request_params.setdefault("prompt_cache_key", cache_key)
        return request_params

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        model_dict = super().to_dict()
        collected = self._collect_request_params()
        # Special handling for response_format
        if "response_format" in collected:
            rf = collected["response_format"]
            collected["response_format"] = rf if isinstance(rf, dict) else str(rf)
        model_dict.update(collected)
        if self.tools is not None:
            tools_for_api = self.get_tools_for_api()
            if tools_for_api:
                model_dict["tools"] = tools_for_api
                tool_choice = self.get_tool_choice()
                model_dict["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        cleaned_dict = {k: v for k, v in model_dict.items() if v is not None}
        return cleaned_dict

    def format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message into the format expected by OpenAI."""
        if message.role == "user":
            if message.images is not None:
                message = self.add_images_to_message(message=message, images=message.images)

        if message.audio is not None:
            message = self.add_audio_to_message(message=message, audio=message.audio)

        wire = message.to_model_dict()
        # reasoning_content is response-only: providers accept it on input but
        # ignore it (Taiji/hy4 measured: prompt_tokens unchanged with 215KB of
        # it; DeepSeek documents the same). History messages carrying it would
        # ship megabytes of dead weight every request, so strip it at the wire
        # boundary. Persistence/replay keep it via to_replay_dict().
        if "reasoning_content" in wire:
            del wire["reasoning_content"]
        # A thinking block without a signature is never valid to replay
        # (the signer is gone). Drop it regardless of model id.
        content = wire.get("content")
        if isinstance(content, list):
            filtered = [b for b in content if not _is_unsigned_thinking_block(b)]
            if filtered != content:
                wire["content"] = filtered
        return wire

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format history for the chat wire, omitting assistant turns with nothing to send.

        A turn that spent its whole output budget on reasoning keeps only
        ``reasoning_content``, which this wire strips. What remains is an
        assistant message with no content, no tool call and no audio: it
        carries no information, and some gateways reject it (4000 ``message has no
        content``), which kills every later request of the session. The
        Anthropic path skips such turns the same way.

        A Claude-native tool round (an in-turn fallback from a Claude primary)
        is rewritten to this wire's shape first; see
        ``claude_blocks_to_shared_shape``.

        A tool call that carries no ``extra_content`` was issued by some other
        model (Gemini returns its ``thought_signature`` there on every call).
        Gemini validates the signature of the calls in the current turn and
        rejects an unsigned one (400 "Function call is missing a
        thought_signature"), which is exactly what an in-turn fallback to
        Gemini replays. Such calls get Google's documented placeholder
        signature; the other chat endpoints ignore the field (measured on
        opus / glm / deepseek / hy4 / gpt chat endpoints).
        """
        # Imported here: agentica.agent imports this module.
        from agentica.agent.history_filter import claude_blocks_to_shared_shape

        formatted = []
        for message in claude_blocks_to_shared_shape(messages):
            wire = self.format_message(message)
            if (
                wire["role"] == "assistant"
                and not wire.get("tool_calls")
                and not wire.get("audio")
                and not _has_cacheable_content(wire["content"])
            ):
                continue
            if wire.get("tool_calls"):
                wire["tool_calls"] = [
                    call if "extra_content" in call else {**call, "extra_content": _FOREIGN_CALL_SIGNATURE}
                    for call in wire["tool_calls"]
                ]
            formatted.append(wire)
        return formatted

    # ── Prompt caching (Anthropic-via-OpenAI proxies) ──

    def _prompt_cache_key(self) -> Optional[str]:
        """Routing-affinity key for OpenAI's implicit prompt cache.

        Scoped to the conversation: one session shares a prompt prefix, and
        OpenAI asks that a key stay under roughly 15 requests per minute, which
        a single session cannot exceed. Falls back to the endpoint's sticky
        routing id when the caller never set a session (a bare SDK ``Agent``),
        since an arbitrary stable key still beats none — without one, requests
        are spread across machines for load balancing and the cache goes with
        them.
        """
        if not self.enable_prompt_cache_key:
            return None
        # Resolved per call, not memoized — a first request before
        # update_model() sets session_id must not pin the fallback id.
        return resolve_cache_session_id(session_id=self.session_id, base_url=self.base_url)

    def _with_cache_session_header(self, existing: Any) -> Any:
        """Merge a stable per-run session id into extra_headers for sticky routing.

        Cache hits require consecutive requests to land on the same backend.
        Some proxies route by a request header (e.g. ``X-Session-Id``);
        the id is loaded-or-created once per base_url and persisted in
        ``~/.agentica/cache/cache_routing.json`` so cache affinity survives CLI
        restarts. Only merges into a dict-typed extra_headers; other types are
        returned unchanged (we cannot safely merge into openai Header objects).

        Session-scoped when there is a live session (matches ``_prompt_cache_key``
        and the Anthropic path): one conversation shares one prompt prefix, and
        a fresh session re-roles the routing dice — the escape hatch when a
        proxy pins you to a misbehaving backend. The endpoint-persistent id is
        the fallback for bare SDK use with no session.
        """
        if not self.cache_control_session_header:
            return existing
        # Resolved per call (see cache_routing.resolve_cache_session_id): a
        # first request before update_model() sets session_id must not pin the
        # fallback id for the rest of the run.
        sid = resolve_cache_session_id(session_id=self.session_id, base_url=self.base_url)
        if isinstance(existing, dict):
            headers = dict(existing)
        elif existing is None:
            headers = {}
        else:
            return existing
        headers.setdefault(self.cache_control_session_header, sid)
        return headers

    def _arm_cache_keepalive(self) -> None:
        """(Re)start the idle cache keep-alive timer after a successful request.

        Each call bumps a generation counter; any older worker thread exits when
        it notices, so at most one ping chain is ever in flight. The chain stops
        on its own after ``cache_keepalive_max_pings`` pings with no new traffic.
        Requires a sticky routing header: without backend affinity a ping lands
        on a random backend and becomes a full-price cache WRITE, which would
        multiply cost instead of saving it.
        """
        if not (
            self.enable_cache_control
            and self.cache_keepalive
            and self.cache_control_session_header
            and self._last_cache_payload
        ):
            return
        self._cache_keepalive_generation += 1
        worker = threading.Thread(
            target=self._cache_keepalive_loop,
            args=(self._cache_keepalive_generation,),
            daemon=True,
        )
        worker.start()

    def _cache_keepalive_loop(self, generation: int) -> None:
        for _ in range(max(self.cache_keepalive_max_pings, 0)):
            time.sleep(self.cache_keepalive_interval)
            if generation != self._cache_keepalive_generation:
                return  # newer traffic armed a fresh worker; this chain is done
            try:
                read_tokens = asyncio.run(self._cache_keepalive_ping())
                if read_tokens:
                    logger.debug(f"cache keepalive ping refreshed prefix (read={read_tokens} tokens)")
            except Exception as e:
                # File only. This thread outlives the turn, and a console
                # handler (or /debug) would paint the provider error into the
                # transcript after the answer has already finished.
                log_file_only("cache keepalive ping failed, stopping chain: %s", e, exc_info=e)
                return

    async def _cache_keepalive_ping(self) -> Optional[int]:
        """Re-read the cached tools+system prefix with a minimal request.

        A cache read refreshes the entry TTL (verified on a proxy that does so), so pinging
        during idle think-time keeps the expensive prefix alive instead of paying
        a full re-write on the next real request. Runs in a worker thread with
        its own event loop and a throwaway client so it never touches the main
        loop's client. Returns the number of cache-read tokens, if reported.
        """
        payload = self._last_cache_payload
        if payload is None:
            return None
        messages, request_kwargs = payload
        system_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "system"]
        if not system_messages:
            return None
        ping_kwargs: Dict[str, Any] = {
            "model": self.id,
            "messages": system_messages + [{"role": "user", "content": "cache keepalive"}],
            "max_tokens": 1,
        }
        if request_kwargs.get("tools"):
            ping_kwargs["tools"] = request_kwargs["tools"]
        if request_kwargs.get("extra_headers"):
            ping_kwargs["extra_headers"] = request_kwargs["extra_headers"]
        if self.extra_body:
            # Mirror the main request shape (e.g. thinking toggles) so strict
            # backends don't reject the ping for differing params.
            ping_kwargs["extra_body"] = self.extra_body
        http_client = httpx.AsyncClient(limits=httpx.Limits(max_connections=10, max_keepalive_connections=2))
        try:
            client_params = self.get_client_params()
            client_params.pop("http_client", None)  # we supply the throwaway one
            client = AsyncOpenAIClient(http_client=http_client, **client_params)
            response = await self._create_with_auto_cap(
                lambda kwargs: client.chat.completions.create(**kwargs),
                ping_kwargs,
            )
        finally:
            await http_client.aclose()
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        return getattr(details, "cache_read_tokens", None) if details else None

    def _apply_cache_control(self, formatted: List[Dict[str, Any]], request_kwargs: Dict[str, Any]) -> tuple:
        """Inject Anthropic-style cache_control breakpoints within the 4-breakpoint budget.

        Returns ``(messages, request_kwargs)`` (copies, original not mutated).
        When caching is off, warns if the request already carries manual
        cache_control blocks (so the user knows to flip enable_cache_control on).
        """
        tools = request_kwargs.get("tools")
        if not self.enable_cache_control:
            if _request_has_cache_control(formatted, tools):
                logger.warning(
                    "cache_control blocks found in the request but "
                    "enable_cache_control is off; the backend may ignore them. "
                    "Set enable_cache_control=True to turn on prompt caching."
                )
            # The volatile marker only drives the cache split, which is off on
            # this path; strip it so it never pollutes the prompt.
            formatted = [_strip_volatile_marker_from_message(m) for m in formatted]
            return formatted, request_kwargs

        budget = _CACHE_CONTROL_BREAKPOINT_BUDGET
        new_kwargs = dict(request_kwargs)

        # 1) Tools block: tag the last tool definition.
        if tools:
            new_kwargs["tools"] = _tag_tools_cache_control(tools)
            budget -= 1

        # 2) System message: tag the stable prefix; if the prompt builder
        #    marked a volatile tail (memory/experiences/summary/date), split it
        #    out so tail churn never invalidates the cached stable prefix.
        out = list(formatted)
        for i, m in enumerate(out):
            if isinstance(m, dict) and m.get("role") == "system":
                tagged = dict(m)
                tagged["content"] = _tag_system_content_cache_control(m.get("content"))
                out[i] = tagged
                budget -= 1
                break

        # 3) Trailing USER/TOOL messages: rolling cache window. Per typical
        #    proxy cache examples, both user turns and tool results carry cache_control
        #    breakpoints (block-list form). Assistant turns are skipped: in
        #    OpenAI format their tool_calls live in a separate field, not in
        #    content blocks, so there is no block to attach cache_control to.
        #    Empty-content messages are skipped without consuming budget (a
        #    cache breakpoint on empty content is invalid and would be dropped).
        if budget > 0:
            k = min(max(self.cache_control_messages, 0), budget)
            tagged = 0
            for j in range(len(out) - 1, -1, -1):
                if tagged >= k:
                    break
                m = out[j]
                if not (isinstance(m, dict) and m.get("role") in ("user", "tool")):
                    continue
                if not _has_cacheable_content(m.get("content")):
                    continue
                tagged_msg = dict(m)
                tagged_msg["content"] = _tag_content_block_cache_control(m.get("content"))
                out[j] = tagged_msg
                tagged += 1

        # Sticky routing so consecutive invokes hit the same backend cache.
        if self.cache_control_session_header:
            new_kwargs["extra_headers"] = self._with_cache_session_header(new_kwargs.get("extra_headers"))
        # Remember the tagged payload for cache keep-alive pings.
        self._last_cache_payload = (out, new_kwargs)
        return out, new_kwargs

    async def _send_with_system_position_repair(
        self,
        formatted: List[Dict[str, Any]],
        send: Callable[[List[Dict[str, Any]]], Awaitable[Any]],
    ) -> Any:
        """Repair a client-side mid-conversation system placement once."""
        try:
            return await send(formatted)
        except Exception as error:
            if _SYSTEM_POSITION_ERROR not in str(error).lower():
                raise
            repaired = _repair_claude_system_positions(formatted)
            if repaired is None:
                logger.warning(
                    "[system_position] %s rejected messages, but the client wire "
                    "has no safely repairable mid-conversation system directive; "
                    "not retrying (roles=%s, system_indexes=%s): %s",
                    self.id,
                    [message.get("role") for message in formatted],
                    [i for i, message in enumerate(formatted) if message.get("role") == "system"],
                    error,
                )
                raise
            self._requires_system_position_repair = True
            logger.warning(
                "[system_position] %s rejected a client-side system placement; "
                "retrying once after structural repair "
                "(roles=%s -> %s, system_indexes=%s -> %s): %s",
                self.id,
                [message.get("role") for message in formatted],
                [message.get("role") for message in repaired],
                [i for i, message in enumerate(formatted) if message.get("role") == "system"],
                [i for i, message in enumerate(repaired) if message.get("role") == "system"],
                error,
            )
            return await send(repaired)

    @override
    async def invoke(self, messages: List[Message]) -> Union[ChatCompletion, ParsedChatCompletion]:
        """Send a chat completion request to the OpenAI API (async-only)."""
        formatted = self.format_messages(messages)
        if self._requires_system_position_repair:
            formatted = _repair_claude_system_positions(formatted) or formatted
        # Inject Anthropic-style cache_control breakpoints for OpenAI-compatible
        # proxies that front Claude. No-op when caching is off.
        formatted, request_kwargs = self._apply_cache_control(formatted, self.request_kwargs)

        if self.response_format is not None and self.use_structured_outputs:
            try:
                if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                    async def _parse(payload: List[Dict[str, Any]]) -> Any:
                        return await self._create_with_auto_cap(
                            lambda kwargs: self.get_client().beta.chat.completions.parse(
                                model=self.id,
                                messages=payload,  # type: ignore
                                **kwargs,
                            ),
                            request_kwargs,
                        )

                    response = await self._send_with_system_position_repair(formatted, _parse)
                    self._arm_cache_keepalive()
                    return response
                else:
                    raise ValueError("response_format must be a subclass of BaseModel if use_structured_outputs=True")
            except Exception as e:
                self._learn_context_limit_from_error(str(e))
                logger.error(f"Error from OpenAI API structured outputs: {e}")
                raise

        async def _create(payload: List[Dict[str, Any]]) -> Any:
            return await self._create_with_auto_cap(
                lambda kwargs: self.get_client().chat.completions.create(
                    model=self.id,
                    messages=payload,  # type: ignore
                    **kwargs,
                ),
                request_kwargs,
            )

        try:
            response = await self._send_with_system_position_repair(formatted, _create)
            self._arm_cache_keepalive()
            return response
        except Exception as e:
            self._learn_context_limit_from_error(str(e))
            raise

    @override
    async def invoke_stream(self, messages: List[Message]) -> AsyncIterator[ChatCompletionChunk]:
        """Send a streaming chat completion request to the OpenAI API (async-only).

        Wraps the open+iterate cycle in ``stream_with_retry``, which reopens the
        stream on a transient failure up to ``max_retries`` times (2 by default;
        set ``OpenAIChat(max_retries=0)`` to disable). Once the first chunk has
        been yielded, any subsequent error propagates verbatim because retrying
        would duplicate output.
        """
        formatted = self.format_messages(messages)
        if self._requires_system_position_repair:
            formatted = _repair_claude_system_positions(formatted) or formatted
        formatted, request_kwargs = self._apply_cache_control(formatted, self.request_kwargs)

        async def _create_stream(payload: List[Dict[str, Any]]) -> Any:
            return await self._create_with_auto_cap(
                lambda kwargs: self.get_client().chat.completions.create(
                    model=self.id,
                    messages=payload,  # type: ignore
                    stream=True,
                    stream_options={"include_usage": True},
                    **kwargs,
                ),
                request_kwargs,
            )

        async def _open() -> AsyncIterator[ChatCompletionChunk]:
            try:
                return await self._send_with_system_position_repair(formatted, _create_stream)
            except Exception as e:
                self._learn_context_limit_from_error(str(e))
                raise

        async for chunk in stream_with_retry(
            _open,
            extra_substrings=self.extra_retryable_substrings,
            max_retries=self.max_retries or 0,
            provider_label=f"openai/{self.id}",
        ):
            yield chunk
        self._arm_cache_keepalive()

    # handle_tool_calls and handle_stream_tool_calls are inherited from Model base class.

    @staticmethod
    def _provider_reasoning(obj: Any) -> Optional[str]:
        """DeepSeek uses ``reasoning_content``; NVIDIA uses ``reasoning``.

        ``Message.reasoning_content`` is ``Optional[str]`` — skip anything else.
        """
        for name in ("reasoning_content", "reasoning"):
            if hasattr(obj, name):
                value = getattr(obj, name)
                if isinstance(value, str):
                    return value
        return None

    def create_assistant_message(
        self,
        response_message: ChatCompletionMessage,
        metrics: Metrics,
        response_usage: Optional[CompletionUsage],
    ) -> Message:
        """Create an assistant message from the response."""
        content = response_message.content
        if isinstance(content, str):
            content = content.lstrip("\n")
        assistant_message = Message(
            role=response_message.role or "assistant",
            content=content,
        )
        assistant_message.provider_data = response_message.model_dump(exclude_none=True)
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            try:
                # ``type`` is the discriminator the runner dispatches on, and
                # ``get_function_call_for_tool_call`` requires it to be
                # "function". Gemini leaves it null; left as-is, the call is
                # dropped and the model is told the tool does not exist.
                assistant_message.tool_calls = [
                    {**dump, "type": dump.get("type") or "function"}
                    for dump in (t.model_dump() for t in response_message.tool_calls)
                ]
            except Exception as e:
                logger.warning(f"Error processing tool calls: {e}")
        if hasattr(response_message, "audio") and response_message.audio is not None:
            try:
                assistant_message.audio = response_message.audio.model_dump()
            except Exception as e:
                logger.warning(f"Error processing audio: {e}")

        reasoning = self._provider_reasoning(response_message)
        if reasoning is not None:
            assistant_message.reasoning_content = reasoning

        self.update_usage_metrics(assistant_message, metrics, response_usage)
        return assistant_message

    @override
    async def response(self, messages: List[Message]) -> ModelResponse:
        """Generate a response from OpenAI (async-only)."""
        self.sanitize_messages(messages)
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()

        metrics.response_timer.start()
        response: Union[ChatCompletion, ParsedChatCompletion] = await self.invoke(messages=messages)
        metrics.response_timer.stop()

        first_choice = require_first_choice(response, context=f"OpenAIChat '{self.id}'")
        response_message: ChatCompletionMessage = first_choice.message
        # Defensive: usage may be None when stream_options are not set or API omits it
        response_usage: Optional[CompletionUsage] = getattr(response, "usage", None)

        # Parse structured outputs — fallback to text content on validation failure
        try:
            if (
                self.response_format is not None
                and self.use_structured_outputs
                and issubclass(self.response_format, BaseModel)
            ):
                parsed_object = response_message.parsed  # type: ignore
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            logger.warning(f"Structured output parse failed for model '{self.id}', falling back to text content: {e}")

        assistant_message = self.create_assistant_message(
            response_message=response_message, metrics=metrics, response_usage=response_usage
        )
        if (
            self.run_tools
            and self.functions
            and not assistant_message.tool_calls
            and isinstance(assistant_message.content, str)
            and self._uses_claude_text_tool_call_compatibility()
        ):
            text_tool_calls = self._parse_claude_text_tool_calls(assistant_message.content)
            if text_tool_calls:
                assistant_message.content = None
                assistant_message.tool_calls = text_tool_calls

        messages.append(assistant_message)
        assistant_message.log()
        metrics.log()

        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        if assistant_message.audio is not None:
            model_response.audio = assistant_message.audio

        # Expose finish_reason so Runner's agentic loop can detect
        # truncated output (finish_reason == "length") for max_tokens recovery.
        model_response.finish_reason = first_choice.finish_reason
        self.last_finish_reason = model_response.finish_reason
        assistant_message.finish_reason = model_response.finish_reason

        tool_role = "tool"
        if (
            await self.handle_tool_calls(
                assistant_message=assistant_message,
                messages=messages,
                model_response=model_response,
                tool_role=tool_role,
            )
            is not None
        ):
            return model_response
        return model_response

    def add_response_usage_to_metrics(self, metrics: Metrics, response_usage: CompletionUsage):
        metrics.input_tokens = response_usage.prompt_tokens
        metrics.prompt_tokens = response_usage.prompt_tokens
        metrics.output_tokens = response_usage.completion_tokens
        metrics.completion_tokens = response_usage.completion_tokens
        metrics.total_tokens = response_usage.total_tokens
        if response_usage.prompt_tokens_details is not None:
            if isinstance(response_usage.prompt_tokens_details, dict):
                metrics.prompt_tokens_details = response_usage.prompt_tokens_details
            elif isinstance(response_usage.prompt_tokens_details, BaseModel):
                metrics.prompt_tokens_details = response_usage.prompt_tokens_details.model_dump(exclude_none=True)
        if response_usage.completion_tokens_details is not None:
            if isinstance(response_usage.completion_tokens_details, dict):
                metrics.completion_tokens_details = response_usage.completion_tokens_details
            elif isinstance(response_usage.completion_tokens_details, BaseModel):
                metrics.completion_tokens_details = response_usage.completion_tokens_details.model_dump(
                    exclude_none=True
                )

    @override
    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """Generate a streaming response from OpenAI (async-only)."""
        self.sanitize_messages(messages)
        self._log_messages(messages)
        stream_data: StreamData = StreamData()
        metrics: Metrics = Metrics()
        buffer_content_for_claude_xml = bool(
            self.run_tools
            and self.functions
            and self._uses_claude_text_tool_call_compatibility()
        )

        stream_finish_reason: Optional[str] = None
        metrics.response_timer.start()
        async for response in self.invoke_stream(messages=messages):
            if response.choices and len(response.choices) > 0:
                if metrics.completion_tokens is None:
                    metrics.completion_tokens = 0

                metrics.completion_tokens += 1
                if metrics.completion_tokens == 1:
                    metrics.time_to_first_token = metrics.response_timer.elapsed

                # Capture finish_reason from the final chunk
                if response.choices[0].finish_reason is not None:
                    stream_finish_reason = response.choices[0].finish_reason

                response_delta: ChoiceDelta = response.choices[0].delta

                reasoning_delta = self._provider_reasoning(response_delta)
                if reasoning_delta:
                    stream_data.response_reasoning_content += reasoning_delta
                    yield ModelResponse(reasoning_content=reasoning_delta)

                if hasattr(response_delta, "content") and response_delta.content:
                    stream_data.response_content += response_delta.content
                    if not buffer_content_for_claude_xml:
                        yield ModelResponse(content=response_delta.content)

                if hasattr(response_delta, "audio"):
                    response_audio = response_delta.audio
                    stream_data.response_audio = response_audio
                    yield ModelResponse(audio=response_audio)

                if hasattr(response_delta, "tool_calls") and response_delta.tool_calls:
                    if stream_data.response_tool_calls is None:
                        stream_data.response_tool_calls = []
                    stream_data.response_tool_calls.extend(response_delta.tool_calls)

            if response.usage:
                self.add_response_usage_to_metrics(metrics=metrics, response_usage=response.usage)
        metrics.response_timer.stop()

        assistant_message = Message(role="assistant")
        if stream_data.response_content:
            assistant_message.content = stream_data.response_content.lstrip("\n")

        if stream_data.response_reasoning_content:
            assistant_message.reasoning_content = stream_data.response_reasoning_content

        if stream_data.response_audio:
            assistant_message.audio = stream_data.response_audio

        if stream_data.response_tool_calls:
            _tool_calls = self.build_tool_calls(stream_data.response_tool_calls)
            if len(_tool_calls) > 0:
                assistant_message.tool_calls = _tool_calls

        if (
            buffer_content_for_claude_xml
            and not assistant_message.tool_calls
            and isinstance(assistant_message.content, str)
        ):
            text_tool_calls = self._parse_claude_text_tool_calls(assistant_message.content)
            if text_tool_calls:
                assistant_message.content = None
                assistant_message.tool_calls = text_tool_calls

        if buffer_content_for_claude_xml and assistant_message.content:
            yield ModelResponse(content=assistant_message.get_content_string())

        self.update_stream_metrics(assistant_message=assistant_message, metrics=metrics)

        # Expose finish_reason so Runner's agentic loop can detect truncated output
        self.last_finish_reason = stream_finish_reason
        assistant_message.finish_reason = stream_finish_reason

        messages.append(assistant_message)
        assistant_message.log()
        metrics.log()

        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            tool_role = "tool"
            async for tool_call_response in self.handle_stream_tool_calls(
                assistant_message=assistant_message, messages=messages, tool_role=tool_role
            ):
                yield tool_call_response

    def build_tool_calls(self, tool_calls_data: List[ChoiceDeltaToolCall]) -> List[Dict[str, Any]]:
        """Build tool calls from streaming tool call deltas.

        Standard OpenAI streams number each tool_call delta with a stable
        ``index`` so fragments (name in one chunk, arguments split across the
        next) can be reassembled positionally. Some OpenAI-compatible proxies —
        notably ones that translate Anthropic Claude tool_use into the OpenAI
        wire format — omit ``index`` (send ``None``) or hold it at
        0 while relying on ``id`` to delimit calls. The original positional-only
        logic raised ``TypeError`` on ``None`` index, dropping every tool call
        and silently degrading the turn to plain text.

        This reassembles defensively:
          * ``index`` present  -> positional accumulation (standard path).
          * ``index`` missing  -> group by ``id``; a fragment with a fresh id
            starts a new call, id-less fragments append to the current one.
        Missing name/arguments/id/type fragments are tolerated throughout.
        """
        tool_calls: List[Dict[str, Any]] = []
        # Maps an id -> position in ``tool_calls`` for the index-less fallback.
        id_to_pos: Dict[str, int] = {}

        def _passthrough(_tool_call) -> Dict[str, Any]:
            """Provider fields a delta carries outside the OpenAI schema.

            Gemini streams its ``thought_signature`` (and ``extra_content``
            wrapping it) as extra keys on the tool-call delta. They are the
            provider's replay state: dropping them makes Gemini reject its own
            earlier call on the next request (400 "missing a
            thought_signature"). Carried on the stored call so the wire gets
            them back.
            """
            return {k: v for k, v in (getattr(_tool_call, "model_extra", None) or {}).items() if v is not None}

        def _new_entry(tid, ttype, fname, fargs, extra) -> Dict[str, Any]:
            return {
                "id": tid or "",
                "type": ttype or "function",
                "function": {"name": fname or "", "arguments": fargs or ""},
                **extra,
            }

        def _merge_into(entry: Dict[str, Any], tid, ttype, fname, fargs, extra) -> None:
            if fname:
                entry["function"]["name"] += fname
            if fargs:
                entry["function"]["arguments"] += fargs
            if tid:
                entry["id"] = tid
            if ttype:
                entry["type"] = ttype
            entry.update(extra)

        for _tool_call in tool_calls_data:
            _index = getattr(_tool_call, "index", None)
            _tool_call_id = getattr(_tool_call, "id", None)
            _tool_call_type = getattr(_tool_call, "type", None)
            _fn = getattr(_tool_call, "function", None)
            _function_name = _fn.name if _fn else None
            _function_arguments = _fn.arguments if _fn else None

            if isinstance(_index, int):
                # Standard positional path.
                if len(tool_calls) <= _index:
                    tool_calls.extend([{}] * (_index - len(tool_calls) + 1))
                entry = tool_calls[_index]
                if not entry:
                    tool_calls[_index] = _new_entry(
                        _tool_call_id, _tool_call_type, _function_name, _function_arguments, _passthrough(_tool_call)
                    )
                else:
                    _merge_into(
                        entry, _tool_call_id, _tool_call_type, _function_name, _function_arguments,
                        _passthrough(_tool_call),
                    )
                continue

            # ── index-less fallback (loose proxy) ──
            # A fragment carrying a NEW id begins a new call; otherwise it
            # continues the most recent call (or starts the first one).
            if _tool_call_id and _tool_call_id in id_to_pos:
                _merge_into(
                    tool_calls[id_to_pos[_tool_call_id]],
                    _tool_call_id,
                    _tool_call_type,
                    _function_name,
                    _function_arguments,
                    _passthrough(_tool_call),
                )
            elif _tool_call_id:
                tool_calls.append(_new_entry(
                    _tool_call_id, _tool_call_type, _function_name, _function_arguments, _passthrough(_tool_call)
                ))
                id_to_pos[_tool_call_id] = len(tool_calls) - 1
            elif tool_calls:
                # id-less continuation of the current (last) call.
                _merge_into(
                    tool_calls[-1], _tool_call_id, _tool_call_type, _function_name, _function_arguments,
                    _passthrough(_tool_call),
                )
            else:
                # First fragment has no id and no index — open a fresh call.
                tool_calls.append(_new_entry(
                    _tool_call_id, _tool_call_type, _function_name, _function_arguments, _passthrough(_tool_call)
                ))

        # Drop any empty slots left by sparse positional indices.
        return [tc for tc in tool_calls if tc]
