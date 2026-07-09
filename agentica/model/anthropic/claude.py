import json
from os import getenv
from dataclasses import dataclass, field
import sys
from typing import Optional, List, AsyncIterator, Dict, Any, Union, Tuple

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import asyncio

from pydantic import BaseModel

from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.metrics import Metrics
from agentica.model.response import ModelResponse
from agentica.model.anthropic._max_tokens import (
    resolve_anthropic_messages_max_tokens,
    parse_available_output_tokens_from_error,
)
from agentica.tools.base import FunctionCall, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer

# Safety margin subtracted from the API-reported ``available_tokens`` when
# retrying after a "max_tokens too large given prompt" error. Without this,
# the retry sometimes hits the same boundary again because token counting
# is approximate.
_MAX_TOKENS_RETRY_SAFETY_MARGIN = 64

try:
    from anthropic import AsyncAnthropic as AnthropicClient
    from anthropic.types import (
        Message as AnthropicMessage,
        TextBlock,
        ToolUseBlock,
        Usage,
        TextDelta,
        ThinkingBlock,
        RedactedThinkingBlock,
        ThinkingDelta,
        SignatureDelta,
    )
    from anthropic.lib.streaming._types import (
        MessageStopEvent,
        RawContentBlockDeltaEvent,
        ContentBlockStopEvent,
    )

    # The high-level ``messages.stream()`` helper yields *parsed* event
    # subtypes (ParsedContentBlockStopEvent / ParsedMessageStopEvent) that do
    # NOT inherit from the base ContentBlockStopEvent / MessageStopEvent. If we
    # only isinstance-check the base types, tool_use blocks and stop_reason are
    # silently dropped from the stream (text still flows, but tools never fire).
    # Import the parsed variants and match against both. Guard with try/except
    # so older SDKs without these names still import.
    try:
        from anthropic.lib.streaming._types import (
            ParsedContentBlockStopEvent,
            ParsedMessageStopEvent,
        )
    except ImportError:
        ParsedContentBlockStopEvent = None
        ParsedMessageStopEvent = None
    # Build isinstance tuples that include only the classes that resolved.
    _CONTENT_BLOCK_STOP_TYPES = tuple(t for t in (ContentBlockStopEvent, ParsedContentBlockStopEvent) if t is not None)
    _MESSAGE_STOP_TYPES = tuple(t for t in (MessageStopEvent, ParsedMessageStopEvent) if t is not None)
except (ModuleNotFoundError, ImportError):
    AnthropicClient = None
    AnthropicMessage = None
    TextBlock = None
    ToolUseBlock = None
    Usage = None
    TextDelta = None
    ThinkingBlock = None
    RedactedThinkingBlock = None
    ThinkingDelta = None
    SignatureDelta = None
    MessageStopEvent = None
    RawContentBlockDeltaEvent = None
    ContentBlockStopEvent = None
    ParsedContentBlockStopEvent = None
    ParsedMessageStopEvent = None
    _CONTENT_BLOCK_STOP_TYPES = ()
    _MESSAGE_STOP_TYPES = ()


@dataclass
class MessageData:
    response_content: str = ""
    response_reasoning_content: str = ""
    response_block: List[Any] = field(default_factory=list)
    response_block_content: Optional[Any] = None
    response_usage: Optional[Usage] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_ids: List[str] = field(default_factory=list)


@dataclass
class Claude(Model):
    """
    A class representing Anthropic Claude model.

    For more information, see: https://docs.anthropic.com/en/api/messages
    """

    id: str = "claude-3-5-sonnet-20241022"
    name: str = "Claude"
    provider: str = "Anthropic"
    context_window: int = 200000

    # Request parameters
    # max_tokens is Anthropic's OUTPUT cap (not the context window). Required by
    # the SDK on every call; when None, the per-model ceiling from
    # _ANTHROPIC_OUTPUT_LIMITS is used. See _max_tokens.py for the table.
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # Extended thinking. Two shapes are supported:
    #   * Fixed budget (older models): {"type": "enabled", "budget_tokens": N}
    #   * Adaptive (opus-4-7+/sonnet-4-6+): {"type": "adaptive"} paired with
    #     output_config.effort. Set this directly for full control, or use the
    #     ``reasoning_effort`` shortcut below which builds the adaptive form.
    thinking: Optional[Dict[str, Any]] = None
    # Adaptive-thinking effort shortcut: "low"|"medium"|"high"|"xhigh"|"max".
    # When set (and ``thinking`` is not already given), prepare_request_kwargs
    # enables adaptive thinking via thinking={"type":"adaptive"} +
    # output_config={"effort": ...}. Adaptive thinking requires temperature=1,
    # which is enforced automatically. Mirrors the CLI reasoning_effort knob so
    # one setting controls thinking depth across all providers.
    reasoning_effort: Optional[str] = None
    # Adaptive-thinking display: "summarized" (return a thinking summary) or
    # "omitted" (hide thinking). Only meaningful when adaptive thinking is on.
    # Model default varies: opus-4-7/opus-4-8 default to "omitted" (thinking is
    # NOT surfaced regardless of what you pass), while opus-4-6 / sonnet-4-6
    # default to "summarized". Set this to force a value when the model supports
    # it. Rides along in extra_body next to output_config.
    thinking_display: Optional[str] = None
    request_params: Optional[Dict[str, Any]] = None

    # Client parameters
    api_key: Optional[str] = None
    # Custom Anthropic-compatible endpoint (e.g. a corporate proxy that
    # forwards /v1/messages). When set, it is passed to the SDK client and the
    # bearer Authorization header is seeded (some proxies auth by header
    # rather than the x-api-key the SDK sends by default).
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    default_headers: Optional[Dict[str, str]] = None
    client_params: Optional[Dict[str, Any]] = None

    # Anthropic client
    client: Optional[AnthropicClient] = None
    # The event loop the cached client is bound to. AsyncAnthropic binds its
    # httpx pool to the loop it was created on; the CLI runs each turn on a
    # fresh loop (run_stream_sync -> asyncio.run), so we rebuild the client when
    # the loop changes/closes instead of reusing a pool bound to a dead loop.
    # A hard reference (not id()) is required: loop id()s get recycled after GC,
    # so a stale-but-same-id loop would be mistaken for the live one.
    _client_loop: Optional[Any] = field(default=None, init=False, repr=False, compare=False)

    # Structured output support
    use_structured_outputs: bool = False
    supports_structured_outputs: bool = True

    # Prompt caching: inject cache_control breakpoints into system message
    # and conversation history to reduce input token costs on multi-turn runs.
    # Anthropic charges 0.1x for cache reads (vs 1.25x write on first request).
    # Default True -- automatic caching with zero user friction.
    enable_cache_control: bool = True

    def _get_structured_output_tool(self) -> Optional[Dict[str, Any]]:
        """Build a synthetic tool from response_format for structured output via tool_use."""
        if (
            self.response_format is not None
            and self.use_structured_outputs
            and isinstance(self.response_format, type)
            and issubclass(self.response_format, BaseModel)
        ):
            schema = self.response_format.model_json_schema()
            # Remove $defs at top level, inline refs not needed for Anthropic tool schema
            schema.pop("$defs", None)
            return {
                "name": "structured_output",
                "description": f"Return structured output as {self.response_format.__name__}",
                "input_schema": schema,
            }
        return None

    def get_client(self) -> AnthropicClient:
        """
        Returns an instance of the Anthropic client.

        Returns:
            AnthropicClient: An instance of the Anthropic client
        """
        if AnthropicClient is None:
            raise ImportError("`anthropic` not installed. Please install using `pip install anthropic`")

        # Reuse the client only within the SAME event loop. AsyncAnthropic binds
        # its httpx connection pool to the loop it was created on; the CLI's
        # run_stream_sync runs each turn on a fresh loop (asyncio.run), so a
        # client cached across runs would carry a pool bound to a now-closed
        # loop — reuse would crash and, at GC time, the stale pool's aclose()
        # prints "RuntimeError: Event loop is closed". Reuse the client only
        # when the current running loop is the exact same object we built it on
        # (identity, not id(), since loop id()s are recycled after GC).
        # An externally-injected client (tests / manual override) has no
        # associated _client_loop; honour it unconditionally. Otherwise reuse
        # the auto-built client only while the loop it was created on is still
        # the current, open loop.
        if self.client is not None and self._client_loop is None:
            return self.client
        try:
            _loop: Optional[Any] = asyncio.get_running_loop()
        except RuntimeError:
            _loop = None
        if self.client is not None and _loop is not None and self._client_loop is _loop and not _loop.is_closed():
            return self.client

        self.api_key = self.api_key or getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not set. Please set the ANTHROPIC_API_KEY environment variable.")

        _client_params: Dict[str, Any] = {}
        # Set client parameters if they are provided
        if self.api_key:
            _client_params["api_key"] = self.api_key
        # Custom endpoint (corporate proxy). Also seed the bearer header —
        # proxies like Venus authenticate via Authorization: Bearer <token>
        # rather than the SDK's default x-api-key, so send both to be safe.
        if self.base_url:
            _client_params["base_url"] = self.base_url
        _headers: Dict[str, str] = {}
        if self.base_url and self.api_key:
            _headers["Authorization"] = f"Bearer {self.api_key}"
        if self.default_headers:
            _headers.update(self.default_headers)
        if _headers:
            _client_params["default_headers"] = _headers
        if self.timeout is not None:
            _client_params["timeout"] = self.timeout
        if self.client_params:
            _client_params.update(self.client_params)
        self.client = AnthropicClient(**_client_params)
        self._client_loop = _loop
        return self.client

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """
        Generate keyword arguments for API requests.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        _request_params: Dict[str, Any] = {}
        # Resolve the output cap: positive user-passed value, else per-model
        # ceiling, optionally clamped to context_window - 1 for small endpoints.
        _request_params["max_tokens"] = resolve_anthropic_messages_max_tokens(
            self.max_tokens,
            self.id,
            context_length=self.context_window,
        )
        # Use `is not None` for numeric sampling params so legitimate falsy
        # values (temperature=0 deterministic, top_p=0/top_k=0) reach the API
        # instead of being silently dropped.
        if self.temperature is not None:
            _request_params["temperature"] = self.temperature
        if self.stop_sequences:
            _request_params["stop_sequences"] = self.stop_sequences
        if self.top_p is not None:
            _request_params["top_p"] = self.top_p
        if self.top_k is not None:
            _request_params["top_k"] = self.top_k
        # Thinking configuration. Priority: an explicit ``thinking`` dict wins;
        # otherwise ``reasoning_effort`` enables adaptive thinking.
        if self.thinking:
            _request_params["thinking"] = self.thinking
        elif self.reasoning_effort:
            # Adaptive thinking (opus-4-7+/sonnet-4-6+): thinking.type=adaptive
            # plus output_config.effort. output_config is not a native SDK param,
            # so it rides along in extra_body (the SDK forwards it verbatim).
            # Adaptive thinking requires temperature=1 — enforce it here so a
            # stale temperature can't 400 the request.
            _request_params["thinking"] = {"type": "adaptive"}
            _extra = _request_params.get("extra_body") or {}
            _extra["output_config"] = {"effort": self.reasoning_effort}
            _request_params["extra_body"] = _extra
            _request_params["temperature"] = 1
        if self.request_params:
            _request_params.update(self.request_params)
        return _request_params

    def describe_thinking_mode(self) -> str:
        """Describe Anthropic extended-thinking configuration.

        ``self.thinking`` is shaped like ``{"type": "enabled", "budget_tokens": N}``
        (fixed budget) or ``{"type": "adaptive"}`` (paired with reasoning_effort).
        """
        if isinstance(self.thinking, dict):
            t = self.thinking.get("type")
            if t == "adaptive":
                return f"adaptive(effort={self.reasoning_effort})" if self.reasoning_effort else "adaptive"
            budget = self.thinking.get("budget_tokens")
            if t == "enabled":
                return f"on(budget={budget})" if budget else "on"
        if self.reasoning_effort:
            return f"adaptive(effort={self.reasoning_effort})"
        return "off"

    async def format_messages(self, messages: List[Message]) -> Tuple[List[Dict[str, str]], str]:
        """
        Process the list of messages and separate them into API messages and system messages.

        Args:
            messages (List[Message]): The list of messages to process.

        Returns:
            Tuple[List[Dict[str, str]], str]: A tuple containing the list of API messages and the concatenated system messages.
        """
        chat_messages: List[Dict[str, str]] = []
        system_messages: List[str] = []

        for idx, message in enumerate(messages):
            content = message.content or ""
            if message.role == "system" or (
                message.role != "user"
                and message.role != "assistant"
                and idx in [0, 1]
                and not getattr(message, "tool_call_id", None)
            ):
                system_messages.append(content)  # type: ignore
            elif message.role == "tool":
                # OpenAI-style tool-result message (role="tool"). Anthropic only
                # allows "user"/"assistant", and this class already emits tool
                # results as a "user" message carrying tool_result blocks (see
                # format_function_call_results). The shared Runner may leave a
                # duplicate role="tool" copy in history; skip it so we don't send
                # an invalid role or a duplicate tool_result for the same id.
                continue
            else:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                if message.role == "user" and message.images is not None:
                    for image in message.images:
                        image_content = await self.add_image(image)
                        if image_content:
                            content.append(image_content)

                chat_messages.append({"role": message.role, "content": content})  # type: ignore

        # system_and_3 strategy: cache the system prompt (breakpoint added in
        # prepare_request_kwargs) + the last 3 messages for a rolling cache
        # window. Anthropic caps the WHOLE request at 4 cache_control blocks.
        #
        # Two subtleties this must handle to avoid a 400 "max 4 blocks" error:
        #   1. ``messages`` is a mutable list reused across turns; a message may
        #      already carry a cache_control block from a previous request's
        #      formatting. Re-adding one on the same message would double-count.
        #   2. A single message's content list may already end with a block that
        #      has cache_control (e.g. a tool_result). We must not stack a second.
        # So we first STRIP every existing cache_control block, then re-apply at
        # most 3 (system takes the 4th), newest-first.
        if self.enable_cache_control and chat_messages:
            self._strip_cache_control(chat_messages)
            applied = 0
            # Newest messages first so the freshest context stays cached when we
            # hit the budget. Budget is 3 here; the system prompt takes the 4th.
            for msg in reversed(chat_messages):
                if applied >= 3:
                    break
                content = msg.get("content")
                if isinstance(content, list) and content:
                    last_block = content[-1]
                    if isinstance(last_block, dict):
                        last_block["cache_control"] = {"type": "ephemeral"}
                        applied += 1
                elif isinstance(content, str) and content:
                    msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                    applied += 1

        return chat_messages, " ".join(system_messages)

    @staticmethod
    def _strip_cache_control(chat_messages: List[Dict[str, Any]]) -> None:
        """Remove any existing cache_control blocks from message content.

        ``messages`` is reused across turns, so a message formatted in a prior
        request may already carry a cache_control breakpoint. Stripping first
        guarantees we re-apply a clean, bounded set (Anthropic caps the request
        at 4 total) instead of accumulating one per turn until the API 400s.
        """
        for msg in chat_messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        block.pop("cache_control", None)

    async def add_image(self, image: Union[str, bytes]) -> Optional[Dict[str, Any]]:
        """
        Add an image to a message by converting it to base64 encoded format.

        Args:
            image: URL string, local file path, or bytes object

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing the processed image information if successful
        """
        import base64
        import imghdr

        type_mapping = {"jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}

        try:
            content = None
            # Case 1: Image is a string
            if isinstance(image, str):
                # Case 1.1: Image is a URL
                if image.startswith(("http://", "https://")):
                    import httpx

                    async with httpx.AsyncClient() as client:
                        resp = await client.get(image)
                        content = resp.content
                # Case 1.2: Image is a local file path
                else:
                    from pathlib import Path

                    path = Path(image)
                    if path.exists() and path.is_file():
                        loop = asyncio.get_running_loop()
                        content = await loop.run_in_executor(None, path.read_bytes)
                    else:
                        logger.error(f"Image file not found: {image}")
                        return None
            # Case 2: Image is a bytes object
            elif isinstance(image, bytes):
                content = image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None

            img_type = imghdr.what(None, h=content)
            if not img_type:
                logger.error("Unable to determine image type")
                return None

            media_type = type_mapping.get(img_type)
            if not media_type:
                logger.error(f"Unsupported image type: {img_type}")
                return None

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.b64encode(content).decode("utf-8"),
                },
            }

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def prepare_request_kwargs(self, system_message: str) -> Dict[str, Any]:
        """
        Prepare the request keyword arguments for the API call.

        Args:
            system_message (str): The concatenated system messages.

        Returns:
            Dict[str, Any]: The request keyword arguments.
        """
        request_kwargs = self.request_kwargs.copy()

        # Prompt caching: wrap system message as content block with cache_control.
        # Anthropic API accepts system as str OR List[ContentBlock].
        # The cache breakpoint on the system block keeps the static prefix cached
        # across multi-turn conversations (cache read = 0.1x input cost).
        if self.enable_cache_control and system_message:
            request_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_message,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            request_kwargs["system"] = system_message

        # Structured output via tool_use: inject synthetic tool and force tool_choice
        structured_tool = self._get_structured_output_tool()
        if structured_tool is not None:
            tools = request_kwargs.get("tools") or []
            tools = list(tools) + [structured_tool]
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = {"type": "tool", "name": "structured_output"}
        elif self.tools:
            request_kwargs["tools"] = self.get_tools()

        return request_kwargs

    def get_tools(self) -> Optional[List[Dict[str, Any]]]:
        """
        Transforms function definitions into a format accepted by the Anthropic API.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of tools formatted for the API, or None if no functions are defined.
        """
        if not self.functions:
            return None

        tools: List[Dict[str, Any]] = []
        for func_name, func_def in self.functions.items():
            parameters: Dict[str, Any] = func_def.parameters or {}
            properties: Dict[str, Any] = parameters.get("properties", {})
            required_params: List[str] = []

            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "")
                param_type_list: List[str] = [param_type] if isinstance(param_type, str) else param_type or []

                if "null" not in param_type_list:
                    required_params.append(param_name)

            input_properties: Dict[str, Dict[str, Union[str, List[str]]]] = {
                param_name: {
                    "type": param_info.get("type", ""),
                    "description": param_info.get("description", ""),
                }
                for param_name, param_info in properties.items()
            }

            tool = {
                "name": func_name,
                "description": func_def.description or "",
                "input_schema": {
                    "type": parameters.get("type", "object"),
                    "properties": input_properties,
                    "required": required_params,
                },
            }
            tools.append(tool)
        return tools

    def _maybe_recover_max_tokens(self, error: Exception, request_kwargs: Dict[str, Any]) -> bool:
        """Detect "max_tokens too large given prompt" and adjust in-place.

        Returns True when the caller should retry the request with the
        mutated ``request_kwargs``. The new cap is the API-reported
        ``available_tokens`` minus a small safety margin. Returns False for
        any other error (caller re-raises).
        """
        available = parse_available_output_tokens_from_error(str(error))
        if available is None:
            return False
        safe_out = max(1, available - _MAX_TOKENS_RETRY_SAFETY_MARGIN)
        old_cap = request_kwargs.get("max_tokens")
        request_kwargs["max_tokens"] = safe_out
        logger.warning(
            f"Anthropic max_tokens={old_cap} too large for current prompt; "
            f"retrying with max_tokens={safe_out:,} "
            f"(available_tokens={available:,})."
        )
        return True

    @override
    async def invoke(self, messages: List[Message]) -> AnthropicMessage:
        """
        Send a request to the Anthropic API to generate a response.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            AnthropicMessage: The response from the model.
        """
        chat_messages, system_message = await self.format_messages(messages)
        request_kwargs = self.prepare_request_kwargs(system_message)

        try:
            return await self.get_client().messages.create(
                model=self.id,
                messages=chat_messages,
                **request_kwargs,
            )
        except Exception as e:
            self._learn_context_limit_from_error(str(e))
            if self._maybe_recover_max_tokens(e, request_kwargs):
                return await self.get_client().messages.create(
                    model=self.id,
                    messages=chat_messages,
                    **request_kwargs,
                )
            raise

    @override
    async def invoke_stream(self, messages: List[Message]) -> Any:
        """
        Return a MessageStreamManager for the Anthropic streaming API.

        NOTE: Unlike ``messages.create``, ``messages.stream`` is a synchronous
        constructor that returns an async context manager. The real API call
        (and thus any error) happens inside ``async with response as stream``
        in :py:meth:`response_stream`. A try/except here cannot catch API
        errors — recovery (``_learn_context_limit_from_error`` +
        ``_maybe_recover_max_tokens``) lives in ``response_stream``.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Any: An ``AsyncMessageStreamManager`` (anthropic SDK).
        """
        chat_messages, system_message = await self.format_messages(messages)
        request_kwargs = self.prepare_request_kwargs(system_message)
        return self.get_client().messages.stream(
            model=self.id,
            messages=chat_messages,
            **request_kwargs,
        )

    async def _open_stream_with_recovery(self, messages: List[Message]) -> Tuple[Any, Any]:
        """Open an Anthropic stream, retrying once on max_tokens-too-large.

        Returns ``(stream_mgr, stream)`` where ``stream_mgr`` is the context
        manager (caller must ``await stream_mgr.__aexit__(...)``) and
        ``stream`` is the entered stream iterator.

        Mirrors :py:meth:`invoke`'s retry path: API errors at ``__aenter__``
        are first run through ``_learn_context_limit_from_error`` (for
        "prompt too long"), then through ``_maybe_recover_max_tokens`` (for
        "max_tokens too large given prompt"). On a successful match, the
        stream is reopened once with the adjusted output cap.
        """
        import asyncio as _asyncio
        import random as _random
        from agentica.model.stream_retry import default_is_parser_error

        chat_messages, system_message = await self.format_messages(messages)
        request_kwargs = self.prepare_request_kwargs(system_message)

        def _open():
            return self.get_client().messages.stream(
                model=self.id,
                messages=chat_messages,
                **request_kwargs,
            )

        async def _try_enter() -> Tuple[Any, Any]:
            mgr = _open()
            stream = await mgr.__aenter__()
            return mgr, stream

        # Retry the open step on transient parser / gateway / 5xx errors.
        # max_tokens recovery is layered on top so both paths compose.
        max_retries = 2
        attempt = 0
        while True:
            try:
                return await _try_enter()
            except Exception as e:
                self._learn_context_limit_from_error(str(e))
                if self._maybe_recover_max_tokens(e, request_kwargs):
                    # max_tokens-too-large: reopen once with adjusted cap;
                    # do not consume retry budget for transient bucket.
                    return await _try_enter()
                if attempt < max_retries and default_is_parser_error(
                    e,
                    extra_substrings=self.extra_retryable_substrings,
                ):
                    wait = 0.5 * (2**attempt) + _random.uniform(0.0, 0.25)
                    logger.warning(
                        "[stream-retry] anthropic/%s open attempt %d/%d failed, retrying in %.2fs: %s",
                        self.id,
                        attempt + 1,
                        max_retries + 1,
                        wait,
                        e,
                    )
                    await _asyncio.sleep(wait)
                    attempt += 1
                    continue
                raise

    def update_usage_metrics(
        self,
        assistant_message: Message,
        usage: Optional[Usage] = None,
        metrics: Metrics = Metrics(),
    ) -> None:
        """
        Update the usage metrics for the assistant message.

        Args:
            assistant_message (Message): The assistant message.
            usage (Optional[Usage]): The usage metrics returned by the model.
            metrics (Metrics): The metrics to update.
        """
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)
        if usage:
            metrics.input_tokens = usage.input_tokens or 0
            metrics.output_tokens = usage.output_tokens or 0
            metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

            if metrics.input_tokens is not None:
                assistant_message.metrics["input_tokens"] = metrics.input_tokens
                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + metrics.input_tokens
            if metrics.output_tokens is not None:
                assistant_message.metrics["output_tokens"] = metrics.output_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + metrics.output_tokens
            if metrics.total_tokens is not None:
                assistant_message.metrics["total_tokens"] = metrics.total_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + metrics.total_tokens
            if metrics.time_to_first_token is not None:
                assistant_message.metrics["time_to_first_token"] = metrics.time_to_first_token
                self.metrics.setdefault("time_to_first_token", []).append(metrics.time_to_first_token)

            # Build structured RequestUsage entry
            from agentica.model.usage import RequestUsage, TokenDetails

            entry = RequestUsage(
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                total_tokens=metrics.total_tokens,
                response_time=metrics.response_timer.elapsed,
            )
            # Anthropic cache_creation_input_tokens / cache_read_input_tokens
            cache_read = usage.cache_read_input_tokens or 0
            cache_write = usage.cache_creation_input_tokens or 0
            if cache_read:
                entry.input_tokens_details = TokenDetails(cached_tokens=cache_read)
            self.usage.add(entry)

            # Cost tracking: pass both cache_read and cache_write for accurate cost calculation
            if self._cost_tracker is not None:
                self._cost_tracker.record(
                    model_id=self.id,
                    input_tokens=metrics.input_tokens,
                    output_tokens=metrics.output_tokens,
                    cache_read_tokens=cache_read,
                    cache_write_tokens=cache_write,
                )

    def create_assistant_message(self, response: AnthropicMessage, metrics: Metrics) -> Tuple[Message, str, List[str]]:
        """
        Create an assistant message from the response.

        Args:
            response (AnthropicMessage): The response from the model.
            metrics (Metrics): The metrics for the response.

        Returns:
            Tuple[Message, str, List[str]]: A tuple containing the assistant message, the response content, and the tool ids.
        """
        message_data = MessageData()

        if response.content:
            message_data.response_block = response.content
            message_data.response_usage = response.usage

            # Find the first non-thinking content block
            for block in response.content:
                if isinstance(block, ThinkingBlock):
                    message_data.response_reasoning_content += block.thinking
                elif isinstance(block, RedactedThinkingBlock):
                    pass
                elif isinstance(block, TextBlock):
                    if not message_data.response_content:
                        message_data.response_content = block.text
                    else:
                        message_data.response_content += block.text
                elif isinstance(block, ToolUseBlock):
                    if not message_data.response_content:
                        tool_block_input = block.input
                        if tool_block_input and isinstance(tool_block_input, dict):
                            message_data.response_content = tool_block_input.get("query", "")

        # Create assistant message
        assistant_message = Message(
            role=response.role or "assistant",
            content=message_data.response_content,
        )

        # Set reasoning_content from thinking blocks
        if message_data.response_reasoning_content:
            assistant_message.reasoning_content = message_data.response_reasoning_content

        # Extract tool calls from the response
        if response.stop_reason == "tool_use":
            for block in message_data.response_block:
                if isinstance(block, ToolUseBlock):
                    tool_use: ToolUseBlock = block
                    tool_name = tool_use.name
                    tool_input = tool_use.input
                    message_data.tool_ids.append(tool_use.id)

                    function_def = {"name": tool_name}
                    if tool_input:
                        function_def["arguments"] = json.dumps(tool_input, ensure_ascii=False)
                    message_data.tool_calls.append(
                        {
                            "type": "function",
                            "function": function_def,
                            "id": tool_use.id,
                        }
                    )

        # Update assistant message if tool calls are present
        if len(message_data.tool_calls) > 0:
            assistant_message.tool_calls = message_data.tool_calls
            # Store content blocks as plain dicts rather than raw anthropic SDK
            # objects: the SDK models carry internal fields (e.g. __json_buf)
            # that trip pydantic serializer warnings on later model_dump()
            # (session persistence / logging), and the API accepts dicts too.
            assistant_message.content = [
                b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in message_data.response_block
            ]

        # Update usage metrics
        self.update_usage_metrics(assistant_message, message_data.response_usage, metrics)

        return assistant_message, message_data.response_content, message_data.tool_ids

    def get_function_calls_to_run(self, assistant_message: Message, messages: List[Message]) -> List[FunctionCall]:
        """
        Prepare function calls for the assistant message.

        Args:
            assistant_message (Message): The assistant message.
            messages (List[Message]): The list of conversation messages.

        Returns:
            List[FunctionCall]: A list of function calls to run.
        """
        function_calls_to_run: List[FunctionCall] = []
        if assistant_message.tool_calls is not None:
            for tool_call in assistant_message.tool_calls:
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(Message(role="user", content="Could not find function to call."))
                    continue
                if _function_call.error is not None:
                    messages.append(Message(role="user", content=_function_call.error))
                    continue
                function_calls_to_run.append(_function_call)
        return function_calls_to_run

    def format_function_call_results(
        self, function_call_results: List[Message], tool_ids: List[str], messages: List[Message]
    ) -> None:
        """
        Handle the results of function calls.

        Args:
            function_call_results (List[Message]): The results of the function calls.
            tool_ids (List[str]): The tool ids.
            messages (List[Message]): The list of conversation messages.
        """
        if len(function_call_results) > 0:
            fc_responses: List = []
            for _fc_message_index, _fc_message in enumerate(function_call_results):
                fc_responses.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_ids[_fc_message_index],
                        "content": _fc_message.content,
                    }
                )
            messages.append(Message(role="user", content=fc_responses))

    def parse_tool_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        tool_role: str = "tool",
    ) -> tuple:
        """Parse tool calls for Anthropic format.

        Anthropic tool_calls include 'id' (tool_use_id) which is needed for result formatting.
        Errors are appended as role="user" messages (Anthropic convention).
        """
        function_calls_to_run: List[FunctionCall] = []
        tool_ids: List[str] = []
        if assistant_message.tool_calls is None or len(assistant_message.tool_calls) == 0:
            return function_calls_to_run, {"tool_ids": tool_ids}

        for tool_call in assistant_message.tool_calls:
            _tool_use_id = tool_call.get("id", "")
            tool_ids.append(_tool_use_id)
            _function_call = get_function_call_for_tool_call(tool_call, self.functions)
            if _function_call is None:
                messages.append(Message(role="user", content="Could not find function to call."))
                continue
            if _function_call.error is not None:
                messages.append(Message(role="user", content=_function_call.error))
                continue
            function_calls_to_run.append(_function_call)

        return function_calls_to_run, {"tool_ids": tool_ids}

    def format_tool_results(
        self,
        function_call_results: List[Message],
        messages: List[Message],
        provider_metadata: dict,
    ) -> None:
        """Format tool results for Anthropic (role='user' with tool_result content blocks)."""
        tool_ids = provider_metadata.get("tool_ids", [])
        self.format_function_call_results(function_call_results, tool_ids, messages)

    async def handle_tool_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        model_response: ModelResponse,
        response_content: str,
        tool_ids: List[str],
    ) -> Optional[ModelResponse]:
        """
        Handle tool calls in the assistant message.

        Args:
            assistant_message (Message): The assistant message.
            messages (List[Message]): A list of messages.
            model_response [ModelResponse]: The model response.
            response_content (str): The response content.
            tool_ids (List[str]): The tool ids.

        Returns:
            Optional[ModelResponse]: The model response.
        """
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            model_response.content = str(response_content)
            model_response.content += "\n\n"
            function_calls_to_run = self.get_function_calls_to_run(assistant_message, messages)
            function_call_results: List[Message] = []

            async for _ in self.run_function_calls(
                function_calls=function_calls_to_run,
                function_call_results=function_call_results,
            ):
                pass

            self.format_function_call_results(function_call_results, tool_ids, messages)

            return model_response
        return None

    @override
    async def response(self, messages: List[Message]) -> ModelResponse:
        """
        Send a chat completion request to the Anthropic API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            ModelResponse: The response from the model.
        """
        self.sanitize_messages(messages)
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()

        metrics.response_timer.start()
        response: AnthropicMessage = await self.invoke(messages=messages)
        metrics.response_timer.stop()

        # Defensive: Anthropic should always return content, but guard against edge cases
        if not response.content:
            raise ValueError(
                f"Anthropic API returned empty content for model '{self.id}'. "
                "This may indicate a content filter or a transient API error."
            )

        # -*- Create assistant message
        assistant_message, response_content, tool_ids = self.create_assistant_message(
            response=response, metrics=metrics
        )

        # -*- Extract structured output from tool_use block if response_format is set
        if (
            self.response_format is not None
            and self.use_structured_outputs
            and isinstance(self.response_format, type)
            and issubclass(self.response_format, BaseModel)
            and response.stop_reason == "tool_use"
        ):
            try:
                for block in response.content:
                    if isinstance(block, ToolUseBlock) and block.name == "structured_output":
                        parsed_object = self.response_format.model_validate(block.input)
                        model_response.parsed = parsed_object
                        # Use the parsed JSON as content
                        model_response.content = parsed_object.model_dump_json()
                        break
            except Exception as e:
                logger.warning(f"Error parsing structured output from Claude tool_use: {e}")

            # Don't treat structured_output tool_use as a real tool call
            # Add assistant message and return
            messages.append(assistant_message)
            assistant_message.log()
            metrics.log()
            if model_response.content is None and assistant_message.content is not None:
                model_response.content = assistant_message.get_content_string()
            return model_response

        # -*- Add assistant message to messages
        messages.append(assistant_message)

        # -*- Log response and metrics
        assistant_message.log()
        metrics.log()

        # -*- Handle tool calls
        # Expose stop_reason so Runner's agentic loop can detect
        # max_tokens truncation (stop_reason == "max_tokens") for recovery.
        model_response.finish_reason = "length" if response.stop_reason == "max_tokens" else response.stop_reason
        self.last_finish_reason = model_response.finish_reason
        assistant_message.finish_reason = model_response.finish_reason

        if await self.handle_tool_calls(assistant_message, messages, model_response, response_content, tool_ids):
            return model_response

        # -*- Update model response
        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.reasoning_content:
            model_response.reasoning_content = assistant_message.reasoning_content

        return model_response

    async def handle_stream_tool_calls(
        self,
        assistant_message: Message,
        messages: List[Message],
        tool_ids: List[str],
    ) -> AsyncIterator[ModelResponse]:
        """
        Parse and run function calls from the assistant message.

        Args:
            assistant_message (Message): The assistant message containing tool calls.
            messages (List[Message]): The list of conversation messages.
            tool_ids (List[str]): The list of tool IDs.

        Yields:
            AsyncIterator[ModelResponse]: Yields model responses during function execution.
        """
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            yield ModelResponse(content="\n\n")
            function_calls_to_run = self.get_function_calls_to_run(assistant_message, messages)
            function_call_results: List[Message] = []

            async for intermediate_model_response in self.run_function_calls(
                function_calls=function_calls_to_run, function_call_results=function_call_results
            ):
                yield intermediate_model_response

            self.format_function_call_results(function_call_results, tool_ids, messages)

    @override
    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        self.sanitize_messages(messages)
        self._log_messages(messages)
        message_data = MessageData()
        metrics = Metrics()

        # Generate response. Open the stream via _open_stream_with_recovery
        # so "max_tokens too large given prompt" at __aenter__ triggers one
        # retry with a reduced cap; see the helper's docstring.
        metrics.response_timer.start()
        stream_mgr, stream = await self._open_stream_with_recovery(messages)
        try:
            async for delta in stream:
                if isinstance(delta, RawContentBlockDeltaEvent):
                    if isinstance(delta.delta, TextDelta):
                        yield ModelResponse(content=delta.delta.text)
                        message_data.response_content += delta.delta.text
                        metrics.output_tokens += 1
                        if metrics.output_tokens == 1:
                            metrics.time_to_first_token = metrics.response_timer.elapsed
                    elif isinstance(delta.delta, ThinkingDelta):
                        yield ModelResponse(reasoning_content=delta.delta.thinking)
                        message_data.response_reasoning_content += delta.delta.thinking
                    elif isinstance(delta.delta, SignatureDelta):
                        pass

                if isinstance(delta, _CONTENT_BLOCK_STOP_TYPES):
                    if isinstance(delta.content_block, ToolUseBlock):
                        tool_use = delta.content_block
                        tool_name = tool_use.name
                        tool_input = tool_use.input
                        message_data.tool_ids.append(tool_use.id)

                        function_def = {"name": tool_name}
                        if tool_input:
                            function_def["arguments"] = json.dumps(tool_input, ensure_ascii=False)
                        message_data.tool_calls.append(
                            {
                                "type": "function",
                                "function": function_def,
                                "id": tool_use.id,
                            }
                        )
                    # Store the assistant content block as a plain dict, not the
                    # raw anthropic SDK object. The SDK's Parsed*/ToolUseBlock
                    # models carry internal fields (e.g. __json_buf) that trip
                    # pydantic serializer warnings when this Message is later
                    # model_dump()'d for session persistence / logging. A dict is
                    # also what the Anthropic API accepts on the next request.
                    _block = delta.content_block
                    if hasattr(_block, "model_dump"):
                        message_data.response_block.append(_block.model_dump(exclude_none=True))
                    else:
                        message_data.response_block.append(_block)

                if isinstance(delta, _MESSAGE_STOP_TYPES):
                    message_data.response_usage = delta.message.usage
                    # Capture stop_reason: "max_tokens" → "length" for consistency with OpenAI
                    _stop = delta.message.stop_reason
                    self.last_finish_reason = "length" if _stop == "max_tokens" else _stop
        finally:
            await stream_mgr.__aexit__(None, None, None)
        yield ModelResponse(content="\n\n")

        metrics.response_timer.stop()

        # Create assistant message
        assistant_message = Message(
            role="assistant",
            content=message_data.response_content,
        )
        if message_data.response_reasoning_content:
            assistant_message.reasoning_content = message_data.response_reasoning_content

        # Update assistant message if tool calls are present
        if len(message_data.tool_calls) > 0:
            assistant_message.content = message_data.response_block
            assistant_message.tool_calls = message_data.tool_calls

        # Update usage metrics
        self.update_usage_metrics(assistant_message, message_data.response_usage, metrics)
        assistant_message.finish_reason = self.last_finish_reason

        # Add assistant message to messages
        messages.append(assistant_message)

        # Log response and metrics
        assistant_message.log()
        metrics.log()

        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            async for _resp in self.handle_stream_tool_calls(assistant_message, messages, message_data.tool_ids):
                yield _resp

    def get_tool_call_prompt(self) -> Optional[str]:
        if self.functions is not None and len(self.functions) > 0:
            tool_call_prompt = "Do not reflect on the quality of the returned search results in your response"
            return tool_call_prompt
        return None

    def get_system_message_for_model(self) -> Optional[str]:
        return self.get_tool_call_prompt()
