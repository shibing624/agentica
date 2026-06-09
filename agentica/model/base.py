# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import asyncio
import collections.abc
import contextvars
import io
import json
import base64
import os
import re
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import GeneratorType
from typing import List, Iterator, AsyncIterator, Optional, Dict, Any, Callable, Union, Sequence

from agentica.run_response import AgentCancelledError
from agentica.utils.log import logger
from agentica.model.message import Message
from agentica.model.metrics import Metrics
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.model.usage import Usage, RequestUsage, TokenDetails
from agentica.security.redact import redact_sensitive_text
from agentica.tools.base import ModelTool, Tool, Function, FunctionCall, ToolCallException, get_function_call_for_tool_call
from agentica.utils.timer import Timer
from agentica.cost_tracker import CostTracker, get_model_context_window
from agentica.utils.langfuse_integration import langfuse_span_context, update_langfuse_span
from agentica.hooks import RunHooks, _CompositeRunHooks


def require_first_choice(response: Any, *, context: str) -> Any:
    """Return ``response.choices[0]`` or raise a clear ``ValueError``.

    Providers occasionally return ``choices=[]`` (rate limits, content
    filters, transient API errors). Subclasses MUST funnel every
    ``response.choices[0]`` access through this helper instead of indexing
    directly, so failures surface as a single, actionable exception.
    """
    choices = response.choices
    if not choices:
        raise ValueError(
            f"{context} returned empty choices. "
            "This may indicate a content filter, a quota issue, or a transient API error."
        )
    return choices[0]


_STREAM_REDACTION_MAX_BUFFER_CHARS = 4096
_STREAM_REDACTION_TAIL_CHARS = 512
_STREAM_PRIVATE_KEY_BEGIN_RE = re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----")
_STREAM_PRIVATE_KEY_END_RE = re.compile(r"-----END[A-Z ]*PRIVATE KEY-----")


@dataclass
class ModelRunState:
    """Task-local mutable state for one model run."""

    function_call_stack: List[FunctionCall] = field(default_factory=list)
    failed_call_counts: Dict[str, int] = field(default_factory=dict)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


_MODEL_RUN_STATE: contextvars.ContextVar[Optional[ModelRunState]] = contextvars.ContextVar(
    "agentica_model_run_state",
    default=None,
)


def _run_hook_method_overridden(hooks: Optional[RunHooks], method_name: str) -> bool:
    if hooks is None:
        return False
    if isinstance(hooks, _CompositeRunHooks):
        return any(
            getattr(type(hook), method_name) is not getattr(RunHooks, method_name)
            for hook in hooks._hooks_list
        )
    return getattr(type(hooks), method_name) is not getattr(RunHooks, method_name)


def _redact_unterminated_private_key_block(text: str) -> str:
    key_begin = _STREAM_PRIVATE_KEY_BEGIN_RE.search(text)
    if key_begin and not _STREAM_PRIVATE_KEY_END_RE.search(text, key_begin.end()):
        prefix = redact_sensitive_text(text[:key_begin.start()]) or ""
        return prefix + "***REDACTED_PRIVATE_KEY***"
    return redact_sensitive_text(text) or ""


def _take_redactable_stream_text(buffer: str, final: bool = False) -> tuple[str, str]:
    """Return text that can be redacted and streamed without splitting secrets."""
    if not buffer:
        return "", ""
    if final:
        return _redact_unterminated_private_key_block(buffer), ""

    key_begin = _STREAM_PRIVATE_KEY_BEGIN_RE.search(buffer)
    if key_begin and not _STREAM_PRIVATE_KEY_END_RE.search(buffer, key_begin.end()):
        prefix = buffer[:key_begin.start()]
        if prefix:
            return redact_sensitive_text(prefix) or "", buffer[key_begin.start():]
        return "", buffer

    newline_idx = max(buffer.rfind("\n"), buffer.rfind("\r"))
    if newline_idx >= 0:
        split_at = newline_idx + 1
        return redact_sensitive_text(buffer[:split_at]) or "", buffer[split_at:]

    if len(buffer) <= _STREAM_REDACTION_MAX_BUFFER_CHARS:
        return "", buffer

    split_limit = max(0, len(buffer) - _STREAM_REDACTION_TAIL_CHARS)
    split_at = -1
    for idx in range(split_limit - 1, -1, -1):
        if buffer[idx].isspace():
            split_at = idx + 1
            break
    if split_at <= 0:
        split_at = split_limit
    return redact_sensitive_text(buffer[:split_at]) or "", buffer[split_at:]


@dataclass
class Model(ABC):
    """Abstract base class for LLM models. Subclasses must implement invoke/invoke_stream/response/response_stream."""

    # ID of the model to use.
    id: str = "not-provided"
    # Name for this Model. This is not sent to the Model API.
    name: Optional[str] = None
    # Provider for this Model. This is not sent to the Model API.
    provider: Optional[str] = None
    # Metrics collected for this Model. This is not sent to the Model API.
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Structured usage tracking (cross-request aggregation).
    usage: Usage = field(default_factory=Usage)
    response_format: Optional[Any] = None

    # -*- Model capability limits (not sent to the API) -*-
    context_window: int = 128000

    # Extra retryable error substrings, merged on top of the SDK's default
    # protocol-level transients (see ``LoopState.RETRYABLE_SUBSTRINGS``).
    #
    # Use this for deployment-specific API-gateway / proxy markers that the
    # SDK cannot know about — e.g. a private corp gateway returning
    # ``venus_error 4001`` or ``aiproxy_busy`` as a 400-bodied transient.
    #
    # Resolution order: this field > env var ``AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS``
    # (comma-separated) > none. Matched case-insensitively via substring.
    extra_retryable_substrings: Optional[List[str]] = None

    # A list of tools provided to the Model.
    tools: Optional[List[Union[ModelTool, Dict]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_tools: bool = True
    tool_call_limit: Optional[int] = None
    max_concurrent_tools: int = 10

    # -*- Functions available to the Model to call -*-
    functions: Optional[Dict[str, Function]] = None
    # Legacy mirrors for direct run_function_calls() callers. Agent/Runner paths
    # keep these per-run values in ModelRunState instead of shared Model fields.
    function_call_stack: Optional[List[FunctionCall]] = None
    # Per-run tracker for repeated failed tool calls (soft notice, not blocking).
    # Kept on Model only as a legacy mirror for direct run_function_calls().
    _failed_call_counts: Optional[Dict[str, int]] = None

    # System prompt from the model added to the Agent.
    system_prompt: Optional[str] = None
    # Instructions from the model added to the Agent.
    instructions: Optional[List[str]] = None

    # Session ID of the calling Agent or Workflow.
    session_id: Optional[str] = None
    # User ID of the calling Agent.
    user_id: Optional[str] = None
    # Agent name for tracing.
    agent_name: Optional[str] = None
    # Whether to use the structured outputs with this Model.
    use_structured_outputs: Optional[bool] = None
    # Whether the Model supports structured outputs.
    supports_structured_outputs: bool = False

    # --- Private fields (not in __init__ signature, used internally) ---
    _agent_ref: Optional[weakref.ref] = field(init=False, repr=False, default=None)

    # Cost tracker (v3): accumulates USD cost across all invoke() calls in a run.
    # Reset to a fresh CostTracker at the start of each Agent.run().
    _cost_tracker: Optional[CostTracker] = field(init=False, repr=False, default=None)

    # Finish reason captured from the most recent response / response_stream() call.
    # Set by provider; consumed by Runner's agentic loop for max_tokens recovery.
    last_finish_reason: Optional[str] = field(init=False, repr=False, default=None)

    _DEFAULT_CONTEXT_WINDOW: int = 128000

    def __post_init__(self):
        # Auto-set provider if not provided
        if self.provider is None:
            self.provider = f"{self.name} ({self.id})" if self.name else self.id

        if self.context_window == self._DEFAULT_CONTEXT_WINDOW and self.id != "not-provided":
            catalog_cw = get_model_context_window(self.id, default=0)
            if catalog_cw > 0:
                self.context_window = catalog_cw

    @staticmethod
    def begin_run_state(tool_choice: Optional[Union[str, Dict[str, Any]]] = None) -> contextvars.Token:
        return _MODEL_RUN_STATE.set(ModelRunState(tool_choice=tool_choice))

    @staticmethod
    def reset_run_state(token: contextvars.Token) -> None:
        _MODEL_RUN_STATE.reset(token)

    @staticmethod
    def current_run_state() -> Optional[ModelRunState]:
        return _MODEL_RUN_STATE.get()

    def _get_model_run_state(self) -> ModelRunState:
        state = _MODEL_RUN_STATE.get()
        if state is None:
            state = ModelRunState(
                function_call_stack=self.function_call_stack if self.function_call_stack is not None else [],
                failed_call_counts=self._failed_call_counts if self._failed_call_counts is not None else {},
                tool_choice=self.tool_choice,
            )
            _MODEL_RUN_STATE.set(state)
        return state

    def _get_function_call_stack(self) -> List[FunctionCall]:
        return self._get_model_run_state().function_call_stack

    def _get_failed_call_counts(self) -> Dict[str, int]:
        return self._get_model_run_state().failed_call_counts

    def get_tool_choice(self) -> Optional[Union[str, Dict[str, Any]]]:
        state = _MODEL_RUN_STATE.get()
        if state is not None:
            return state.tool_choice
        return self.tool_choice

    def set_tool_choice(self, tool_choice: Optional[Union[str, Dict[str, Any]]]) -> None:
        state = _MODEL_RUN_STATE.get()
        if state is not None:
            state.tool_choice = tool_choice
            return
        self.tool_choice = tool_choice

    _CONTEXT_LIMIT_PATTERN = re.compile(
        r"(?:maximum context (?:length|window)|context_length|max_context_length)[^\d]*(\d[\d,]*)",
        re.IGNORECASE,
    )

    def get_retryable_substrings(self, defaults: tuple) -> tuple:
        """Merge SDK default retry markers with user-extended markers.

        Sources merged (de-duplicated, all lowercased, all matched as substrings):
          1. ``defaults`` — SDK protocol-level transients (passed in by caller,
             usually ``LoopState.RETRYABLE_SUBSTRINGS``).
          2. ``self.extra_retryable_substrings`` — per-model deployment hook.
          3. Env var ``AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS`` — comma-separated
             global override, useful for ops without touching code.

        Example: a private corp gateway named ``venus`` that returns
        ``Error code: 400 - {'type': 'venus_error', ...}`` for transient
        upstream hiccups::

            model = OpenAIChat(id="gpt-5", extra_retryable_substrings=["venus_error"])

        or, deployment-side without code changes::

            export AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS="venus_error,aiproxy_busy"
        """
        merged = {s.lower() for s in defaults}
        if self.extra_retryable_substrings:
            merged.update(s.lower() for s in self.extra_retryable_substrings if s)
        env_extra = os.environ.get("AGENTICA_EXTRA_RETRYABLE_SUBSTRINGS", "")
        if env_extra:
            merged.update(s.strip().lower() for s in env_extra.split(",") if s.strip())
        return tuple(merged)

    def _learn_context_limit_from_error(self, error_message: str) -> None:
        """Extract context window size from API error messages and update self.context_window."""
        match = self._CONTEXT_LIMIT_PATTERN.search(error_message)
        if match:
            limit = int(match.group(1).replace(",", ""))
            if limit > 1000 and limit != self.context_window:
                old = self.context_window
                self.context_window = limit
                logger.info(f"Learned context_window={limit} from API error (was {old}) for model {self.id}")

    @property
    @abstractmethod
    def request_kwargs(self) -> Dict[str, Any]:
        """Build the API request parameters dict. Subclasses must implement."""
        ...

    def describe_thinking_mode(self) -> str:
        """Return a short, human-readable description of this model's thinking/reasoning
        configuration for logging.

        Default is ``"off"``. Subclasses (OpenAIChat, Claude, ...) override this to
        introspect their provider-specific knobs (``reasoning_effort``,
        ``extra_body.thinking``, ``extra_body.enable_thinking``, Anthropic ``thinking``).
        """
        return "off"

    def to_dict(self) -> Dict[str, Any]:
        _dict = {"name": self.name, "id": self.id, "provider": self.provider, "metrics": self.metrics}
        if self.functions:
            _dict["functions"] = {k: v.to_dict() for k, v in self.functions.items()}
            _dict["tool_call_limit"] = self.tool_call_limit
        return _dict

    def __repr__(self) -> str:
        """Concise representation for logging."""
        tools_count = len(self.tools) if self.tools else 0
        # Show first 3 + *** + last 4 chars of api_key for readability
        api_key = getattr(self, 'api_key', None) or ""
        if api_key and len(api_key) >= 8:
            key_hint = f"{api_key[:3]}***{api_key[-4:]}"
        elif api_key and len(api_key) >= 4:
            key_hint = f"***{api_key[-4:]}"
        else:
            key_hint = ""
        # Show base_url
        base_url = getattr(self, 'base_url', None) or ""
        parts = [f"id={self.id!r}"]
        if base_url:
            parts.append(f"base_url={str(base_url)!r}")
        if key_hint:
            parts.append(f"api_key='{key_hint}'")
        parts.append(f"tools={tools_count}")
        return f"{self.name or self.__class__.__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.__repr__()

    # --- Async-only abstract methods (subclasses must implement) ---

    @abstractmethod
    async def invoke(self, messages: List[Message]) -> Any:
        """Invoke the LLM API, returns the raw SDK response."""
        ...

    @abstractmethod
    async def invoke_stream(self, messages: List[Message]) -> Any:
        """Stream-invoke the LLM API, yields raw SDK chunks."""
        ...

    @abstractmethod
    async def response(self, messages: List[Message]) -> ModelResponse:
        """Full response (including tool-call loop). Returns ModelResponse."""
        ...

    @abstractmethod
    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """Streaming response (including tool-call loop). Yields ModelResponse."""
        ...

    @staticmethod
    def sanitize_messages(messages: List[Message]) -> List[Message]:
        """Validate and fix tool call message sequences.

        OpenAI API requires that every assistant message with 'tool_calls' must be
        followed by tool messages responding to each 'tool_call_id'. If any tool
        response is missing (e.g. due to an interrupted execution or corrupted
        history), this method adds a placeholder tool response so the API call
        does not fail.

        The messages list is modified **in-place** and also returned.

        Args:
            messages: The list of messages to sanitize.

        Returns:
            The same list of messages after sanitization.
        """
        i = 0
        while i < len(messages):
            msg = messages[i]
            # Only process assistant messages that have tool_calls
            if msg.role == "assistant" and msg.tool_calls:
                expected_ids = {}
                for tc in msg.tool_calls:
                    tc_id = tc.get("id") if isinstance(tc, dict) else tc.id
                    if tc_id:
                        expected_ids[tc_id] = tc

                # Scan the following messages for matching tool responses.
                # We scan all messages until the next assistant message (or end),
                # because additional non-tool messages (e.g. from ToolCallException)
                # may be interleaved between tool responses.
                j = i + 1
                first_non_tool_pos = None
                while j < len(messages):
                    next_msg = messages[j]
                    if next_msg.role == "tool" and next_msg.tool_call_id in expected_ids:
                        del expected_ids[next_msg.tool_call_id]
                        j += 1
                    elif next_msg.role == "assistant":
                        # Reached the next assistant turn — stop scanning
                        break
                    else:
                        # Track first non-tool position for placeholder insertion
                        if first_non_tool_pos is None:
                            first_non_tool_pos = j
                        j += 1

                # Insert placeholder responses for any missing tool_call_ids.
                # Insert right after the assistant message + existing tool responses,
                # before any non-tool messages.
                if expected_ids:
                    insert_pos = first_non_tool_pos if first_non_tool_pos is not None else j
                    for tc_id, tc in expected_ids.items():
                        func_info = tc.get("function", {}) if isinstance(tc, dict) else {}
                        func_name = func_info.get("name", "unknown") if isinstance(func_info, dict) else "unknown"
                        logger.debug(
                            f"Missing tool response for tool_call_id={tc_id} "
                            f"(function={func_name}), inserting placeholder."
                        )
                        placeholder = Message(
                            role="tool",
                            tool_call_id=tc_id,
                            content=f"Error: tool call '{func_name}' did not return a response (execution may have been interrupted).",
                        )
                        messages.insert(insert_pos, placeholder)
                        insert_pos += 1
                    # Re-scan from current position since we inserted messages
                    continue
            i += 1
        return messages

    def _log_messages(self, messages: List[Message]) -> None:
        """
        Log messages for debugging.
        """
        for m in messages:
            m.log()

    def get_tools_for_api(self) -> Optional[List[Dict[str, Any]]]:
        if self.tools is None:
            return None

        tools_for_api = []
        for tool in self.tools:
            if isinstance(tool, ModelTool):
                tools_for_api.append(tool.to_dict())
            elif isinstance(tool, Dict):
                function_name = tool.get("function", {}).get("name")
                if function_name and self.functions is not None:
                    fn = self.functions.get(function_name)
                    if fn is not None and not fn.is_available():
                        continue
                tools_for_api.append(tool)
        return tools_for_api

    def add_tool(
            self, tool: Union[ModelTool, Tool, Callable, Dict, Function], strict: bool = False,
            agent: Optional[Any] = None
    ) -> None:
        if self.tools is None:
            self.tools = []

        # If the tool is a Tool or Dict, add it directly to the Model
        if isinstance(tool, ModelTool) or isinstance(tool, Dict):
            if tool not in self.tools:
                self.tools.append(tool)
                logger.debug(f"Added tool {tool} to model.")

        # If the tool is a Callable or Toolkit, process and add to the Model
        elif callable(tool) or isinstance(tool, Tool) or isinstance(tool, Function):
            if self.functions is None:
                self.functions = {}

            if isinstance(tool, Tool):
                # For each function in the toolkit, process entrypoint and add to self.tools
                for name, func in tool.functions.items():
                    # If the function does not exist in self.functions, add to self.tools
                    if name not in self.functions:
                        func._agent = agent
                        func.process_entrypoint(strict=strict)
                        if strict and self.supports_structured_outputs:
                            func.strict = True
                        self.functions[name] = func
                        # Deferred tools: register in functions (executable) but
                        # don't add schema to tools (invisible to LLM until discovered).
                        if not func.deferred and func.is_available():
                            self.tools.append({"type": "function", "function": func.to_dict()})
                        logger.debug(f"Function {name} from {tool.name} added to model.")

            elif isinstance(tool, Function):
                if tool.name not in self.functions:
                    tool._agent = agent
                    tool.process_entrypoint(strict=strict)
                    if strict and self.supports_structured_outputs:
                        tool.strict = True
                    self.functions[tool.name] = tool
                    if not tool.deferred and tool.is_available():
                        self.tools.append({"type": "function", "function": tool.to_dict()})
                    logger.debug(f"Function {tool.name} added to model.")

            elif callable(tool):
                try:
                    function_name = tool.__name__
                    if function_name not in self.functions:
                        func = Function.from_callable(tool, strict=strict)
                        func._agent = agent
                        if strict and self.supports_structured_outputs:
                            func.strict = True
                        self.functions[func.name] = func
                        if not func.deferred and func.is_available():
                            self.tools.append({"type": "function", "function": func.to_dict()})
                        logger.debug(f"Function {func.name} added to model.")
                except Exception as e:
                    logger.warning(f"Could not add function {tool}: {e}")

    def deactivate_function_calls(self) -> None:
        # Deactivate tool calls by setting future tool calls to "none"
        # This is triggered when the function call limit is reached.
        self.set_tool_choice("none")

    # ── Tool call parsing and result formatting (provider-overridable) ────────
    # Runner calls these to decouple tool execution from the Model layer.
    # Default implementations handle OpenAI-compatible format.
    # Providers with different protocols (Anthropic, Ollama) override.

    def parse_tool_calls(
        self, assistant_message: Message, messages: List[Message], tool_role: str = "tool",
    ) -> tuple:
        """Parse tool calls from assistant message into FunctionCall objects.

        Returns:
            (function_calls_to_run, provider_metadata)
            - function_calls_to_run: List[FunctionCall] ready for execution
            - provider_metadata: opaque dict for format_tool_results() (e.g. tool_ids for Anthropic)
        """
        function_calls_to_run: List[FunctionCall] = []
        if assistant_message.tool_calls is None or len(assistant_message.tool_calls) == 0:
            return function_calls_to_run, {}

        for tool_call in assistant_message.tool_calls:
            _tool_call_id = tool_call.get("id")
            _function_call = get_function_call_for_tool_call(tool_call, self.functions)
            if _function_call is None:
                messages.append(
                    Message(role=tool_role, tool_call_id=_tool_call_id, content="Could not find function to call.")
                )
                continue
            if _function_call.error is not None:
                messages.append(
                    Message(role=tool_role, tool_call_id=_tool_call_id, content=_function_call.error)
                )
                continue
            function_calls_to_run.append(_function_call)

        return function_calls_to_run, {"tool_role": tool_role}

    def format_tool_results(
        self, function_call_results: List[Message], messages: List[Message], provider_metadata: dict,
    ) -> None:
        """Append tool results to messages in provider-appropriate format.

        Default (OpenAI): extend messages directly (tool results already have role="tool").
        Anthropic overrides to wrap in role="user" with tool_result content blocks.
        """
        if function_call_results:
            messages.extend(function_call_results)

    async def run_function_calls(
            self, function_calls: List[FunctionCall], function_call_results: List[Message], tool_role: str = "tool"
    ) -> AsyncIterator[ModelResponse]:
        token = None
        if self.current_run_state() is None:
            token = self.begin_run_state(tool_choice=self.tool_choice)
        try:
            async for response in self._run_function_calls_impl(
                    function_calls=function_calls,
                    function_call_results=function_call_results,
                    tool_role=tool_role,
            ):
                yield response
        finally:
            if token is not None:
                state = self._get_model_run_state()
                self.function_call_stack = list(state.function_call_stack)
                self._failed_call_counts = dict(state.failed_call_counts)
                self.tool_choice = state.tool_choice
                self.reset_run_state(token)

    async def _run_function_calls_impl(
            self, function_calls: List[FunctionCall], function_call_results: List[Message], tool_role: str = "tool"
    ) -> AsyncIterator[ModelResponse]:
        """Execute tool calls with concurrency-split execution.

        Strategy (mirrors CC's StreamingToolExecutor):
        - concurrency_safe=True  tools run in parallel with each other.
        - concurrency_safe=False tools run sequentially, one at a time.
        - A *bash/execute* error aborts any remaining unsafe tools
          (sibling_error pattern from CC).

        Phase 1: Emit tool_call_started events (in order)
        Phase 2a: Execute safe tools in parallel (asyncio.gather)
        Phase 2b: Execute unsafe tools sequentially
        Phase 3: Process results in original order
        """
        function_call_stack = self._get_function_call_stack()
        failed_call_counts = self._get_failed_call_counts()

        # Phase 1: Emit started events for all function calls
        _agent = self._agent_ref() if self._agent_ref is not None else None
        for function_call in function_calls:

            # --- Lifecycle: tool start ---
            if (
                _agent is not None
                and hasattr(_agent, '_run_hooks')
                and _run_hook_method_overridden(_agent._run_hooks, "on_tool_start")
            ):
                tool_start_input = {
                    "tool_name": function_call.function.name,
                    "tool_call_id": function_call.call_id or "",
                    "tool_args": function_call.arguments,
                }
                with langfuse_span_context(
                    name="hook.run.on_tool_start",
                    input_data=tool_start_input,
                    metadata={
                        "hook": "run.on_tool_start",
                        "agent_id": _agent.agent_id,
                        "agent_name": _agent.name or "Agent",
                        "run_id": _agent.run_id,
                    },
                ) as span:
                    await _agent._run_hooks.on_tool_start(
                        agent=_agent,
                        tool_name=function_call.function.name,
                        tool_call_id=function_call.call_id or "",
                        tool_args=function_call.arguments,
                    )
                    update_langfuse_span(span, output={"status": "completed"})

            yield ModelResponse(
                content=function_call.get_call_str(),
                tool_call={
                    "role": tool_role,
                    "tool_call_id": function_call.call_id,
                    "tool_name": function_call.function.name,
                    "tool_args": function_call.arguments,
                },
                event=ModelResponseEvent.tool_call_started.value,
            )

        # Phase 2: Concurrency-split execution
        # -----------------------------------------------------------------
        # Split into safe (read-only, parallel-ok) vs unsafe (write/shell, serial).
        # Maintains original ordering so Phase 3 can index by position.
        # -----------------------------------------------------------------
        _SHELL_TOOL_NAMES = {"execute", "bash", "shell", "run_command"}
        timers = [Timer() for _ in function_calls]
        exceptions: List[Optional[BaseException]] = [None] * len(function_calls)
        results: List[bool] = [False] * len(function_calls)

        safe_indices   = [i for i, fc in enumerate(function_calls) if fc.function.concurrency_safe]
        unsafe_indices = [i for i, fc in enumerate(function_calls) if not fc.function.concurrency_safe]

        if safe_indices or unsafe_indices:
            logger.debug(
                f"[tool-exec] batch: {len(function_calls)} call(s), "
                f"safe={len(safe_indices)} (parallel), "
                f"unsafe={len(unsafe_indices)} (serial)"
            )

        # Default timeout for tool execution (seconds)
        _DEFAULT_TOOL_TIMEOUT = 120

        # Guardrail facade (lazy import, runs once per call)
        if _agent is not None and (_agent.tool_input_guardrails or _agent.tool_output_guardrails):
            from agentica.guardrails.tool import check_input_guardrails, check_output_guardrails
            _has_guardrails = True
        else:
            _has_guardrails = False

        # Phase 2a: run safe tools in parallel
        async def _execute_safe(idx: int, fc: FunctionCall) -> None:
            # Input guardrail check
            if _has_guardrails:
                _fc_args = json.dumps(fc.arguments) if fc.arguments else None
                reject_msg = await check_input_guardrails(
                    _agent, fc.function.name, _fc_args, fc.call_id,
                )
                if reject_msg is not None:
                    fc.result = reject_msg
                    fc.error = reject_msg
                    results[idx] = False
                    timers[idx].start()
                    timers[idx].stop()
                    return
            timers[idx].start()
            try:
                if fc.function.manages_own_timeout:
                    results[idx] = await fc.execute()
                else:
                    _timeout = fc.function.timeout or _DEFAULT_TOOL_TIMEOUT
                    results[idx] = await asyncio.wait_for(fc.execute(), timeout=_timeout)
                # Output guardrail check
                if _has_guardrails:
                    _fc_args = json.dumps(fc.arguments) if fc.arguments else None
                    reject_msg = await check_output_guardrails(
                        _agent, fc.function.name, _fc_args, fc.call_id, fc.result,
                    )
                    if reject_msg is not None:
                        fc.result = reject_msg
                        fc.error = reject_msg
                        results[idx] = False
            except asyncio.TimeoutError:
                _timeout = fc.function.timeout or _DEFAULT_TOOL_TIMEOUT
                exceptions[idx] = TimeoutError(
                    f"Tool '{fc.function.name}' timed out after {_timeout}s"
                )
                results[idx] = False
            except ToolCallException as tce:
                exceptions[idx] = tce
                results[idx] = False
            except AgentCancelledError:
                # Hard cancellation must propagate (don't treat as tool failure).
                raise
            except Exception as exc:
                exceptions[idx] = exc
                results[idx] = False
            finally:
                timers[idx].stop()

        if safe_indices:
            gather_results = await asyncio.gather(
                *[_execute_safe(i, function_calls[i]) for i in safe_indices],
                return_exceptions=True,
            )
            # Re-raise CancelledError / AgentCancelledError immediately so
            # user-initiated cancel propagates out of the runner instead of
            # being captured as a tool failure result.
            for r in gather_results:
                if isinstance(r, (asyncio.CancelledError, AgentCancelledError)):
                    raise r
            for gi, idx in enumerate(safe_indices):
                if isinstance(gather_results[gi], BaseException) and exceptions[idx] is None:
                    exceptions[idx] = gather_results[gi]
                    results[idx] = False

        # Phase 2b: run unsafe tools serially; bash error → cancel rest
        bash_errored = False
        for idx in unsafe_indices:
            fc = function_calls[idx]
            if bash_errored:
                # Sibling-error cancellation (mirrors CC's siblingAbortController)
                exceptions[idx] = RuntimeError(
                    f"Cancelled: sibling bash/execute tool errored"
                )
                results[idx] = False
                timers[idx].start()
                timers[idx].stop()
                continue
            # Cancellation check before each unsafe tool (interrupt_behavior aware).
            # "cancel" tools are skipped; "block" tools are allowed to run.
            if _agent is not None and getattr(_agent, '_cancelled', False):
                if fc.function.interrupt_behavior == "cancel":
                    exceptions[idx] = RuntimeError("Tool cancelled by user")
                    results[idx] = False
                    timers[idx].start()
                    timers[idx].stop()
                    continue
            # Input guardrail check (before execution)
            if _has_guardrails:
                _fc_args = json.dumps(fc.arguments) if fc.arguments else None
                reject_msg = await check_input_guardrails(
                    _agent, fc.function.name, _fc_args, fc.call_id,
                )
                if reject_msg is not None:
                    fc.result = reject_msg
                    fc.error = reject_msg
                    results[idx] = False
                    timers[idx].start()
                    timers[idx].stop()
                    continue
            timers[idx].start()
            try:
                if fc.function.manages_own_timeout:
                    results[idx] = await fc.execute()
                else:
                    _timeout = fc.function.timeout or _DEFAULT_TOOL_TIMEOUT
                    results[idx] = await asyncio.wait_for(fc.execute(), timeout=_timeout)
                # Output guardrail check (after execution)
                if _has_guardrails:
                    _fc_args = json.dumps(fc.arguments) if fc.arguments else None
                    reject_msg = await check_output_guardrails(
                        _agent, fc.function.name, _fc_args, fc.call_id, fc.result,
                    )
                    if reject_msg is not None:
                        fc.result = reject_msg
                        fc.error = reject_msg
                        results[idx] = False
            except asyncio.TimeoutError:
                exceptions[idx] = TimeoutError(
                    f"Tool '{fc.function.name}' timed out after {fc.function.timeout or _DEFAULT_TOOL_TIMEOUT}s"
                )
                results[idx] = False
                if fc.function.name in _SHELL_TOOL_NAMES:
                    bash_errored = True
            except ToolCallException as tce:
                exceptions[idx] = tce
                results[idx] = False
                if fc.function.name in _SHELL_TOOL_NAMES:
                    bash_errored = True
            except AgentCancelledError:
                raise
            except Exception as exc:
                exceptions[idx] = exc
                results[idx] = False
                if fc.function.name in _SHELL_TOOL_NAMES:
                    bash_errored = True
            finally:
                timers[idx].stop()

        # Phase 3: Process results in original order
        for i, function_call in enumerate(function_calls):
            function_call_success = results[i] if not isinstance(results[i], Exception) else False
            stop_execution_after_tool_call = False
            additional_messages_from_function_call = []

            # Handle exceptions captured during execution
            exc = exceptions[i]
            if exc is not None:
                if isinstance(exc, ToolCallException):
                    tce = exc
                    if tce.user_message is not None:
                        if isinstance(tce.user_message, str):
                            additional_messages_from_function_call.append(Message(role="user", content=tce.user_message))
                        else:
                            additional_messages_from_function_call.append(tce.user_message)
                    if tce.agent_message is not None:
                        if isinstance(tce.agent_message, str):
                            additional_messages_from_function_call.append(
                                Message(role="assistant", content=tce.agent_message)
                            )
                        else:
                            additional_messages_from_function_call.append(tce.agent_message)
                    if tce.messages is not None and len(tce.messages) > 0:
                        for m in tce.messages:
                            if isinstance(m, Message):
                                additional_messages_from_function_call.append(m)
                            elif isinstance(m, dict):
                                try:
                                    additional_messages_from_function_call.append(Message(**m))
                                except Exception as e:
                                    logger.warning(f"Failed to convert dict to Message: {e}")
                    if tce.stop_execution:
                        stop_execution_after_tool_call = True
                        if len(additional_messages_from_function_call) > 0:
                            for m in additional_messages_from_function_call:
                                m.stop_after_tool_call = True
                else:
                    # Generic exception — treat as tool failure
                    function_call.error = str(exc)
                    logger.warning(f"Tool {function_call.function.name} failed: {exc}")

            function_call_output: Optional[Union[List[Any], str]] = ""
            if isinstance(function_call.result, (GeneratorType, collections.abc.Iterator)):
                stream_redaction_buffer = ""
                for item in function_call.result:
                    item_text = str(item)
                    function_call_output += item_text
                    if function_call.function.show_result:
                        stream_redaction_buffer += item_text
                        redacted_chunk, stream_redaction_buffer = _take_redactable_stream_text(
                            stream_redaction_buffer
                        )
                        if redacted_chunk:
                            yield ModelResponse(content=redacted_chunk)
                if function_call.function.show_result and stream_redaction_buffer:
                    redacted_chunk, stream_redaction_buffer = _take_redactable_stream_text(
                        stream_redaction_buffer,
                        final=True,
                    )
                    if redacted_chunk:
                        yield ModelResponse(content=redacted_chunk)
            else:
                function_call_output = function_call.result
                # Ensure output is always str for tool Message.content — some
                # providers reject list-type tool results (M-02 fix)
                if function_call_output is not None and not isinstance(function_call_output, str):
                    function_call_output = str(function_call_output)

            if isinstance(function_call_output, str):
                function_call_output = _redact_unterminated_private_key_block(function_call_output)
            if function_call.function.show_result and not isinstance(function_call.result, (GeneratorType, collections.abc.Iterator)):
                yield ModelResponse(content=function_call_output)

            # --- Layer 1: per-tool large result persistence ---
            # Persist to ~/.agentica/projects/<project-hash>/<session-id>/tool-results/
            if (
                function_call_success
                and isinstance(function_call_output, str)
                and function_call.function.max_result_size_chars is not None
            ):
                try:
                    from agentica.compression.tool_result_storage import maybe_persist_result
                    _agent = self._agent_ref() if self._agent_ref else None
                    _sid = _agent.run_id or 'default' if _agent else 'default'
                    # user_id partitions the persisted-output directory so two
                    # tenants whose agents share a cwd can't read each other's
                    # spills. None falls back to "default" inside the helper.
                    _uid = (
                        _agent.workspace.user_id
                        if _agent and _agent.workspace is not None
                        else None
                    )
                    function_call_output = maybe_persist_result(
                        tool_name=function_call.function.name,
                        tool_use_id=function_call.call_id or f"call_{i}",
                        content=function_call_output,
                        session_id=_sid,
                        max_result_size_chars=function_call.function.max_result_size_chars,
                        user_id=_uid,
                    )
                except Exception as persist_err:
                    logger.debug(f"Tool result persistence skipped: {persist_err}")

            # Soft-error detection: builtin tools return error as prefixed
            # strings ("Error: File not found", "Error: Directory not found", ...)
            # rather than raising. Without this, ExperienceCaptureHooks and the
            # death-spiral detector treat them as successes, and the whole
            # self-evolution pipeline goes blind to "tool ran but failed".
            # We flip the error flag but keep the original output as content so
            # the LLM still sees the error message verbatim.
            soft_error = (
                function_call_success
                and isinstance(function_call_output, str)
                and function_call_output.lstrip().startswith("Error:")
            )
            if soft_error:
                function_call_success = False

            # Track repeated failures and append a notice (not blocking).
            # SECURITY: tool error strings are written back into the LLM context
            # in the next turn. They may carry traceback fragments containing
            # API keys, Authorization headers, env-style assignments, JWTs, or
            # URL tokens (e.g. when a tool wraps an HTTPS client and re-raises
            # the URL on failure). We sanitize the error branch before it ever
            # touches Message.content. Successful outputs are NOT sanitized to
            # preserve structured payloads verbatim.
            if function_call_success or soft_error:
                _result_content = function_call_output
            else:
                _raw_err = function_call.error
                _result_content = (
                    redact_sensitive_text(_raw_err) if isinstance(_raw_err, str) else _raw_err
                )
            if not function_call_success:
                _call_key = f"{function_call.function.name}:{json.dumps(function_call.arguments, sort_keys=True, default=str)}"
                failed_call_counts[_call_key] = failed_call_counts.get(_call_key, 0) + 1
                _n = failed_call_counts[_call_key]
                if _n >= 2 and isinstance(_result_content, str):
                    _result_content += (
                        f"\n\n[Notice: This exact call has failed {_n} times "
                        f"this run with the same error. Consider a different approach.]"
                    )

            function_call_result = Message(
                role=tool_role,
                content=_result_content,
                tool_call_id=function_call.call_id,
                tool_name=function_call.function.name,
                tool_args=function_call.arguments,
                tool_call_error=not function_call_success,
                stop_after_tool_call=function_call.function.stop_after_tool_call or stop_execution_after_tool_call,
                metrics={"time": timers[i].elapsed},
            )

            yield ModelResponse(
                content=f"{function_call.get_call_str()} completed in {timers[i].elapsed:.4f}s.",
                tool_call=function_call_result.model_dump(
                    include={
                        "content",
                        "tool_call_id",
                        "tool_name",
                        "tool_args",
                        "tool_call_error",
                        "metrics",
                        "created_at",
                    }
                ),
                event=ModelResponseEvent.tool_call_completed.value,
            )

            # --- Lifecycle: tool end ---
            if (
                _agent is not None
                and hasattr(_agent, '_run_hooks')
                and _run_hook_method_overridden(_agent._run_hooks, "on_tool_end")
            ):
                # For soft-errors (builtin tools returning "Error: ..." strings
                # without raising), function_call.error is None but the output
                # carries the human-readable error message. Forward the output
                # so ExperienceCaptureHooks records a non-empty error.
                hook_result = (
                    function_call_output
                    if (function_call_success or soft_error)
                    else function_call.error
                )
                tool_end_input = {
                    "tool_name": function_call.function.name,
                    "tool_call_id": function_call.call_id or "",
                    "tool_args": function_call.arguments,
                    "is_error": not function_call_success,
                    "elapsed": timers[i].elapsed,
                }
                with langfuse_span_context(
                    name="hook.run.on_tool_end",
                    input_data=tool_end_input,
                    metadata={
                        "hook": "run.on_tool_end",
                        "agent_id": _agent.agent_id,
                        "agent_name": _agent.name or "Agent",
                        "run_id": _agent.run_id,
                    },
                ) as span:
                    await _agent._run_hooks.on_tool_end(
                        agent=_agent,
                        tool_name=function_call.function.name,
                        tool_call_id=function_call.call_id or "",
                        tool_args=function_call.arguments,
                        result=hook_result,
                        is_error=not function_call_success,
                        elapsed=timers[i].elapsed,
                    )
                    span_output = hook_result
                    if span_output is None:
                        span_output = {"status": "completed"}
                    update_langfuse_span(span, output=span_output)

            if "tool_call_times" not in self.metrics:
                self.metrics["tool_call_times"] = {}
            if function_call.function.name not in self.metrics["tool_call_times"]:
                self.metrics["tool_call_times"][function_call.function.name] = []
            self.metrics["tool_call_times"][function_call.function.name].append(timers[i].elapsed)

            function_call_results.append(function_call_result)
            if len(additional_messages_from_function_call) > 0:
                function_call_results.extend(additional_messages_from_function_call)
            function_call_stack.append(function_call)

        # Check tool_call_limit after processing all results in the current batch.
        # Moving this outside the loop ensures every tool_call_id from the assistant
        # message gets a corresponding tool result message (required by OpenAI API).
        if self.tool_call_limit and len(function_call_stack) >= self.tool_call_limit:
            self.deactivate_function_calls()

        # --- Layer 2: per-message budget enforcement ---
        # If the total tool results in this batch exceed the budget, persist the
        # largest ones to disk until under budget.
        try:
            from agentica.compression.tool_result_storage import enforce_tool_result_budget
            _agent = self._agent_ref() if self._agent_ref else None
            _sid = _agent.run_id or 'default' if _agent else 'default'
            _uid = (
                _agent.workspace.user_id
                if _agent and _agent.workspace is not None
                else None
            )
            enforce_tool_result_budget(
                tool_results=function_call_results,
                session_id=_sid,
                user_id=_uid,
            )
        except Exception as budget_err:
            logger.warning(f"Tool result budget enforcement failed: {budget_err}")

    # ── Default tool call handling (OpenAI-compatible protocol) ──────────────
    # Providers using a different protocol (e.g. Anthropic) override these.

    async def handle_tool_calls(
            self,
            assistant_message: Message,
            messages: List[Message],
            model_response: ModelResponse,
            tool_role: str = "tool",
    ) -> Optional[ModelResponse]:
        """Handle tool calls in the assistant message (OpenAI-compatible default).

        Providers with a different tool-call protocol (e.g. Claude) should override.
        """
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            if model_response.content is None:
                model_response.content = ""
            function_call_results: List[Message] = []
            function_calls_to_run: List[FunctionCall] = []
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(
                        Message(role=tool_role, tool_call_id=_tool_call_id, content="Could not find function to call.")
                    )
                    continue
                if _function_call.error is not None:
                    messages.append(
                        Message(role=tool_role, tool_call_id=_tool_call_id, content=_function_call.error)
                    )
                    continue
                function_calls_to_run.append(_function_call)

            async for tool_response in self.run_function_calls(
                    function_calls=function_calls_to_run, function_call_results=function_call_results,
                    tool_role=tool_role
            ):
                pass

            if len(function_call_results) > 0:
                messages.extend(function_call_results)

            return model_response
        return None

    async def handle_stream_tool_calls(
            self,
            assistant_message: Message,
            messages: List[Message],
            tool_role: str = "tool",
    ) -> AsyncIterator[ModelResponse]:
        """Handle tool calls for response stream (OpenAI-compatible default).

        Providers with a different tool-call protocol (e.g. Claude, Ollama) should override.
        """
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            function_calls_to_run: List[FunctionCall] = []
            function_call_results: List[Message] = []
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(
                        Message(role=tool_role, tool_call_id=_tool_call_id, content="Could not find function to call.")
                    )
                    continue
                if _function_call.error is not None:
                    messages.append(
                        Message(role=tool_role, tool_call_id=_tool_call_id, content=_function_call.error)
                    )
                    continue
                function_calls_to_run.append(_function_call)

            async for function_call_response in self.run_function_calls(
                    function_calls=function_calls_to_run, function_call_results=function_call_results,
                    tool_role=tool_role
            ):
                yield function_call_response

            if len(function_call_results) > 0:
                messages.extend(function_call_results)

    # ── Default usage metrics update (shared across providers) ───────────────

    def update_usage_metrics(
            self, assistant_message: Message, metrics: Metrics, response_usage: Optional[Any]
    ) -> None:
        """Update usage metrics from a non-streaming response.

        Default implementation handles OpenAI-style CompletionUsage.
        Providers with different usage formats (Anthropic, Ollama) override this.
        """
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)
        if response_usage:
            # Two paths: Ollama returns dict, OpenAI returns CompletionUsage object
            if isinstance(response_usage, dict):
                prompt_tokens = response_usage.get("prompt_eval_count", 0)
                completion_tokens = response_usage.get("eval_count", 0)
                total_tokens = prompt_tokens + completion_tokens
            else:
                prompt_tokens = response_usage.prompt_tokens or 0
                completion_tokens = response_usage.completion_tokens or 0
                total_tokens = response_usage.total_tokens or (prompt_tokens + completion_tokens)

            if prompt_tokens:
                metrics.input_tokens = prompt_tokens
                metrics.prompt_tokens = prompt_tokens
                assistant_message.metrics["input_tokens"] = prompt_tokens
                assistant_message.metrics["prompt_tokens"] = prompt_tokens
                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + prompt_tokens
                self.metrics["prompt_tokens"] = self.metrics.get("prompt_tokens", 0) + prompt_tokens
            if completion_tokens:
                metrics.output_tokens = completion_tokens
                metrics.completion_tokens = completion_tokens
                assistant_message.metrics["output_tokens"] = completion_tokens
                assistant_message.metrics["completion_tokens"] = completion_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + completion_tokens
                self.metrics["completion_tokens"] = self.metrics.get("completion_tokens", 0) + completion_tokens
            if total_tokens:
                metrics.total_tokens = total_tokens
                assistant_message.metrics["total_tokens"] = total_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + total_tokens

            entry = RequestUsage(
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                total_tokens=metrics.total_tokens,
                response_time=metrics.response_timer.elapsed,
            )

            # Parse prompt_tokens_details (only on CompletionUsage, not dict)
            if not isinstance(response_usage, dict):
                prompt_details = response_usage.prompt_tokens_details
                if prompt_details is not None:
                    from pydantic import BaseModel as PydanticBaseModel
                    if isinstance(prompt_details, dict):
                        metrics.prompt_tokens_details = prompt_details
                    elif isinstance(prompt_details, PydanticBaseModel):
                        metrics.prompt_tokens_details = prompt_details.model_dump(exclude_none=True)
                    assistant_message.metrics["prompt_tokens_details"] = metrics.prompt_tokens_details
                    if metrics.prompt_tokens_details is not None:
                        entry.input_tokens_details = TokenDetails(
                            cached_tokens=metrics.prompt_tokens_details.get("cached_tokens", 0),
                        )
                        if "prompt_tokens_details" not in self.metrics:
                            self.metrics["prompt_tokens_details"] = {}
                        for k, v in metrics.prompt_tokens_details.items():
                            self.metrics["prompt_tokens_details"][k] = self.metrics["prompt_tokens_details"].get(k, 0) + v

            # Parse completion_tokens_details (only on CompletionUsage, not dict)
            if not isinstance(response_usage, dict):
                completion_details = response_usage.completion_tokens_details
                if completion_details is not None:
                    from pydantic import BaseModel as PydanticBaseModel
                    if isinstance(completion_details, dict):
                        metrics.completion_tokens_details = completion_details
                    elif isinstance(completion_details, PydanticBaseModel):
                        metrics.completion_tokens_details = completion_details.model_dump(exclude_none=True)
                    assistant_message.metrics["completion_tokens_details"] = metrics.completion_tokens_details
                    if metrics.completion_tokens_details is not None:
                        entry.output_tokens_details = TokenDetails(
                            reasoning_tokens=metrics.completion_tokens_details.get("reasoning_tokens", 0),
                        )
                        if "completion_tokens_details" not in self.metrics:
                            self.metrics["completion_tokens_details"] = {}
                        for k, v in metrics.completion_tokens_details.items():
                            self.metrics["completion_tokens_details"][k] = self.metrics["completion_tokens_details"].get(k, 0) + v

            self.usage.add(entry)

            # Cost tracking (v3): record USD cost for this invoke()
            if self._cost_tracker is not None:
                cache_read = 0
                prompt_details_dict = metrics.prompt_tokens_details or {}
                if isinstance(prompt_details_dict, dict):
                    cache_read = prompt_details_dict.get("cached_tokens", 0)
                self._cost_tracker.record(
                    model_id=self.id,
                    input_tokens=metrics.input_tokens,
                    output_tokens=metrics.output_tokens,
                    cache_read_tokens=cache_read,
                )

    def update_stream_metrics(self, assistant_message: Message, metrics: Metrics) -> None:
        """Update usage metrics from a streaming response.

        Shared across all providers that use streaming.
        """
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)

        if metrics.time_to_first_token is not None:
            assistant_message.metrics["time_to_first_token"] = metrics.time_to_first_token
            self.metrics.setdefault("time_to_first_token", []).append(metrics.time_to_first_token)

        if metrics.input_tokens:
            assistant_message.metrics["input_tokens"] = metrics.input_tokens
            self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + metrics.input_tokens
        if metrics.output_tokens:
            assistant_message.metrics["output_tokens"] = metrics.output_tokens
            self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + metrics.output_tokens
        if metrics.prompt_tokens:
            assistant_message.metrics["prompt_tokens"] = metrics.prompt_tokens
            self.metrics["prompt_tokens"] = self.metrics.get("prompt_tokens", 0) + metrics.prompt_tokens
        if metrics.completion_tokens:
            assistant_message.metrics["completion_tokens"] = metrics.completion_tokens
            self.metrics["completion_tokens"] = self.metrics.get("completion_tokens", 0) + metrics.completion_tokens
        if metrics.total_tokens:
            assistant_message.metrics["total_tokens"] = metrics.total_tokens
            self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + metrics.total_tokens

        entry = RequestUsage(
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            response_time=metrics.response_timer.elapsed,
        )
        if metrics.prompt_tokens_details is not None:
            assistant_message.metrics["prompt_tokens_details"] = metrics.prompt_tokens_details
            entry.input_tokens_details = TokenDetails(
                cached_tokens=metrics.prompt_tokens_details.get("cached_tokens", 0),
            )
            if "prompt_tokens_details" not in self.metrics:
                self.metrics["prompt_tokens_details"] = {}
            for k, v in metrics.prompt_tokens_details.items():
                self.metrics["prompt_tokens_details"][k] = self.metrics["prompt_tokens_details"].get(k, 0) + v
        if metrics.completion_tokens_details is not None:
            assistant_message.metrics["completion_tokens_details"] = metrics.completion_tokens_details
            entry.output_tokens_details = TokenDetails(
                reasoning_tokens=metrics.completion_tokens_details.get("reasoning_tokens", 0),
            )
            if "completion_tokens_details" not in self.metrics:
                self.metrics["completion_tokens_details"] = {}
            for k, v in metrics.completion_tokens_details.items():
                self.metrics["completion_tokens_details"][k] = self.metrics["completion_tokens_details"].get(k, 0) + v
        self.usage.add(entry)

        # Cost tracking (v3): record USD cost for this streaming invoke()
        if self._cost_tracker is not None:
            cache_read = 0
            if metrics.prompt_tokens_details is not None:
                cache_read = metrics.prompt_tokens_details.get("cached_tokens", 0)
            self._cost_tracker.record(
                model_id=self.id,
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                cache_read_tokens=cache_read,
            )

    def _process_string_image(self, image: str) -> Dict[str, Any]:
        """Process string-based image (base64, URL, or file path)."""

        # Process Base64 encoded image
        if image.startswith("data:image"):
            return {"type": "image_url", "image_url": {"url": image}}

        # Process URL image
        if image.startswith(("http://", "https://")):
            return {"type": "image_url", "image_url": {"url": image}}

        # Process local file image
        import mimetypes
        from pathlib import Path

        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image}")

        mime_type = mimetypes.guess_type(image)[0] or "image/jpeg"
        with open(path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:{mime_type};base64,{base64_image}"
            return {"type": "image_url", "image_url": {"url": image_url}}

    def _process_pil_image(self, image) -> Dict[str, Any]:
        """Process PIL Image data."""
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Convert to base64
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        image_url = f"data:image/png;base64,{base64_image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    def _process_bytes_image(self, image: bytes) -> Dict[str, Any]:
        """Process bytes image data."""
        base64_image = base64.b64encode(image).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{base64_image}"
        return {"type": "image_url", "image_url": {"url": image_url}}

    def process_image(self, image: Any) -> Optional[Dict[str, Any]]:
        """Process an image based on the format."""
        from PIL.Image import Image as PILImage
        if isinstance(image, dict):
            return {"type": "image_url", "image_url": image}

        if isinstance(image, str):
            return self._process_string_image(image)

        if isinstance(image, bytes):
            return self._process_bytes_image(image)

        if isinstance(image, PILImage):
            return self._process_pil_image(image)

        logger.warning(f"Unsupported image type: {type(image)}")
        return None

    def add_images_to_message(self, message: Message, images: Optional[Sequence[Any]] = None) -> Message:
        """
        Add images to a message for the model. By default, we use the OpenAI image format but other Models
        can override this method to use a different image format.
        Args:
            message: The message for the Model
            images: Sequence of images in various formats:
                - str: base64 encoded image, URL, or file path
                - Dict: pre-formatted image data
                - bytes: raw image data

        Returns:
            Message content with images added in the format expected by the model
        """
        # If no images are provided, return the message as is
        if images is None or len(images) == 0:
            return message

        # Ignore non-string message content
        # because we assume that the images/audio are already added to the message
        if not isinstance(message.content, str):
            return message

        # Create a default message content with text
        message_content_with_image: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]

        # Add images to the message content
        for image in images:
            try:
                image_data = self.process_image(image)
                if image_data:
                    message_content_with_image.append(image_data)
            except Exception as e:
                logger.error(f"Failed to process image: {str(e)}")
                continue

        # Update the message content with the images
        message.content = message_content_with_image
        return message

    def add_audio_to_message(self, message: Message, audio: Optional[Any] = None) -> Message:
        """
        Add audio to a message for the model. By default, we use the OpenAI audio format but other Models
        can override this method to use a different audio format.
        Args:
            message: The message for the Model
            audio: Pre-formatted audio data like {
                        "data": encoded_string,
                        "format": "wav"
                    }

        Returns:
            Message content with audio added in the format expected by the model
        """
        if audio is None:
            return message

        # If `id` is in the audio, this means the audio is already processed
        # This is used in multi-turn conversations
        if "id" in audio:
            message.content = ""
            message.audio = {"id": audio["id"]}
        # If `data` is in the audio, this means the audio is raw data
        # And an input audio
        elif "data" in audio:
            # Create a message with audio
            message.content = [
                {"type": "text", "text": message.content},
                {"type": "input_audio", "input_audio": audio},
            ]
        return message

    def get_system_message_for_model(self) -> Optional[str]:
        return self.system_prompt

    def get_instructions_for_model(self) -> Optional[List[str]]:
        return self.instructions

    def clear(self) -> None:
        """Clears the Model's state."""
        self.metrics = {}
        self.usage = Usage()
        self.functions = None
        self.function_call_stack = None
        self._failed_call_counts = None
        state = self.current_run_state()
        if state is not None:
            state.function_call_stack = []
            state.failed_call_counts = {}
            state.tool_choice = None
        self.session_id = None
        self.last_finish_reason = None
