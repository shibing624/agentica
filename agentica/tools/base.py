# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
This module provides classes for managing tools.
It includes an abstract base class for tools and a class for managing tool instances.
"""

import asyncio
import ast
import inspect
import json
import re
import weakref
from collections import OrderedDict
from typing import Callable, get_type_hints, Any, Dict, Union, Optional, Type, TypeVar, List
from pydantic import BaseModel, Field, ValidationError, field_validator, validate_call
from agentica.model.message import Message
from agentica.tools.origin import ToolOrigin
from agentica.utils.log import logger

T = TypeVar("T")

# OpenAI / Anthropic / most providers require tool names to match this pattern.
# See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_TOOL_NAME_ILLEGAL_CHAR_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_TOOL_NAME_REPEAT_UNDERSCORE_RE = re.compile(r"_{2,}")
_TOOL_NAME_REPEAT_DASH_RE = re.compile(r"-{2,}")


def validate_tool_name(name: str) -> None:
    """Raise ``ValueError`` iff ``name`` is not a legal LLM tool-call function name.

    Used for explicit names provided by the user (``Function(name=...)`` and
    ``Agent.as_tool(tool_name=...)``). Auto-derived names go through
    :func:`normalize_tool_name` instead.
    """
    if not isinstance(name, str) or not _TOOL_NAME_RE.match(name):
        raise ValueError(
            f"Invalid tool name {name!r}: must match ^[a-zA-Z0-9_-]{{1,64}}$ "
            "(letters, digits, '_' or '-', length 1..64)."
        )


def normalize_tool_name(raw: str) -> str:
    """Coerce ``raw`` into a legal tool-call function name.

    Pipeline: lowercase → replace illegal chars with '_' → collapse repeats
    → strip leading/trailing '_' or '-' → truncate to 64 → fall back to
    ``"tool"`` for empty input. Idempotent.
    """
    if not isinstance(raw, str) or not raw:
        return "tool"
    lowered = raw.lower()
    cleaned = _TOOL_NAME_ILLEGAL_CHAR_RE.sub("_", lowered)
    cleaned = _TOOL_NAME_REPEAT_UNDERSCORE_RE.sub("_", cleaned)
    cleaned = _TOOL_NAME_REPEAT_DASH_RE.sub("-", cleaned)
    cleaned = cleaned.strip("_-")
    if not cleaned:
        return "tool"
    return cleaned[:64]


class ToolCallException(Exception):
    def __init__(
            self,
            exc,
            user_message: Optional[Union[str, Message]] = None,
            agent_message: Optional[Union[str, Message]] = None,
            messages: Optional[List[Union[dict, Message]]] = None,
            stop_execution: bool = False,
    ):
        super().__init__(exc)
        self.user_message = user_message
        self.agent_message = agent_message
        self.messages = messages
        self.stop_execution = stop_execution


class RetryAgentRun(ToolCallException):
    """Exception raised when a tool call should be retried."""


class StopAgentRun(ToolCallException):
    """Exception raised when an agent should stop executing entirely."""

    def __init__(
            self,
            exc,
            user_message: Optional[Union[str, Message]] = None,
            agent_message: Optional[Union[str, Message]] = None,
            messages: Optional[List[Union[dict, Message]]] = None,
    ):
        super().__init__(
            exc, user_message=user_message, agent_message=agent_message, messages=messages, stop_execution=True
        )


def _format_tool_error(tool_name: str, exc: Exception) -> str:
    """Render a tool execution error as a single human-readable line.

    The CLI surfaces this string to the end user (and the model also sees it
    as the tool result), so we strip the Python class boilerplate and — for
    pydantic ``ValidationError`` — extract just the offending field + reason
    instead of dumping the multi-line pydantic banner with a ``errors.pydantic.dev``
    URL.
    """
    if isinstance(exc, ValidationError):
        parts: List[str] = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ())) or "?"
            msg = err.get("msg", "invalid value")
            given = err.get("input")
            given_repr = ""
            if given is not None:
                gs = repr(given)
                if len(gs) > 60:
                    gs = gs[:57] + "..."
                given_repr = f" (got {gs})"
            parts.append(f"{loc}: {msg}{given_repr}")
        joined = "; ".join(parts)
        return f"invalid arguments for {tool_name}: {joined}"

    text = str(exc).strip()
    if not text:
        text = exc.__class__.__name__
    return text


def _safe_validate_call(c: Callable) -> Callable:
    """Wrap validate_call, stripping unresolvable forward-reference annotations.

    Pydantic's validate_call internally calls get_type_hints(), which tries to
    eval string annotations like ``self: "Agent"``. If the referenced class isn't
    importable in the function's module namespace, it raises NameError.

    Fix: remove such annotations from __annotations__ before validation. We only
    strip ``self`` (mixin methods) — other forward refs should be fixed at source.
    """
    # Handle bound methods - need to access underlying function via __func__
    func = c.__func__ if inspect.ismethod(c) else c
    annotations = getattr(func, "__annotations__", None)
    if annotations and "self" in annotations and isinstance(annotations["self"], str):
        # Don't mutate the original — copy annotations dict
        func.__annotations__ = {k: v for k, v in annotations.items() if k != "self"}
    return validate_call(c)


class Function(BaseModel):
    """Model for storing functions that can be called by an agent."""

    # The name of the function to be called.
    # Must match ^[a-zA-Z0-9_-]{1,64}$ (validated below).
    name: str

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        validate_tool_name(v)
        return v
    # A description of what the function does, used by the model to choose when and how to call the function.
    description: Optional[str] = None
    # The parameters the functions accepts, described as a JSON Schema object.
    # To describe a function that accepts no parameters, provide the value {"type": "object", "properties": {}}.
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema object describing function parameters",
    )
    # If set, `process_entrypoint` will NOT overwrite `parameters` with the
    # schema auto-derived from the entrypoint signature. Use this when the
    # function signature cannot carry rich structural information (e.g.
    # `edits: List[Dict[str, Any]]` collapses each item to a bare
    # `{"type": "object"}` with no properties, leaving the LLM no structural
    # guidance and often triggering "stringified array" tool-call failures).
    # The overriding dict IS the final schema exposed to the LLM.
    parameters_override: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    # The function to be called.
    entrypoint: Optional[Callable] = None
    # If True, the entrypoint processing is skipped and the Function is used as is.
    skip_entrypoint_processing: bool = False
    # If True, the arguments are sanitized before being passed to the function.
    sanitize_arguments: bool = True
    # If True, the function call will show the result along with sending it to the model.
    show_result: bool = False
    # If True, the agent will stop after the function call.
    stop_after_tool_call: bool = False
    # Hook that runs before the function is executed.
    # If defined, can accept the FunctionCall instance as a parameter.
    pre_hook: Optional[Callable] = None
    # Hook that runs after the function is executed, regardless of success/failure.
    # If defined, can accept the FunctionCall instance as a parameter.
    post_hook: Optional[Callable] = None
    # If True, this tool may execute concurrently with other concurrency_safe tools.
    # Read-only tools (read_file, glob, grep, web_search, fetch_url …) should be True.
    # Write/shell tools (execute, write_file, edit_file …) must remain False (default).
    # Mirrors CC's StreamingToolExecutor.isConcurrencySafe flag.
    concurrency_safe: bool = False
    # If True, the tool only reads data and never modifies state.
    # Used by permission systems to skip confirmation for safe operations.
    is_read_only: bool = False
    # If True, the tool performs irreversible operations (delete, overwrite, send, execute).
    # Used by permission systems to require extra caution or user confirmation.
    is_destructive: bool = False
    # Maximum result size (chars) before persisting to disk.
    # None = never persist (e.g. read_file — avoids reading its own persisted file).
    # Set a positive int to enable (e.g. 50_000 for execute/bash tools).
    # Mirrors CC's toolResultStorage maxResultSizeChars.
    max_result_size_chars: Optional[int] = None
    # Execution timeout in seconds. None = use default (120s).
    # Each tool invocation is wrapped with asyncio.wait_for(timeout=).
    timeout: Optional[int] = None
    # If True, the tool handles its own timeout internally (e.g. execute tool).
    # The outer asyncio.wait_for wrapper in Model.run_function_calls is skipped.
    manages_own_timeout: bool = False
    # Optional input validator called BEFORE execution.
    # Signature: (arguments: Dict[str, Any]) -> Optional[str]
    # Return None = valid; return error string = reject (skip execution, return as tool error).
    # Mirrors CC's validateInput() layer that runs before checkPermissions().
    validate_input: Optional[Callable] = None
    # Interrupt behavior when user cancels during tool execution.
    # "cancel" = tool can be cleanly terminated mid-execution (e.g. shell commands).
    # "block"  = tool must complete before honoring cancellation (e.g. agent delegation).
    # Mirrors CC's interruptBehavior() declaration.
    interrupt_behavior: str = "cancel"  # "cancel" | "block"
    # If True, tool description is NOT sent to LLM by default.
    # The tool can be discovered via a tool_search mechanism and loaded on demand.
    # Reduces per-call token cost when many tools are registered.
    # Mirrors CC's shouldDefer / isDeferredTool pattern.
    deferred: bool = False
    # Optional availability check. If set, called before including this tool
    # in the LLM schema. Return True = available, False = skip.
    # Used for tools that require API keys, specific platforms, etc.
    # Pattern borrowed from hermes-agent ToolRegistry.check_fn.
    available_when: Optional[Callable[[], bool]] = None
    # Where this tool came from. Resolved by Tool.register / Agent.as_tool /
    # McpTool when the Function is materialized; defaults to None on bare
    # construction so callers can opt out.
    origin: Optional[ToolOrigin] = None

    # --*-- FOR INTERNAL USE ONLY --*--
    # Weak reference to the agent that the function is associated with.
    # Using weakref to avoid circular references that prevent garbage collection.
    # The reference chain Agent -> Model -> functions -> Function._agent -> Agent
    # would cause memory leaks without weakref.
    _agent_ref: Optional[weakref.ReferenceType] = None

    @property
    def _agent(self) -> Optional[Any]:
        """Get the agent from weak reference. Returns None if agent was garbage collected."""
        if self._agent_ref is not None:
            return self._agent_ref()
        return None

    @_agent.setter
    def _agent(self, agent: Optional[Any]) -> None:
        """Set the agent as a weak reference to avoid circular references."""
        if agent is not None:
            self._agent_ref = weakref.ref(agent)
        else:
            self._agent_ref = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, include={"name", "description", "parameters", "strict"})

    def is_available(self) -> bool:
        """Check if this tool is currently available.

        If available_when is set, calls it and returns the result.
        If available_when is None, the tool is always available.
        Exceptions in available_when are caught and treated as unavailable.
        """
        if self.available_when is None:
            return True
        try:
            return bool(self.available_when())
        except Exception:
            return False

    @staticmethod
    def _parse_parameters(entrypoint: Callable, strict: bool = False) -> Dict[str, Any]:
        """Parse callable's type hints into JSON Schema parameters.

        Shared logic for from_callable() and process_entrypoint().
        """
        from inspect import signature
        from agentica.utils.json_schema import get_json_schema

        parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        sig = signature(entrypoint)
        try:
            type_hints = get_type_hints(entrypoint)
        except NameError:
            type_hints = {}

        if "agent" in sig.parameters:
            type_hints.pop("agent", None)

        param_type_hints = {
            name: type_hints[name]
            for name in sig.parameters
            if name in type_hints and name != "return" and name != "agent"
        }
        parameters = get_json_schema(type_hints=param_type_hints, strict=strict)

        if strict:
            parameters["required"] = [name for name in parameters["properties"] if name != "agent"]
        else:
            parameters["required"] = [
                name
                for name, param in sig.parameters.items()
                if param.default == param.empty and name != "self" and name != "agent"
            ]
        return parameters

    @classmethod
    def from_callable(cls, c: Callable, strict: bool = False) -> "Function":
        from inspect import getdoc

        function_name = normalize_tool_name(c.__name__)
        try:
            parameters = cls._parse_parameters(c, strict=strict)
        except Exception as e:
            logger.warning(f"Could not parse args for {function_name}: {e}", exc_info=True)
            parameters = {"type": "object", "properties": {}, "required": []}

        # Check for @tool decorator metadata
        metadata = getattr(c, "_tool_metadata", None)
        if metadata:
            return cls(
                name=metadata["name"],
                description=metadata.get("description") or getdoc(c),
                parameters=parameters,
                entrypoint=_safe_validate_call(c),
                show_result=metadata.get("show_result", False),
                sanitize_arguments=metadata.get("sanitize_arguments", True),
                stop_after_tool_call=metadata.get("stop_after_tool_call", False),
                concurrency_safe=metadata.get("concurrency_safe", False),
                is_read_only=metadata.get("is_read_only", False),
                is_destructive=metadata.get("is_destructive", False),
                deferred=metadata.get("deferred", False),
                interrupt_behavior=metadata.get("interrupt_behavior", "cancel"),
                available_when=metadata.get("available_when", None),
            )

        return cls(
            name=function_name,
            description=getdoc(c),
            parameters=parameters,
            entrypoint=_safe_validate_call(c),
        )

    def process_entrypoint(self, strict: bool = False):
        """Process the entrypoint and make it ready for use by an agent."""
        from inspect import getdoc
        if self.skip_entrypoint_processing:
            return
        if self.entrypoint is None:
            return

        try:
            if self.parameters_override is not None:
                # Caller supplied a hand-written schema (typically because the
                # signature type `List[Dict[str, Any]]` erases item structure).
                # Use it verbatim — do NOT let the auto-derived one clobber it.
                self.parameters = self.parameters_override
            else:
                self.parameters = self._parse_parameters(self.entrypoint, strict=strict)
        except Exception as e:
            logger.warning(f"Could not parse args for {self.name}: {e}", exc_info=True)

        self.description = getdoc(self.entrypoint) or self.description
        self.entrypoint = _safe_validate_call(self.entrypoint)

    def get_type_name(self, t: Type[T]):
        name = str(t)
        if "list" in name or "dict" in name:
            return name
        else:
            return t.__name__

    def get_definition_for_prompt_dict(self) -> Optional[Dict[str, Any]]:
        """Returns a function definition that can be used in a prompt."""

        if self.entrypoint is None:
            return None

        try:
            type_hints = get_type_hints(self.entrypoint)
        except NameError:
            type_hints = {}
        return_type = type_hints.get("return", None)
        returns = None
        if return_type is not None:
            returns = self.get_type_name(return_type)

        function_info = {
            "name": self.name,
            "description": self.description,
            "arguments": self.parameters.get("properties", {}),
            "returns": returns,
        }
        return function_info

    def get_definition_for_prompt(self) -> Optional[str]:
        """Returns a function definition that can be used in a prompt."""
        import json

        function_info = self.get_definition_for_prompt_dict()
        if function_info is not None:
            return json.dumps(function_info, indent=2)
        return None


class FunctionCall(BaseModel):
    """Model for Function Calls"""

    # The function to be called.
    function: Function
    # The arguments to call the function with.
    arguments: Optional[Dict[str, Any]] = None
    # The result of the function call.
    result: Optional[Any] = None
    # The ID of the function call.
    call_id: Optional[str] = None

    # Error while parsing arguments or running the function.
    error: Optional[str] = None

    def get_call_str(self) -> str:
        """Returns a string representation of the function call."""
        if self.arguments is None:
            return f"{self.function.name}()"

        trimmed_arguments = {}
        for k, v in self.arguments.items():
            if isinstance(v, str) and len(v) > 100:
                trimmed_arguments[k] = "..."
            else:
                trimmed_arguments[k] = v
        call_str = f"{self.function.name}({', '.join([f'{k}={v}' for k, v in trimmed_arguments.items()])})"
        return call_str

    async def _call_func(self, func: Callable, **kwargs) -> Any:
        """Call a function, auto-detecting sync/async.

        Async functions are awaited directly.
        Sync functions run in a thread pool to avoid blocking the event loop.
        """
        import functools
        if inspect.iscoroutinefunction(func):
            return await func(**kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, **kwargs)
        )

    def _build_args(self, func: Callable) -> Dict[str, Any]:
        """Build agent/fc args for a callable based on its signature."""
        from inspect import signature
        args = {}
        params = signature(func).parameters
        if "agent" in params:
            args["agent"] = self.function._agent
        if "fc" in params:
            args["fc"] = self
        return args

    async def _run_hook(self, hook: Optional[Callable]) -> None:
        """Execute a pre/post hook if it exists."""
        if hook is None:
            return
        hook_args = self._build_args(hook)
        try:
            await self._call_func(hook, **hook_args)
        except ToolCallException:
            raise
        except Exception as e:
            logger.warning(f"Error in hook callback: {e}")
            logger.exception(e)

    async def execute(self) -> bool:
        """Execute the function (async-first, single implementation).

        Supports both sync and async entrypoints:
        - async entrypoint -> awaited directly
        - sync entrypoint -> executed in thread pool via run_in_executor

        Returns:
            bool: True if the function call was successful, False otherwise.
        """
        if self.function.entrypoint is None:
            self.error = f"No entrypoint found for function: {self.function.name}"
            logger.warning(self.error)
            return False

        # INFO: per-tool-call summary so `tail -f` users can watch agent
        # behavior without enabling full DEBUG. Args are already trimmed by
        # get_call_str() (long strings -> "..."), keeping the line bounded.
        logger.info(f"[tool] {self.get_call_str()}")

        # Input validation — runs BEFORE pre_hook and execution.
        # Mirrors CC's validateInput() → checkPermissions() → call() pipeline.
        if self.function.validate_input is not None:
            try:
                validation_error = self.function.validate_input(self.arguments or {})
                if validation_error is not None:
                    self.error = f"Input validation failed: {validation_error}"
                    logger.debug(f"validate_input rejected {self.function.name}: {validation_error}")
                    return False
            except Exception as ve:
                self.error = f"Input validation error: {ve}"
                return False

        # Pre-hook
        await self._run_hook(self.function.pre_hook)

        # Build entrypoint args
        entrypoint_args = self._build_args(self.function.entrypoint)
        if self.arguments is not None:
            entrypoint_args.update(self.arguments)

        # Execute the function
        try:
            self.result = await self._call_func(self.function.entrypoint, **entrypoint_args)
            function_call_success = True
            # Pair with the "[tool] name(args)" line above so each call has a
            # visible result at INFO. Keep the preview short — full result is
            # available via the model's tool-message at DEBUG / in storage.
            _result_preview = str(self.result) if self.result is not None else ""
            if len(_result_preview) > 150:
                _result_preview = _result_preview[:150] + "..."
            logger.info(f"[tool] {self.function.name} -> {_result_preview}")
        except ToolCallException as e:
            logger.debug(f"{e.__class__.__name__}: {e}")
            self.error = str(e)
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Translate to a clean, actionable error: this is what the model
            # sees as the tool result and what we surface in the CLI. Avoid
            # ``logger.exception`` here — it dumps a full Python traceback to
            # the user's terminal for routine "model passed bad args" cases.
            # The friendly message is logged at DEBUG (not WARNING) so it does
            # not duplicate the CLI's own tool-error line (``🔎 ... - error:
            # ...``); the traceback is preserved at DEBUG for developers.
            friendly = _format_tool_error(self.function.name, e)
            # Mirror the success-path INFO line: every tool call gets one
            # visible outcome line at INFO, success or failure.
            logger.info(f"[tool] {self.function.name} -> error: {friendly}")
            logger.debug("Tool call failed: %s \u2014 %s", self.function.name, friendly)
            logger.debug("Tool call traceback for %s:", self.function.name, exc_info=True)
            self.error = friendly
            return False

        # Post-hook
        await self._run_hook(self.function.post_hook)

        return function_call_success


class ModelTool(BaseModel):
    """Model for Tools"""

    model_config = {"arbitrary_types_allowed": True}

    type: str
    function: Optional[Dict[str, Any]] = None
    # Provider-side / model-hosted tools (web_search, file_search,
    # code_interpreter) carry origin=ToolOrigin(type="model", ...).
    origin: Optional[ToolOrigin] = None

    def to_dict(self) -> Dict[str, Any]:
        # Origin is metadata for hooks / session log only — never sent to
        # the provider. Strip it from the wire payload.
        return self.model_dump(exclude_none=True, exclude={"origin"})


def _coerce_number(value: str, integer_only: bool = False):
    """Try to parse *value* as a number. Returns original string on failure."""
    stripped = value.strip()
    # Try int first to preserve precision for large integers (snowflake IDs, etc.)
    try:
        return int(stripped)
    except ValueError:
        pass
    if integer_only:
        return value
    try:
        f = float(stripped)
    except (ValueError, OverflowError):
        return value
    # Guard against inf/nan
    if f != f or f == float("inf") or f == float("-inf"):
        return f
    return f


def _coerce_boolean(value: str):
    """Try to parse *value* as a boolean. Returns original string on failure."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def _coerce_value(value: str, expected_type):
    """Attempt to coerce a string *value* to *expected_type*.

    Returns the original string when coercion is not applicable or fails.
    """
    if isinstance(expected_type, list):
        # Union type — try each in order, return first successful coercion
        for t in expected_type:
            result = _coerce_value(value, t)
            if result is not value:
                return result
        return value

    if expected_type in ("integer", "number"):
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type in ("array", "object"):
        stripped = value.strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
        # Some models serialise nested array/object args as a Python-repr
        # string (single quotes, True/None) instead of valid JSON. literal_eval
        # recovers those safely (literals only, no code execution).
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, (list, dict)):
            return parsed
        return value
    return value


def coerce_tool_args(args: Dict[str, Any], function: Function) -> Dict[str, Any]:
    """Coerce tool call arguments to match their JSON Schema types.

    LLMs frequently return numbers as strings ("42" instead of 42)
    and booleans as strings ("true" instead of true). This compares
    each argument value against the tool's registered JSON Schema and
    attempts safe coercion when the value is a string but the schema
    expects a different type. Original values are preserved when
    coercion fails.

    Handles "type": "integer", "number", "boolean", "array", "object",
    and union types ("type": ["integer", "string"]).
    """
    if not args or not isinstance(args, dict):
        return args

    properties = (function.parameters or {}).get("properties")
    if not properties:
        return args

    for key, value in args.items():
        if not isinstance(value, str):
            continue
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")
        if not expected:
            continue
        coerced = _coerce_value(value, expected)
        if coerced is not value:
            args[key] = coerced

    return args


def get_function_call(
        name: str,
        arguments: Optional[str] = None,
        call_id: Optional[str] = None,
        functions: Optional[Dict[str, Function]] = None,
) -> Optional[FunctionCall]:
    # logger.debug(f"Getting function {name}")
    # logger.debug(f"Arguments: {arguments}, Call ID: {call_id}, name: {name}, functions: {functions}")
    if functions is None:
        return None

    function_to_call: Optional[Function] = None
    if name in functions:
        function_to_call = functions[name]
    if function_to_call is None:
        logger.error(f"Function {name} not found")
        return None

    function_call = FunctionCall(function=function_to_call)
    if call_id is not None:
        function_call.call_id = call_id
    if arguments is not None and arguments != "":
        try:
            if function_to_call.sanitize_arguments:
                if "None" in arguments:
                    arguments = arguments.replace("None", "null")
                if "True" in arguments:
                    arguments = arguments.replace("True", "true")
                if "False" in arguments:
                    arguments = arguments.replace("False", "false")
            _arguments = json.loads(arguments)
        except Exception as e:
            logger.error(f"Unable to decode function arguments:\n{arguments}\nError: {e}")
            function_call.error = f"Error while decoding function arguments:\n{arguments}\nError: {e}\n\n " \
                                  f"Please make sure we can json.loads() the arguments and retry."
            return function_call

        if not isinstance(_arguments, dict):
            logger.error(f"Function arguments are not a valid JSON object: {arguments}")
            function_call.error = "Function arguments are not a valid JSON object.\n\n Please fix and retry."
            return function_call

        # Schema-aware type coercion: fix LLM string→number/boolean mistakes
        _arguments = coerce_tool_args(_arguments, function_to_call)

        try:
            clean_arguments: Dict[str, Any] = {}
            for k, v in _arguments.items():
                if isinstance(v, str):
                    _v = v.strip().lower()
                    if _v in ("none", "null"):
                        clean_arguments[k] = None
                    elif _v == "true":
                        clean_arguments[k] = True
                    elif _v == "false":
                        clean_arguments[k] = False
                    else:
                        clean_arguments[k] = v.strip()
                else:
                    clean_arguments[k] = v

            function_call.arguments = clean_arguments
        except Exception as e:
            logger.error(f"Unable to parsing function arguments:\n{arguments}\nError: {e}")
            function_call.error = f"Error while parsing function arguments: {e}\n\n Please fix and retry."
            return function_call
    return function_call


def get_function_call_for_tool_call(
        tool_call: Dict[str, Any], functions: Optional[Dict[str, Function]] = None
) -> Optional[FunctionCall]:
    if tool_call.get("type") == "function":
        _tool_call_id = tool_call.get("id")
        _tool_call_function = tool_call.get("function")
        if _tool_call_function is not None:
            _tool_call_function_name = _tool_call_function.get("name")
            _tool_call_function_arguments_str = _tool_call_function.get("arguments")
            if _tool_call_function_name is not None:
                return get_function_call(
                    name=_tool_call_function_name,
                    arguments=_tool_call_function_arguments_str,
                    call_id=_tool_call_id,
                    functions=functions,
                )
    return None


def extract_tool_call_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # Extracting the content between the tags
    return text[start_index:end_index].strip()


def remove_tool_calls_from_string(text: str, start_tag: str = "<tool_call>", end_tag: str = "</tool_call>"):
    """Remove multiple tool calls from a string."""
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text


def extract_tool_from_xml(xml_str):
    # Find tool_name
    tool_name_start = xml_str.find("<tool_name>") + len("<tool_name>")
    tool_name_end = xml_str.find("</tool_name>")
    tool_name = xml_str[tool_name_start:tool_name_end].strip()

    # Find and process parameters block
    params_start = xml_str.find("<parameters>") + len("<parameters>")
    params_end = xml_str.find("</parameters>")
    parameters_block = xml_str[params_start:params_end].strip()

    # Extract individual parameters
    arguments = {}
    while parameters_block:
        # Find the next tag and its closing
        tag_start = parameters_block.find("<") + 1
        tag_end = parameters_block.find(">")
        tag_name = parameters_block[tag_start:tag_end]

        # Find the tag's closing counterpart
        value_start = tag_end + 1
        value_end = parameters_block.find(f"</{tag_name}>")
        value = parameters_block[value_start:value_end].strip()

        # Add to arguments
        arguments[tag_name] = value

        # Move past this tag
        parameters_block = parameters_block[value_end + len(f"</{tag_name}>"):].strip()

    return {"tool_name": tool_name, "parameters": arguments}


def remove_function_calls_from_string(
        text: str, start_tag: str = "<function_calls>", end_tag: str = "</function_calls>"
):
    """Remove multiple function calls from a string."""
    while start_tag in text and end_tag in text:
        start_index = text.find(start_tag)
        end_index = text.find(end_tag) + len(end_tag)
        text = text[:start_index] + text[end_index:]
    return text


class Tool:
    """Tool for managing functions."""

    def __init__(self, name: str = "tool", description: str = ""):
        self.name: str = name
        self.description = description
        self.functions: Dict[str, Function] = OrderedDict()

    def register(self, function: Callable[..., Any], sanitize_arguments: bool = True,
                 concurrency_safe: bool = False, is_read_only: bool = False,
                 is_destructive: bool = False,
                 available_when: Optional[Callable[[], bool]] = None,
                 parameters_override: Optional[Dict[str, Any]] = None):
        """Register a function with the toolkit.

        Args:
            function:           The callable to register.
            sanitize_arguments: If True, the arguments will be sanitized before
                                being passed to the function.
            concurrency_safe:   If True the function may run concurrently with other
                                concurrency_safe tools (e.g. read_file, glob).
            is_read_only:       If True, the function only reads data and never modifies state.
            is_destructive:     If True, the function performs irreversible operations.
            available_when:     Optional callback that returns True when the tool
                                should be exposed to the LLM.
            parameters_override: If given, use this dict verbatim as the LLM-facing
                                JSON schema instead of the one auto-derived from the
                                function signature. Necessary when the signature type
                                erases structure (e.g. `List[Dict[str, Any]]` items
                                collapse to `{"type": "object"}` with no properties,
                                which is a common source of tool-call failures).

        Returns:
            The registered function
        """
        if parameters_override is not None:
            if not isinstance(parameters_override, dict):
                raise TypeError(
                    f"parameters_override for {function.__name__} must be a dict, "
                    f"got {type(parameters_override).__name__}"
                )
            # JSON Schema object keys we actually consume downstream. Anything
            # outside this set is almost certainly a typo (e.g. `propertys`,
            # `require`) that would silently reach the LLM as a broken schema.
            allowed_schema_keys = {
                "type", "properties", "required", "additionalProperties",
                "description", "title", "$defs", "definitions", "enum",
                "items", "oneOf", "anyOf", "allOf", "not",
            }
            unknown = set(parameters_override.keys()) - allowed_schema_keys
            if unknown:
                raise ValueError(
                    f"parameters_override for {function.__name__} contains "
                    f"unknown JSON-schema keys: {sorted(unknown)}. "
                    f"Allowed: {sorted(allowed_schema_keys)}"
                )
        try:
            origin_type = "builtin" if (self.name and self.name.startswith("builtin_")) else "function"
            f = Function(
                name=normalize_tool_name(function.__name__),
                description=function.__doc__ or self.description,
                entrypoint=function,
                sanitize_arguments=sanitize_arguments,
                concurrency_safe=concurrency_safe,
                is_read_only=is_read_only,
                is_destructive=is_destructive,
                available_when=available_when,
                parameters_override=parameters_override,
                origin=ToolOrigin(type=origin_type, source_tool_name=function.__name__),
            )
            self.functions[f.name] = f
        except Exception as e:
            logger.warning(f"Failed to create Function for: {function.__name__}")
            raise e

    def get_system_prompt(self) -> Optional[str]:
        """Get the system prompt to inject for this tool.

        Override this method in subclasses to provide tool-specific system prompts
        that guide the LLM on how to use the tool effectively.

        Returns:
            Optional[str]: The system prompt string, or None if no prompt is needed.
        """
        return None

    def set_agent_model(self, model: Optional[Any]) -> None:
        """Receive the owning agent's model when the tool can use it.

        Most tools do not need an LLM, so the default implementation is a no-op.
        LLM-backed tools override this to avoid creating provider-specific
        fallback models when the parent agent already has a configured model.
        """
        pass

    def clone(self) -> "Tool":
        """Return an instance safe to bind to a different agent.

        Default: returns ``self`` (stateless tools can be shared across agents).
        Subclasses that hold per-agent state (agent reference, workspace,
        accumulated data) MUST override to return a fresh instance whose
        ``functions`` entrypoints are bound to that fresh instance, so multiple
        agents sharing the same logical tool config do not corrupt each other.

        Stateful built-ins that override clone():
        - ``BuiltinTodoTool``  (holds ``_agent``)
        - ``BuiltinTaskTool``  (holds ``_parent_agent``)
        - ``BuiltinMemoryTool`` (holds ``_workspace``)
        - ``SkillTool``        (holds ``_agent``)
        """
        return self

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name} functions={list(self.functions.keys())}>"

    def __str__(self):
        return self.__repr__()
