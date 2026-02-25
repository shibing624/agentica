# -*- encoding: utf-8 -*-
"""
@author: orange-crow, XuMing(xuming624@qq.com)
@description:
part of the code is from phidata
"""
import asyncio
import collections.abc
import io
import base64
import weakref
from dataclasses import dataclass, field
from types import GeneratorType
from typing import List, Iterator, AsyncIterator, Optional, Dict, Any, Callable, Union, Sequence

from agentica.utils.log import logger
from agentica.model.message import Message
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.model.usage import Usage, RequestUsage, TokenDetails
from agentica.tools.base import ModelTool, Tool, Function, FunctionCall, ToolCallException
from agentica.utils.timer import Timer


@dataclass
class Model:
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
    max_output_tokens: Optional[int] = None

    # A list of tools provided to the Model.
    tools: Optional[List[Union[ModelTool, Dict]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    run_tools: bool = True
    tool_call_limit: Optional[int] = None
    max_concurrent_tools: int = 10

    # -*- Functions available to the Model to call -*-
    functions: Optional[Dict[str, Function]] = None
    function_call_stack: Optional[List[FunctionCall]] = None

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
    structured_outputs: Optional[bool] = None
    # Whether the Model supports structured outputs.
    supports_structured_outputs: bool = False

    # --- Private fields (not in __init__ signature, used internally) ---
    _pre_tool_hook: Optional[Callable] = field(init=False, repr=False, default=None)
    _tool_call_hook: Optional[Callable] = field(init=False, repr=False, default=None)
    _post_tool_hook: Optional[Callable] = field(init=False, repr=False, default=None)
    _current_messages: Optional[List[Message]] = field(init=False, repr=False, default=None)
    _agent_ref: Optional[weakref.ref] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        # Auto-set provider if not provided
        if self.provider is None:
            self.provider = f"{self.name} ({self.id})" if self.name else self.id

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError

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

    async def invoke(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def invoke_stream(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    async def response(self, messages: List[Message]) -> ModelResponse:
        raise NotImplementedError

    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        raise NotImplementedError

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
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
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
                        self.tools.append({"type": "function", "function": func.to_dict()})
                        logger.debug(f"Function {name} from {tool.name} added to model.")

            elif isinstance(tool, Function):
                if tool.name not in self.functions:
                    tool._agent = agent
                    tool.process_entrypoint(strict=strict)
                    if strict and self.supports_structured_outputs:
                        tool.strict = True
                    self.functions[tool.name] = tool
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
                        self.tools.append({"type": "function", "function": func.to_dict()})
                        logger.debug(f"Function {func.name} added to model.")
                except Exception as e:
                    logger.warning(f"Could not add function {tool}: {e}")

    def deactivate_function_calls(self) -> None:
        # Deactivate tool calls by setting future tool calls to "none"
        # This is triggered when the function call limit is reached.
        self.tool_choice = "none"

    async def run_function_calls(
            self, function_calls: List[FunctionCall], function_call_results: List[Message], tool_role: str = "tool"
    ) -> AsyncIterator[ModelResponse]:
        """Execute tool calls with parallel execution, sequential result reporting.

        Phase 0: Pre-execution hook (context overflow check)
        Phase 1: Emit all tool_call_started events (in order)
        Phase 2: Execute all tools in parallel via TaskGroup (with concurrency limit)
        Phase 3: Process results sequentially (preserving message order)
        Phase 4: Post-execution hook (reflection/iteration checkpoint)
        """
        if self.function_call_stack is None:
            self.function_call_stack = []

        # Phase 0: Pre-execution hook (e.g., context overflow check)
        if self._pre_tool_hook is not None:
            force_answer, system_msg = self._pre_tool_hook(function_call_results)
            if force_answer:
                # Context hard limit reached — skip tool execution, inject force-answer message
                if system_msg:
                    function_call_results.append(Message(role="user", content=system_msg))
                logger.info("Pre-tool hook triggered force_answer — skipping tool execution")
                return

        # Phase 1: Emit started events for all function calls
        # Collect hook warnings to append AFTER tool results (Phase 3),
        # so we never break the required assistant(tool_calls) → tool sequence.
        deferred_warnings: List[Message] = []
        _agent = self._agent_ref() if self._agent_ref is not None else None
        for function_call in function_calls:
            # Per-tool hook (e.g., repetition detection)
            if self._tool_call_hook is not None:
                warning = self._tool_call_hook(function_call.function.name)
                if warning:
                    deferred_warnings.append(Message(role="user", content=warning))

            # --- Lifecycle: tool start ---
            if _agent is not None and hasattr(_agent, '_run_hooks') and _agent._run_hooks is not None:
                await _agent._run_hooks.on_tool_start(
                    agent=_agent,
                    tool_name=function_call.function.name,
                    tool_call_id=function_call.call_id or "",
                    tool_args=function_call.arguments,
                )

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

        # Phase 2: Execute all tools in parallel (with concurrency limit)
        timers = [Timer() for _ in function_calls]
        exceptions: List[Optional[BaseException]] = [None] * len(function_calls)
        semaphore = asyncio.Semaphore(self.max_concurrent_tools)

        async def _execute_one(idx: int, fc: FunctionCall) -> bool:
            async with semaphore:
                timers[idx].start()
                try:
                    return await fc.execute()
                except ToolCallException as tce:
                    exceptions[idx] = tce
                    return False
                except Exception as exc:
                    exceptions[idx] = exc
                    return False
                finally:
                    timers[idx].stop()

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_execute_one(i, fc)) for i, fc in enumerate(function_calls)]
        results = [t.result() for t in tasks]

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
                for item in function_call.result:
                    function_call_output += item
                    if function_call.function.show_result:
                        yield ModelResponse(content=item)
            else:
                function_call_output = function_call.result
                # Ensure output is str or list for Message.content validation
                if function_call_output is not None and not isinstance(function_call_output, (str, list)):
                    function_call_output = str(function_call_output)
                if function_call.function.show_result:
                    yield ModelResponse(content=function_call_output)

            function_call_result = Message(
                role=tool_role,
                content=function_call_output if function_call_success else function_call.error,
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
            if _agent is not None and hasattr(_agent, '_run_hooks') and _agent._run_hooks is not None:
                await _agent._run_hooks.on_tool_end(
                    agent=_agent,
                    tool_name=function_call.function.name,
                    tool_call_id=function_call.call_id or "",
                    tool_args=function_call.arguments,
                    result=function_call_output if function_call_success else function_call.error,
                    is_error=not function_call_success,
                    elapsed=timers[i].elapsed,
                )

            if "tool_call_times" not in self.metrics:
                self.metrics["tool_call_times"] = {}
            if function_call.function.name not in self.metrics["tool_call_times"]:
                self.metrics["tool_call_times"][function_call.function.name] = []
            self.metrics["tool_call_times"][function_call.function.name].append(timers[i].elapsed)

            function_call_results.append(function_call_result)
            if len(additional_messages_from_function_call) > 0:
                function_call_results.extend(additional_messages_from_function_call)
            self.function_call_stack.append(function_call)

        # Check tool_call_limit after processing all results in the current batch.
        # Moving this outside the loop ensures every tool_call_id from the assistant
        # message gets a corresponding tool result message (required by OpenAI API).
        if self.tool_call_limit and len(self.function_call_stack) >= self.tool_call_limit:
            self.deactivate_function_calls()

        # Append deferred hook warnings AFTER all tool results,
        # preserving the required assistant(tool_calls) → tool result sequence.
        if deferred_warnings:
            function_call_results.extend(deferred_warnings)

        # Phase 4: Post-execution hook (e.g., reflection/iteration checkpoint)
        if self._post_tool_hook is not None:
            self._post_tool_hook(function_call_results)

    async def handle_post_tool_call_messages(self, messages: List[Message], model_response: ModelResponse) -> ModelResponse:
        """Handle messages after tool calls (async-only, single implementation)."""
        last_message = messages[-1]
        if last_message.stop_after_tool_call:
            logger.debug("Stopping execution as stop_after_tool_call=True")
            if (
                    last_message.role == "assistant"
                    and last_message.content is not None
                    and isinstance(last_message.content, str)
            ):
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += last_message.content
        else:
            response_after_tool_calls = await self.response(messages=messages)
            if response_after_tool_calls.content is not None:
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += response_after_tool_calls.content
            if response_after_tool_calls.parsed is not None:
                model_response.parsed = response_after_tool_calls.parsed
            if response_after_tool_calls.audio is not None:
                model_response.audio = response_after_tool_calls.audio
        return model_response

    async def handle_post_tool_call_messages_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """Handle streaming messages after tool calls (async-only, single implementation)."""
        last_message = messages[-1]
        if last_message.stop_after_tool_call:
            logger.debug("Stopping execution as stop_after_tool_call=True")
            if (
                    last_message.role == "assistant"
                    and last_message.content is not None
                    and isinstance(last_message.content, str)
            ):
                yield ModelResponse(content=last_message.content)
        else:
            async for model_response in self.response_stream(messages=messages):
                yield model_response

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

    def _process_pil_image(self, image: 'PIL.Image.Image') -> Dict[str, Any]:
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
        self.session_id = None
