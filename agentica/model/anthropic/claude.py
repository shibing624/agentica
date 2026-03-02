import json
from os import getenv
from dataclasses import dataclass, field
from typing import Optional, List, AsyncIterator, Dict, Any, Union, Tuple, override

import asyncio

from pydantic import BaseModel

from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.metrics import Metrics
from agentica.model.response import ModelResponse
from agentica.tools.base import FunctionCall, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer

try:
    from anthropic import AsyncAnthropic as AnthropicClient
    from anthropic.types import (
        Message as AnthropicMessage, TextBlock, ToolUseBlock, Usage, TextDelta,
        ThinkingBlock, RedactedThinkingBlock, ThinkingDelta, SignatureDelta,
    )
    from anthropic.lib.streaming._types import (
        MessageStopEvent,
        RawContentBlockDeltaEvent,
        ContentBlockStopEvent,
    )
except (ModuleNotFoundError, ImportError):
    raise ImportError("`anthropic` not installed. Please install using `pip install anthropic`")


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
    max_output_tokens: int = 8192

    # Request parameters
    max_tokens: int = 8192
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    # Extended thinking: {"type": "enabled", "budget_tokens": 10000}
    thinking: Optional[Dict[str, Any]] = None
    request_params: Optional[Dict[str, Any]] = None

    # Client parameters
    api_key: Optional[str] = None
    timeout: Optional[float] = None
    client_params: Optional[Dict[str, Any]] = None

    # Anthropic client
    client: Optional[AnthropicClient] = None

    # Structured output support
    structured_outputs: bool = False
    supports_structured_outputs: bool = True

    def _get_structured_output_tool(self) -> Optional[Dict[str, Any]]:
        """Build a synthetic tool from response_format for structured output via tool_use."""
        if (
            self.response_format is not None
            and self.structured_outputs
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
        if self.client:
            return self.client

        self.api_key = self.api_key or getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not set. Please set the ANTHROPIC_API_KEY environment variable.")

        _client_params: Dict[str, Any] = {}
        # Set client parameters if they are provided
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.timeout is not None:
            _client_params["timeout"] = self.timeout
        if self.client_params:
            _client_params.update(self.client_params)
        return AnthropicClient(**_client_params)

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """
        Generate keyword arguments for API requests.

        Returns:
            Dict[str, Any]: A dictionary of keyword arguments for API requests.
        """
        _request_params: Dict[str, Any] = {}
        _request_params["max_tokens"] = self.max_tokens
        if self.temperature:
            _request_params["temperature"] = self.temperature
        if self.stop_sequences:
            _request_params["stop_sequences"] = self.stop_sequences
        if self.top_p:
            _request_params["top_p"] = self.top_p
        if self.top_k:
            _request_params["top_k"] = self.top_k
        if self.thinking:
            _request_params["thinking"] = self.thinking
        if self.request_params:
            _request_params.update(self.request_params)
        return _request_params

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
            if message.role == "system" or (message.role != "user" and idx in [0, 1]):
                system_messages.append(content)  # type: ignore
            else:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                if message.role == "user" and message.images is not None:
                    for image in message.images:
                        image_content = await self.add_image(image)
                        if image_content:
                            content.append(image_content)

                chat_messages.append({"role": message.role, "content": content})  # type: ignore
        return chat_messages, " ".join(system_messages)

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

        return await self.get_client().messages.create(
            model=self.id, messages=chat_messages, **request_kwargs,
        )

    @override
    async def invoke_stream(self, messages: List[Message]) -> Any:
        """
        Stream a response from the Anthropic API.

        Args:
            messages (List[Message]): A list of messages to send to the model.

        Returns:
            Any: The streamed response from the model.
        """
        chat_messages, system_message = await self.format_messages(messages)
        request_kwargs = self.prepare_request_kwargs(system_message)

        return self.get_client().messages.stream(
            model=self.id, messages=chat_messages, **request_kwargs,
        )

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
            cache_read = getattr(usage, 'cache_read_input_tokens', None)
            if cache_read:
                entry.input_tokens_details = TokenDetails(cached_tokens=cache_read)
            self.usage.add(entry)

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
                        }
                    )

        # Update assistant message if tool calls are present
        if len(message_data.tool_calls) > 0:
            assistant_message.tool_calls = message_data.tool_calls
            assistant_message.content = message_data.response_block

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

        # -*- Create assistant message
        assistant_message, response_content, tool_ids = self.create_assistant_message(
            response=response, metrics=metrics
        )

        # -*- Extract structured output from tool_use block if response_format is set
        if (
            self.response_format is not None
            and self.structured_outputs
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
        if await self.handle_tool_calls(assistant_message, messages, model_response, response_content, tool_ids):
            response_after_tool_calls = await self.response(messages=messages)
            if response_after_tool_calls.content is not None:
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += response_after_tool_calls.content
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

        # Generate response
        metrics.response_timer.start()
        response = await self.invoke_stream(messages=messages)
        async with response as stream:
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

                if isinstance(delta, ContentBlockStopEvent):
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
                            }
                        )
                    message_data.response_block.append(delta.content_block)

                if isinstance(delta, MessageStopEvent):
                    message_data.response_usage = delta.message.usage
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

        # Add assistant message to messages
        messages.append(assistant_message)

        # Log response and metrics
        assistant_message.log()
        metrics.log()

        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            async for _resp in self.handle_stream_tool_calls(assistant_message, messages, message_data.tool_ids):

                yield _resp
            async for _resp in self.response_stream(messages=messages):

                yield _resp

    def get_tool_call_prompt(self) -> Optional[str]:
        if self.functions is not None and len(self.functions) > 0:
            tool_call_prompt = "Do not reflect on the quality of the returned search results in your response"
            return tool_call_prompt
        return None

    def get_system_message_for_model(self) -> Optional[str]:
        return self.get_tool_call_prompt()
