from os import getenv
from dataclasses import dataclass, field
from typing import Optional, List, AsyncIterator, Dict, Any, Union

import httpx
from pydantic import BaseModel

from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.tools.base import FunctionCall, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer

try:
    from huggingface_hub import AsyncInferenceClient
    from huggingface_hub import (
        ChatCompletionOutput,
        ChatCompletionStreamOutputDelta,
        ChatCompletionStreamOutputDeltaToolCall,
        ChatCompletionStreamOutput,
        ChatCompletionOutputMessage,
        ChatCompletionOutputUsage,
    )
except (ModuleNotFoundError, ImportError):
    raise ImportError("`huggingface_hub` not installed. Please install using `pip install huggingface_hub`")


@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_details: Optional[dict] = None
    completion_tokens_details: Optional[dict] = None
    time_to_first_token: Optional[float] = None
    response_timer: Timer = field(default_factory=Timer)

    def log(self):
        if self.time_to_first_token is not None:
            logger.debug(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        logger.debug(f"* Time to generate response:   {self.response_timer.elapsed:.4f}s")
        logger.debug(f"* Tokens per second:           {self.output_tokens / self.response_timer.elapsed:.4f} tokens/s")
        logger.debug(f"* Input tokens:                {self.input_tokens or self.prompt_tokens}")
        logger.debug(f"* Output tokens:               {self.output_tokens or self.completion_tokens}")
        logger.debug(f"* Total tokens:                {self.total_tokens}")


@dataclass
class StreamData:
    response_content: str = ""
    response_tool_calls: Optional[List[ChatCompletionStreamOutputDeltaToolCall]] = None


class HuggingFaceChat(Model):
    """
    A class for interacting with HuggingFace Hub Inference models.

    Attributes:
        id (str): The id of the HuggingFace model to use.
        name (str): The name of this chat model instance.
        provider (str): The provider of the model.
    """

    id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    name: str = "HuggingFaceChat"
    provider: str = "HuggingFace"
    context_window: int = 128000
    max_output_tokens: int = 4096

    # Request parameters
    store: Optional[bool] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Any] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Any] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    request_params: Optional[Dict[str, Any]] = None

    # Client parameters
    api_key: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[Any] = None
    default_query: Optional[Any] = None
    http_client: Optional[httpx.Client] = None
    client_params: Optional[Dict[str, Any]] = None

    # HuggingFace Hub Inference async client
    async_client: Optional[AsyncInferenceClient] = None

    def get_client_params(self) -> Dict[str, Any]:
        self.api_key = self.api_key or getenv("HF_TOKEN")
        if not self.api_key:
            logger.error("HF_TOKEN not set. Please set the HF_TOKEN environment variable.")

        _client_params: Dict[str, Any] = {}
        if self.api_key is not None:
            _client_params["api_key"] = self.api_key
        if self.base_url is not None:
            _client_params["base_url"] = self.base_url
        if self.timeout is not None:
            _client_params["timeout"] = self.timeout
        if self.max_retries is not None:
            _client_params["max_retries"] = self.max_retries
        if self.default_headers is not None:
            _client_params["default_headers"] = self.default_headers
        if self.default_query is not None:
            _client_params["default_query"] = self.default_query
        if self.client_params is not None:
            _client_params.update(self.client_params)
        return _client_params

    def get_async_client(self) -> AsyncInferenceClient:
        """Returns an asynchronous HuggingFace Hub client."""
        if self.async_client:
            return self.async_client

        _client_params: Dict[str, Any] = self.get_client_params()

        if self.http_client:
            _client_params["http_client"] = self.http_client
        else:
            _client_params["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )
        return AsyncInferenceClient(**_client_params)

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Returns keyword arguments for inference model client requests."""
        _request_params: Dict[str, Any] = {}
        if self.store is not None:
            _request_params["store"] = self.store
        if self.frequency_penalty is not None:
            _request_params["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            _request_params["logit_bias"] = self.logit_bias
        if self.logprobs is not None:
            _request_params["logprobs"] = self.logprobs
        if self.max_tokens is not None:
            _request_params["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            _request_params["presence_penalty"] = self.presence_penalty
        if self.response_format is not None:
            _request_params["response_format"] = self.response_format
        if self.seed is not None:
            _request_params["seed"] = self.seed
        if self.stop is not None:
            _request_params["stop"] = self.stop
        if self.temperature is not None:
            _request_params["temperature"] = self.temperature
        if self.top_logprobs is not None:
            _request_params["top_logprobs"] = self.top_logprobs
        if self.top_p is not None:
            _request_params["top_p"] = self.top_p
        if self.tools is not None:
            tools_for_api = self.get_tools_for_api()
            if tools_for_api:
                _request_params["tools"] = tools_for_api
                if self.tool_choice is None:
                    _request_params["tool_choice"] = "auto"
                else:
                    _request_params["tool_choice"] = self.tool_choice
        if self.request_params is not None:
            _request_params.update(self.request_params)
        return _request_params

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        _dict = super().to_dict()
        if self.store is not None:
            _dict["store"] = self.store
        if self.frequency_penalty is not None:
            _dict["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias is not None:
            _dict["logit_bias"] = self.logit_bias
        if self.logprobs is not None:
            _dict["logprobs"] = self.logprobs
        if self.max_tokens is not None:
            _dict["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            _dict["presence_penalty"] = self.presence_penalty
        if self.response_format is not None:
            _dict["response_format"] = self.response_format
        if self.seed is not None:
            _dict["seed"] = self.seed
        if self.stop is not None:
            _dict["stop"] = self.stop
        if self.temperature is not None:
            _dict["temperature"] = self.temperature
        if self.top_logprobs is not None:
            _dict["top_logprobs"] = self.top_logprobs
        if self.top_p is not None:
            _dict["top_p"] = self.top_p
        if self.tools is not None:
            tools_for_api = self.get_tools_for_api()
            if tools_for_api:
                _dict["tools"] = tools_for_api
                if self.tool_choice is None:
                    _dict["tool_choice"] = "auto"
                else:
                    _dict["tool_choice"] = self.tool_choice
        return _dict

    async def invoke(self, messages: List[Message]) -> Union[ChatCompletionOutput]:
        """Send an async chat completion request to HuggingFace Hub."""
        return await self.get_async_client().chat.completions.create(
            model=self.id,
            messages=[m.to_dict() for m in messages],
            **self.request_kwargs,
        )

    async def invoke_stream(self, messages: List[Message]) -> Any:
        """Send an async streaming chat completion request to HuggingFace."""
        async_stream = await self.get_async_client().chat.completions.create(
            model=self.id,
            messages=[m.to_dict() for m in messages],
            stream=True,
            stream_options={"include_usage": True},
            **self.request_kwargs,
        )
        async for chunk in async_stream:  # type: ignore
            yield chunk

    async def _handle_tool_calls(
            self, assistant_message: Message, messages: List[Message], model_response: ModelResponse
    ) -> Optional[ModelResponse]:
        """Handle tool calls in the assistant message (async-only)."""
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            model_response.content = ""
            tool_role: str = "tool"
            function_calls_to_run: List[FunctionCall] = []
            function_call_results: List[Message] = []
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=_tool_call_id,
                            content="Could not find function to call.",
                        )
                    )
                    continue
                if _function_call.error is not None:
                    messages.append(
                        Message(
                            role="tool",
                            tool_call_id=_tool_call_id,
                            content=_function_call.error,
                        )
                    )
                    continue
                function_calls_to_run.append(_function_call)

            async for _ in self.run_function_calls(
                    function_calls=function_calls_to_run, function_call_results=function_call_results,
                    tool_role=tool_role
            ):
                pass

            if len(function_call_results) > 0:
                messages.extend(function_call_results)

            return model_response
        return None

    def _update_usage_metrics(
            self, assistant_message: Message, metrics: Metrics, response_usage: Optional[ChatCompletionOutputUsage]
    ) -> None:
        """Update the usage metrics for the assistant message and the model."""
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)
        if response_usage:
            prompt_tokens = response_usage.prompt_tokens
            completion_tokens = response_usage.completion_tokens
            total_tokens = response_usage.total_tokens

            if prompt_tokens is not None:
                metrics.input_tokens = prompt_tokens
                metrics.prompt_tokens = prompt_tokens
                assistant_message.metrics["input_tokens"] = prompt_tokens
                assistant_message.metrics["prompt_tokens"] = prompt_tokens
                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + prompt_tokens
                self.metrics["prompt_tokens"] = self.metrics.get("prompt_tokens", 0) + prompt_tokens
            if completion_tokens is not None:
                metrics.output_tokens = completion_tokens
                metrics.completion_tokens = completion_tokens
                assistant_message.metrics["output_tokens"] = completion_tokens
                assistant_message.metrics["completion_tokens"] = completion_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + completion_tokens
                self.metrics["completion_tokens"] = self.metrics.get("completion_tokens", 0) + completion_tokens
            if total_tokens is not None:
                metrics.total_tokens = total_tokens
                assistant_message.metrics["total_tokens"] = total_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + total_tokens

            # Build structured RequestUsage entry
            from agentica.model.usage import RequestUsage, TokenDetails
            entry = RequestUsage(
                input_tokens=metrics.input_tokens,
                output_tokens=metrics.output_tokens,
                total_tokens=metrics.total_tokens,
                response_time=metrics.response_timer.elapsed,
            )
            # Parse prompt_tokens_details
            if response_usage.prompt_tokens_details is not None:
                if isinstance(response_usage.prompt_tokens_details, dict):
                    metrics.prompt_tokens_details = response_usage.prompt_tokens_details
                elif isinstance(response_usage.prompt_tokens_details, BaseModel):
                    metrics.prompt_tokens_details = response_usage.prompt_tokens_details.model_dump(exclude_none=True)
                assistant_message.metrics["prompt_tokens_details"] = metrics.prompt_tokens_details
                if metrics.prompt_tokens_details is not None:
                    entry.input_tokens_details = TokenDetails(
                        cached_tokens=metrics.prompt_tokens_details.get("cached_tokens", 0),
                    )
                    if "prompt_tokens_details" not in self.metrics:
                        self.metrics["prompt_tokens_details"] = {}
                    for k, v in metrics.prompt_tokens_details.items():
                        self.metrics["prompt_tokens_details"][k] = (
                            self.metrics["prompt_tokens_details"].get(k, 0) + v
                        )
            # Parse completion_tokens_details
            if response_usage.completion_tokens_details is not None:
                if isinstance(response_usage.completion_tokens_details, dict):
                    metrics.completion_tokens_details = response_usage.completion_tokens_details
                elif isinstance(response_usage.completion_tokens_details, BaseModel):
                    metrics.completion_tokens_details = response_usage.completion_tokens_details.model_dump(
                        exclude_none=True
                    )
                assistant_message.metrics["completion_tokens_details"] = metrics.completion_tokens_details
                if metrics.completion_tokens_details is not None:
                    entry.output_tokens_details = TokenDetails(
                        reasoning_tokens=metrics.completion_tokens_details.get("reasoning_tokens", 0),
                    )
                    if "completion_tokens_details" not in self.metrics:
                        self.metrics["completion_tokens_details"] = {}
                    for k, v in metrics.completion_tokens_details.items():
                        self.metrics["completion_tokens_details"][k] = (
                            self.metrics["completion_tokens_details"].get(k, 0) + v
                        )
            self.usage.add(entry)

    def _create_assistant_message(
            self,
            response_message: ChatCompletionOutputMessage,
            metrics: Metrics,
            response_usage: Optional[ChatCompletionOutputUsage],
    ) -> Message:
        """Create an assistant message from the response."""
        assistant_message = Message(
            role=response_message.role or "assistant",
            content=response_message.content,
        )
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            assistant_message.tool_calls = [t.model_dump() for t in response_message.tool_calls]

        return assistant_message

    async def response(self, messages: List[Message]) -> ModelResponse:
        """Generate a response from HuggingFace Hub (async-only)."""
        self.sanitize_messages(messages)
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()

        metrics.response_timer.start()
        response: Union[ChatCompletionOutput] = await self.invoke(messages=messages)
        metrics.response_timer.stop()

        response_message: ChatCompletionOutputMessage = response.choices[0].message
        response_usage: Optional[ChatCompletionOutputUsage] = response.usage

        # Parse structured outputs
        try:
            if (
                    self.response_format is not None
                    and self.structured_outputs
                    and issubclass(self.response_format, BaseModel)
            ):
                parsed_object = response_message.parsed  # type: ignore
                if parsed_object is not None:
                    model_response.parsed = parsed_object
        except Exception as e:
            logger.warning(f"Error retrieving structured outputs: {e}")

        assistant_message = self._create_assistant_message(
            response_message=response_message, metrics=metrics, response_usage=response_usage
        )
        messages.append(assistant_message)
        assistant_message.log()
        metrics.log()

        # Handle tool calls
        if await self._handle_tool_calls(assistant_message, messages, model_response):
            response_after_tool_calls = await self.response(messages=messages)
            if response_after_tool_calls.content is not None:
                if model_response.content is None:
                    model_response.content = ""
                model_response.content += response_after_tool_calls.content
            return model_response

        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()

        return model_response

    def _update_stream_metrics(self, assistant_message: Message, metrics: Metrics):
        """Update the usage metrics for streaming response."""
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)

        if metrics.time_to_first_token is not None:
            assistant_message.metrics["time_to_first_token"] = metrics.time_to_first_token
            self.metrics.setdefault("time_to_first_token", []).append(metrics.time_to_first_token)

        if metrics.input_tokens is not None:
            assistant_message.metrics["input_tokens"] = metrics.input_tokens
            self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + metrics.input_tokens
        if metrics.output_tokens is not None:
            assistant_message.metrics["output_tokens"] = metrics.output_tokens
            self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + metrics.output_tokens
        if metrics.prompt_tokens is not None:
            assistant_message.metrics["prompt_tokens"] = metrics.prompt_tokens
            self.metrics["prompt_tokens"] = self.metrics.get("prompt_tokens", 0) + metrics.prompt_tokens
        if metrics.completion_tokens is not None:
            assistant_message.metrics["completion_tokens"] = metrics.completion_tokens
            self.metrics["completion_tokens"] = self.metrics.get("completion_tokens", 0) + metrics.completion_tokens
        if metrics.total_tokens is not None:
            assistant_message.metrics["total_tokens"] = metrics.total_tokens
            self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + metrics.total_tokens
        if metrics.prompt_tokens_details is not None:
            assistant_message.metrics["prompt_tokens_details"] = metrics.prompt_tokens_details
            if "prompt_tokens_details" not in self.metrics:
                self.metrics["prompt_tokens_details"] = {}
            for k, v in metrics.prompt_tokens_details.items():
                self.metrics["prompt_tokens_details"][k] = (
                    self.metrics["prompt_tokens_details"].get(k, 0) + v
                )
        if metrics.completion_tokens_details is not None:
            assistant_message.metrics["completion_tokens_details"] = metrics.completion_tokens_details
            if "completion_tokens_details" not in self.metrics:
                self.metrics["completion_tokens_details"] = {}
            for k, v in metrics.completion_tokens_details.items():
                self.metrics["completion_tokens_details"][k] = (
                    self.metrics["completion_tokens_details"].get(k, 0) + v
                )

        # Build structured RequestUsage entry
        from agentica.model.usage import RequestUsage, TokenDetails
        entry = RequestUsage(
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            response_time=metrics.response_timer.elapsed,
        )
        if metrics.prompt_tokens_details:
            entry.input_tokens_details = TokenDetails(
                cached_tokens=metrics.prompt_tokens_details.get("cached_tokens", 0),
            )
        if metrics.completion_tokens_details:
            entry.output_tokens_details = TokenDetails(
                reasoning_tokens=metrics.completion_tokens_details.get("reasoning_tokens", 0),
            )
        self.usage.add(entry)

    async def _handle_stream_tool_calls(
            self,
            assistant_message: Message,
            messages: List[Message],
    ) -> AsyncIterator[ModelResponse]:
        """Handle tool calls for response stream (async-only)."""
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            tool_role: str = "tool"
            function_calls_to_run: List[FunctionCall] = []
            function_call_results: List[Message] = []
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(
                        Message(
                            role=tool_role,
                            tool_call_id=_tool_call_id,
                            content="Could not find function to call.",
                        )
                    )
                    continue
                if _function_call.error is not None:
                    messages.append(
                        Message(
                            role=tool_role,
                            tool_call_id=_tool_call_id,
                            content=_function_call.error,
                        )
                    )
                    continue
                function_calls_to_run.append(_function_call)

            async for intermediate_model_response in self.run_function_calls(
                    function_calls=function_calls_to_run, function_call_results=function_call_results,
                    tool_role=tool_role
            ):
                yield intermediate_model_response

            if len(function_call_results) > 0:
                messages.extend(function_call_results)

    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """Generate a streaming response from HuggingFace Hub (async-only)."""
        self.sanitize_messages(messages)
        self._log_messages(messages)
        stream_data: StreamData = StreamData()
        metrics: Metrics = Metrics()

        metrics.response_timer.start()
        async for response in self.invoke_stream(messages=messages):
            if len(response.choices) > 0:
                metrics.completion_tokens += 1
                if metrics.completion_tokens == 1:
                    metrics.time_to_first_token = metrics.response_timer.elapsed

                response_delta: ChatCompletionStreamOutputDelta = response.choices[0].delta
                response_content = response_delta.content
                response_tool_calls = response_delta.tool_calls

                if response_content is not None:
                    stream_data.response_content += response_content
                    yield ModelResponse(content=response_content)

                if response_tool_calls is not None:
                    if stream_data.response_tool_calls is None:
                        stream_data.response_tool_calls = []
                    stream_data.response_tool_calls.extend(response_tool_calls)
        metrics.response_timer.stop()

        assistant_message = Message(role="assistant")
        if stream_data.response_content != "":
            assistant_message.content = stream_data.response_content

        if stream_data.response_tool_calls is not None:
            _tool_calls = self._build_tool_calls(stream_data.response_tool_calls)
            if len(_tool_calls) > 0:
                assistant_message.tool_calls = _tool_calls

        self._update_stream_metrics(assistant_message=assistant_message, metrics=metrics)

        messages.append(assistant_message)
        assistant_message.log()
        metrics.log()

        # Handle tool calls
        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            async for model_response in self._handle_stream_tool_calls(assistant_message, messages):
                yield model_response
            async for model_response in self.response_stream(messages=messages):
                yield model_response

    def _build_tool_calls(self, tool_calls_data: List[Any]) -> List[Dict[str, Any]]:
        """Build tool calls from tool call data."""
        tool_calls: List[Dict[str, Any]] = []
        for _tool_call in tool_calls_data:
            _index = _tool_call.index
            _tool_call_id = _tool_call.id
            _tool_call_type = _tool_call.type
            _function_name = _tool_call.function.name if _tool_call.function else None
            _function_arguments = _tool_call.function.arguments if _tool_call.function else None

            if len(tool_calls) <= _index:
                tool_calls.extend([{}] * (_index - len(tool_calls) + 1))
            tool_call_entry = tool_calls[_index]
            if not tool_call_entry:
                tool_call_entry["id"] = _tool_call_id
                tool_call_entry["type"] = _tool_call_type
                tool_call_entry["function"] = {
                    "name": _function_name or "",
                    "arguments": _function_arguments or "",
                }
            else:
                if _function_name:
                    tool_call_entry["function"]["name"] += _function_name
                if _function_arguments:
                    tool_call_entry["function"]["arguments"] += _function_arguments
                if _tool_call_id:
                    tool_call_entry["id"] = _tool_call_id
                if _tool_call_type:
                    tool_call_entry["type"] = _tool_call_type
        return tool_calls
