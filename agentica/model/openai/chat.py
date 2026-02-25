from os import getenv
from dataclasses import dataclass, field
from typing import Optional, List, AsyncIterator, Dict, Any, Union, Literal, override

import httpx
from enum import Enum, EnumMeta
from pydantic import BaseModel, Field

from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.metrics import Metrics, StreamData
from agentica.model.response import ModelResponse, ModelResponseEvent
from agentica.model.usage import RequestUsage, TokenDetails
from agentica.tools.base import FunctionCall, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer
from agentica.utils.langfuse_integration import is_langfuse_available, build_langfuse_metadata, get_langfuse_openai_client

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


# Request parameter names shared between request_kwargs and to_dict
_OPENAI_REQUEST_PARAMS = [
    "store", "reasoning_effort", "verbosity", "frequency_penalty", "logit_bias",
    "logprobs", "top_logprobs", "max_tokens", "max_completion_tokens",
    "modalities", "audio", "presence_penalty", "response_format", "seed",
    "stop", "temperature", "user", "top_p", "extra_headers", "extra_body",
    "extra_query", "metadata",
]


@dataclass
class OpenAIChat(Model):
    """
    A class for interacting with OpenAI models.

    For more information, see: https://platform.openai.com/docs/api-reference/chat/create
    """

    id: str = "gpt-4o"  # model name
    name: str = "OpenAIChat"
    provider: str = "OpenAI"
    context_window: int = 128000
    max_output_tokens: int = 16384

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
    modalities: Optional[List[str]] = None # "text" and/or "audio"
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

    # Client parameters
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[Any] = None
    default_query: Optional[Any] = None
    http_client: Optional[httpx.Client] = None
    client_params: Optional[Dict[str, Any]] = None

    # OpenAI client (async-only)
    client: Optional[AsyncOpenAIClient] = None

    # Internal parameters. Not used for API requests
    # Whether to use the structured outputs with this Model.
    structured_outputs: bool = False
    # Whether the Model supports structured outputs.
    supports_structured_outputs: bool = True
    # Langfuse tags for tracing (additional tags beyond what Agent provides)
    langfuse_tags: Optional[List[str]] = None
    # Langfuse trace name (defaults to model id)
    langfuse_trace_name: Optional[str] = None

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
        if self.client:
            return self.client

        client_params: Dict[str, Any] = self.get_client_params()
        if self.http_client:
            client_params["http_client"] = self.http_client
        else:
            client_params["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )

        # Try to use Langfuse-wrapped client if available
        _, LangfuseAsyncOpenAI = get_langfuse_openai_client()
        if LangfuseAsyncOpenAI is not None:
            self.client = LangfuseAsyncOpenAI(**client_params)
        else:
            self.client = AsyncOpenAIClient(**client_params)
        return self.client

    def _collect_request_params(self) -> Dict[str, Any]:
        """Collect non-None request parameters (shared by request_kwargs and to_dict)."""
        params: Dict[str, Any] = {}
        for attr_name in _OPENAI_REQUEST_PARAMS:
            val = getattr(self, attr_name, None)
            if val is not None:
                params[attr_name] = val
        return params

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Returns keyword arguments for API requests."""
        request_params = self._collect_request_params()
        if self.tools is not None:
            tools_for_api = self.get_tools_for_api()
            if tools_for_api:
                request_params["tools"] = tools_for_api
                request_params["tool_choice"] = self.tool_choice if self.tool_choice is not None else "auto"
        if self.request_params is not None:
            request_params.update(self.request_params)
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
                model_dict["tool_choice"] = self.tool_choice if self.tool_choice is not None else "auto"
        cleaned_dict = {k: v for k, v in model_dict.items() if v is not None}
        return cleaned_dict

    def format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message into the format expected by OpenAI."""
        if message.role == "user":
            if message.images is not None:
                message = self.add_images_to_message(message=message, images=message.images)

        if message.audio is not None:
            message = self.add_audio_to_message(message=message, audio=message.audio)

        return message.to_dict()

    def _get_langfuse_extra_params(self) -> Dict[str, Any]:
        """Get extra parameters for Langfuse tracing."""
        if not is_langfuse_available():
            return {}

        extra_params: Dict[str, Any] = {}

        if self.langfuse_trace_name:
            trace_name = self.langfuse_trace_name
        elif self.agent_name:
            trace_name = f"{self.agent_name}-{self.id}"
        else:
            trace_name = f"{self.name}-{self.id}"
        extra_params["name"] = trace_name

        metadata = build_langfuse_metadata(
            user_id=self.user_id,
            session_id=self.session_id,
            tags=self.langfuse_tags,
        )
        if metadata:
            extra_params["metadata"] = metadata

        return extra_params

    @override
    async def invoke(self, messages: List[Message]) -> Union[ChatCompletion, ParsedChatCompletion]:
        """Send a chat completion request to the OpenAI API (async-only)."""
        langfuse_params = self._get_langfuse_extra_params()

        if self.response_format is not None and self.structured_outputs:
            try:
                if isinstance(self.response_format, type) and issubclass(self.response_format, BaseModel):
                    return await self.get_client().beta.chat.completions.parse(
                        model=self.id,
                        messages=[self.format_message(m) for m in messages],  # type: ignore
                        **self.request_kwargs,
                        **langfuse_params,
                    )
                else:
                    raise ValueError("response_format must be a subclass of BaseModel if structured_outputs=True")
            except Exception as e:
                logger.error(f"Error from OpenAI API structured outputs: {e}")
                raise

        return await self.get_client().chat.completions.create(
            model=self.id,
            messages=[self.format_message(m) for m in messages],  # type: ignore
            **self.request_kwargs,
            **langfuse_params,
        )

    @override
    async def invoke_stream(self, messages: List[Message]) -> AsyncIterator[ChatCompletionChunk]:
        """Send a streaming chat completion request to the OpenAI API (async-only)."""
        langfuse_params = self._get_langfuse_extra_params()

        async_stream = await self.get_client().chat.completions.create(
            model=self.id,
            messages=[self.format_message(m) for m in messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **self.request_kwargs,
            **langfuse_params,
        )
        async for chunk in async_stream:  # type: ignore
            yield chunk

    # handle_tool_calls and handle_stream_tool_calls are inherited from Model base class.

    def create_assistant_message(
            self,
            response_message: ChatCompletionMessage,
            metrics: Metrics,
            response_usage: Optional[CompletionUsage],
    ) -> Message:
        """Create an assistant message from the response."""
        assistant_message = Message(
            role=response_message.role or "assistant",
            content=response_message.content,
        )
        if response_message.tool_calls is not None and len(response_message.tool_calls) > 0:
            try:
                assistant_message.tool_calls = [t.model_dump() for t in response_message.tool_calls]
            except Exception as e:
                logger.warning(f"Error processing tool calls: {e}")
        if hasattr(response_message, "audio") and response_message.audio is not None:
            try:
                assistant_message.audio = response_message.audio.model_dump()
            except Exception as e:
                logger.warning(f"Error processing audio: {e}")

        if hasattr(response_message, "reasoning_content") and response_message.reasoning_content is not None:
            assistant_message.reasoning_content = response_message.reasoning_content

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

        response_message: ChatCompletionMessage = response.choices[0].message
        response_usage: Optional[CompletionUsage] = response.usage

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

        assistant_message = self.create_assistant_message(
            response_message=response_message, metrics=metrics, response_usage=response_usage
        )

        messages.append(assistant_message)
        assistant_message.log()
        metrics.log()

        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        if assistant_message.audio is not None:
            model_response.audio = assistant_message.audio

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
            return await self.handle_post_tool_call_messages(messages=messages, model_response=model_response)
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

        metrics.response_timer.start()
        async for response in self.invoke_stream(messages=messages):
            if response.choices and len(response.choices) > 0:
                if metrics.completion_tokens is None:
                    metrics.completion_tokens = 0

                metrics.completion_tokens += 1
                if metrics.completion_tokens == 1:
                    metrics.time_to_first_token = metrics.response_timer.elapsed

                response_delta: ChoiceDelta = response.choices[0].delta

                if hasattr(response_delta, "reasoning_content") and response_delta.reasoning_content:
                    stream_data.response_reasoning_content += response_delta.reasoning_content
                    yield ModelResponse(reasoning_content=response_delta.reasoning_content)

                if hasattr(response_delta, "content") and response_delta.content:
                    stream_data.response_content += response_delta.content
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
            assistant_message.content = stream_data.response_content

        if stream_data.response_reasoning_content:
            assistant_message.reasoning_content = stream_data.response_reasoning_content

        if stream_data.response_audio:
            assistant_message.audio = stream_data.response_audio

        if stream_data.response_tool_calls:
            _tool_calls = self.build_tool_calls(stream_data.response_tool_calls)
            if len(_tool_calls) > 0:
                assistant_message.tool_calls = _tool_calls

        self.update_stream_metrics(assistant_message=assistant_message, metrics=metrics)

        messages.append(assistant_message)
        assistant_message.log()
        metrics.log()

        if assistant_message.tool_calls is not None and len(assistant_message.tool_calls) > 0 and self.run_tools:
            tool_role = "tool"
            async for tool_call_response in self.handle_stream_tool_calls(
                    assistant_message=assistant_message, messages=messages, tool_role=tool_role
            ):
                yield tool_call_response
            async for post_tool_call_response in self.handle_post_tool_call_messages_stream(messages=messages):
                yield post_tool_call_response

    def build_tool_calls(self, tool_calls_data: List[ChoiceDeltaToolCall]) -> List[Dict[str, Any]]:
        """Build tool calls from streaming tool call data."""
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
