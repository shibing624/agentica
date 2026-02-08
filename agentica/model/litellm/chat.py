# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: LiteLLM Model Provider

LiteLLM provides a unified interface to call 100+ LLM APIs using the OpenAI format.
Supported providers: OpenAI, Anthropic, Azure, Huggingface, Cohere, Together, Replicate,
Ollama, Bedrock, Vertex AI, and many more.

Usage:
    from agentica.model.litellm import LiteLLM
    
    # Use with provider prefix
    model = LiteLLM(id="openai/gpt-4o")
    model = LiteLLM(id="anthropic/claude-3-opus-20240229")
    model = LiteLLM(id="azure/gpt-4")
    model = LiteLLM(id="ollama/llama2")
    
    # Or use the litellm/ prefix for auto-detection in Agent
    agent = Agent(model="litellm/openai/gpt-4o")

Reference: https://docs.litellm.ai/docs/providers
"""
from os import getenv
from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Dict, Any, Union

from agentica.model.base import Model
from agentica.model.message import Message
from agentica.model.response import ModelResponse
from agentica.tools.base import FunctionCall, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer


@dataclass
class Metrics:
    """Metrics for tracking LLM response performance."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    time_to_first_token: Optional[float] = None
    response_timer: Timer = field(default_factory=Timer)

    def log(self):
        """Log metrics with null checks and safe calculations."""
        if self.time_to_first_token is not None:
            logger.debug(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        elapsed = self.response_timer.elapsed or 0
        logger.debug(f"* Time to generate response:   {elapsed:.4f}s")
        output_tokens = self.output_tokens or self.completion_tokens or 0
        if elapsed > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / elapsed
            logger.debug(f"* Tokens per second:           {tokens_per_second:.4f} tokens/s")
        input_tokens = self.input_tokens or self.prompt_tokens or 0
        logger.debug(f"* Input tokens:                {input_tokens}")
        logger.debug(f"* Output tokens:               {output_tokens}")
        total = self.total_tokens or (input_tokens + output_tokens) or 0
        logger.debug(f"* Total tokens:                {total}")


@dataclass
class StreamData:
    """Data accumulated during streaming response."""
    response_content: str = ""
    response_reasoning_content: str = ""
    response_tool_calls: Optional[List[Any]] = None


class LiteLLM(Model):
    """
    LiteLLM Model Provider - Unified interface to 100+ LLM APIs.
    
    LiteLLM allows you to call OpenAI, Anthropic, Azure, Huggingface, Ollama,
    Replicate, Together, and many more providers using a single interface.
    
    Attributes:
        id (str): Model identifier with provider prefix (e.g., "openai/gpt-4o", "anthropic/claude-3-opus-20240229")
        name (str): Display name for this model instance
        provider (str): Provider name (auto-detected from id)
        api_key (str): API key for the provider (auto-detected from environment)
        base_url (str): Custom API base URL (optional)
        
    Example:
        ```python
        from agentica.model.litellm import LiteLLM
        from agentica import Agent
        
        # Direct usage
        model = LiteLLM(id="openai/gpt-4o")
        
        # With Agent
        agent = Agent(
            model=LiteLLM(id="anthropic/claude-3-opus-20240229"),
            instructions="You are a helpful assistant."
        )
        ```
    
    Supported Providers:
        - openai/: OpenAI models (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
        - anthropic/: Anthropic Claude models
        - azure/: Azure OpenAI models
        - ollama/: Local Ollama models
        - huggingface/: Huggingface models
        - together_ai/: Together AI models
        - bedrock/: AWS Bedrock models
        - vertex_ai/: Google Vertex AI models
        - cohere/: Cohere models
        - replicate/: Replicate models
        - zai/: ZhipuAI models
        - and many more...
        
    Reference: https://docs.litellm.ai/docs/providers
    """
    
    id: str = "openai/gpt-4o"
    name: str = "LiteLLM-gpt-4o"
    provider: str = "openai"
    
    # API configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    api_version: Optional[str] = None  # For Azure
    # Request parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    
    # LiteLLM specific parameters
    timeout: Optional[float] = None
    num_retries: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Thinking/Reasoning parameters for DeepSeek, Claude, etc.
    # thinking: {"type": "enabled"} or {"type": "disabled"}
    thinking: Optional[Dict[str, Any]] = None
    # reasoning_effort: "low", "medium", "high" (maps to thinking enabled)
    reasoning_effort: Optional[str] = None
    
    # Additional request parameters
    request_params: Optional[Dict[str, Any]] = None
    
    # Internal state
    structured_outputs: bool = False
    supports_structured_outputs: bool = True
    
    def __post_init__(self):
        """Validate LiteLLM is installed."""
        try:
            from litellm import completion, acompletion
        except ImportError:
            raise ImportError(
                "LiteLLM is not installed. Please install it with: pip install litellm"
            )
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from instance or environment based on provider."""
        if self.api_key:
            return self.api_key
            
        # Auto-detect API key based on provider prefix
        provider = self.id.split("/")[0].lower() if "/" in self.id else ""
        
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "together_ai": "TOGETHER_API_KEY",
            "together": "TOGETHER_API_KEY",
            "replicate": "REPLICATE_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
            "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "zai": "ZAI_API_KEY",  # ZhipuAI
            "zhipuai": "ZHIPUAI_API_KEY",
        }
        
        env_var = env_key_map.get(provider)
        if env_var:
            return getenv(env_var)
        return None
    
    @property
    def request_kwargs(self) -> Dict[str, Any]:
        """Build request parameters for LiteLLM."""
        params: Dict[str, Any] = {}
        
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            params["stop"] = self.stop
        if self.seed is not None:
            params["seed"] = self.seed
        if self.timeout is not None:
            params["timeout"] = self.timeout
        if self.num_retries is not None:
            params["num_retries"] = self.num_retries
        if self.metadata is not None:
            params["metadata"] = self.metadata
        if self.base_url is not None:
            params["api_base"] = self.base_url  # litellm uses 'api_base' internally
        if self.api_version is not None:
            params["api_version"] = self.api_version
        
        # Thinking/Reasoning parameters
        if self.thinking is not None:
            params["thinking"] = self.thinking
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
            
        # Add tools if present
        if self.tools is not None:
            tools_for_api = self.get_tools_for_api()
            if tools_for_api:  # Only add tools if list is not empty
                params["tools"] = tools_for_api
                params["tool_choice"] = self.tool_choice or "auto"
            
        # Merge additional request params
        if self.request_params:
            params.update(self.request_params)
            
        return params
    
    def get_tools_for_api(self) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to API format."""
        if self.tools is None:
            return None
        return [tool if isinstance(tool, dict) else tool.model_dump() for tool in self.tools]
    
    def format_message(self, message: Message) -> Dict[str, Any]:
        """Format a message for LiteLLM API."""
        msg_dict: Dict[str, Any] = {"role": message.role}
        
        if message.content is not None:
            msg_dict["content"] = message.content
        if message.name is not None:
            msg_dict["name"] = message.name
        if message.tool_call_id is not None:
            msg_dict["tool_call_id"] = message.tool_call_id
        if message.tool_calls is not None:
            msg_dict["tool_calls"] = message.tool_calls
            
        return msg_dict
    
    def invoke(self, messages: List[Message]) -> Any:
        """
        Send a completion request to LiteLLM.
        
        Args:
            messages: List of messages to send
            
        Returns:
            LiteLLM response object
        """
        from litellm import completion
        api_key = self._get_api_key()
        
        response = completion(
            model=self.id,
            messages=[self.format_message(m) for m in messages],
            api_key=api_key,
            **self.request_kwargs,
        )
        return response
    
    async def ainvoke(self, messages: List[Message]) -> Any:
        """
        Send an async completion request to LiteLLM.
        
        Args:
            messages: List of messages to send
            
        Returns:
            LiteLLM response object
        """
        from litellm import acompletion
        api_key = self._get_api_key()
        
        response = await acompletion(
            model=self.id,
            messages=[self.format_message(m) for m in messages],
            api_key=api_key,
            **self.request_kwargs,
        )
        return response
    
    def invoke_stream(self, messages: List[Message]) -> Iterator[Any]:
        """
        Send a streaming completion request to LiteLLM.
        
        Args:
            messages: List of messages to send
            
        Yields:
            Streaming response chunks
        """
        from litellm import completion
        api_key = self._get_api_key()
        
        response = completion(
            model=self.id,
            messages=[self.format_message(m) for m in messages],
            api_key=api_key,
            stream=True,
            **self.request_kwargs,
        )
        yield from response
    
    async def ainvoke_stream(self, messages: List[Message]) -> Any:
        """
        Send an async streaming completion request to LiteLLM.
        
        Args:
            messages: List of messages to send
            
        Returns:
            Async iterator of streaming response chunks
        """
        from litellm import acompletion
        api_key = self._get_api_key()
        
        response = await acompletion(
            model=self.id,
            messages=[self.format_message(m) for m in messages],
            api_key=api_key,
            stream=True,
            **self.request_kwargs,
        )
        async for chunk in response:
            yield chunk
    
    def _update_usage_metrics(
            self, assistant_message: Message, metrics: Metrics, response_usage: Any
    ) -> None:
        """Update usage metrics from response."""
        assistant_message.metrics["time"] = metrics.response_timer.elapsed
        self.metrics.setdefault("response_times", []).append(metrics.response_timer.elapsed)
        
        if response_usage:
            prompt_tokens = getattr(response_usage, 'prompt_tokens', 0)
            completion_tokens = getattr(response_usage, 'completion_tokens', 0)
            total_tokens = getattr(response_usage, 'total_tokens', 0)
            
            if prompt_tokens:
                metrics.input_tokens = prompt_tokens
                metrics.prompt_tokens = prompt_tokens
                assistant_message.metrics["input_tokens"] = prompt_tokens
                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + prompt_tokens
            if completion_tokens:
                metrics.output_tokens = completion_tokens
                metrics.completion_tokens = completion_tokens
                assistant_message.metrics["output_tokens"] = completion_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + completion_tokens
            if total_tokens:
                metrics.total_tokens = total_tokens
                assistant_message.metrics["total_tokens"] = total_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + total_tokens
    
    def _create_assistant_message(
            self,
            response_message: Any,
            metrics: Metrics,
            response_usage: Any,
    ) -> Message:
        """Create assistant message from response."""
        content = getattr(response_message, 'content', None) or ""
        reasoning_content = getattr(response_message, 'reasoning_content', None)
        
        assistant_message = Message(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
        )
        
        # Handle tool calls
        tool_calls = getattr(response_message, 'tool_calls', None)
        if tool_calls:
            assistant_message.tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in tool_calls
            ]
        
        # Update metrics
        self._update_usage_metrics(assistant_message, metrics, response_usage)
        return assistant_message
    
    def _handle_tool_calls(
            self,
            assistant_message: Message,
            messages: List[Message],
            model_response: ModelResponse,
            tool_role: str = "tool",
    ) -> Optional[ModelResponse]:
        """Handle tool calls in assistant message."""
        if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0 and self.run_tools:
            if model_response.content is None:
                model_response.content = ""
            
            function_call_results: List[Message] = []
            function_calls_to_run: List[FunctionCall] = []
            
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(Message(
                        role=tool_role,
                        tool_call_id=_tool_call_id,
                        content="Could not find function to call.",
                    ))
                    continue
                if _function_call.error is not None:
                    messages.append(Message(
                        role=tool_role,
                        tool_call_id=_tool_call_id,
                        content=_function_call.error,
                    ))
                    continue
                function_calls_to_run.append(_function_call)
            
            if self.show_tool_calls:
                model_response.content += "\nRunning:"
                for _f in function_calls_to_run:
                    model_response.content += f"\n - {_f.get_call_str()}"
                model_response.content += "\n\n"
            
            for _ in self.run_function_calls(
                    function_calls=function_calls_to_run,
                    function_call_results=function_call_results,
                    tool_role=tool_role
            ):
                pass
            
            if len(function_call_results) > 0:
                messages.extend(function_call_results)
            
            return model_response
        return None
    
    def response(self, messages: List[Message]) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: List of messages to send
            
        Returns:
            ModelResponse with the generated content
        """
        self.sanitize_messages(messages)
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()
        
        # Generate response
        metrics.response_timer.start()
        response = self.invoke(messages)
        metrics.response_timer.stop()
        
        # Parse response
        response_message = response.choices[0].message
        response_usage = getattr(response, 'usage', None)
        
        # Create assistant message
        assistant_message = self._create_assistant_message(
            response_message=response_message,
            metrics=metrics,
            response_usage=response_usage
        )
        
        # Add assistant message to messages
        messages.append(assistant_message)
        
        # Log response and metrics
        assistant_message.log()
        metrics.log()
        
        # Update model response
        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        
        # Handle tool calls
        tool_role = "tool"
        if self._handle_tool_calls(
                assistant_message=assistant_message,
                messages=messages,
                model_response=model_response,
                tool_role=tool_role,
        ) is not None:
            return self.handle_post_tool_call_messages(messages=messages, model_response=model_response)
        
        return model_response
    
    async def aresponse(self, messages: List[Message]) -> ModelResponse:
        """
        Generate an async response from the model.
        
        Args:
            messages: List of messages to send
            
        Returns:
            ModelResponse with the generated content
        """
        self.sanitize_messages(messages)
        self._log_messages(messages)
        model_response = ModelResponse()
        metrics = Metrics()
        
        # Generate response
        metrics.response_timer.start()
        response = await self.ainvoke(messages)
        metrics.response_timer.stop()
        
        # Parse response
        response_message = response.choices[0].message
        response_usage = getattr(response, 'usage', None)
        
        # Create assistant message
        assistant_message = self._create_assistant_message(
            response_message=response_message,
            metrics=metrics,
            response_usage=response_usage
        )
        
        # Add assistant message to messages
        messages.append(assistant_message)
        
        # Log response and metrics
        assistant_message.log()
        metrics.log()
        
        # Update model response
        if assistant_message.content is not None:
            model_response.content = assistant_message.get_content_string()
        if assistant_message.reasoning_content is not None:
            model_response.reasoning_content = assistant_message.reasoning_content
        
        # Handle tool calls
        tool_role = "tool"
        if self._handle_tool_calls(
                assistant_message=assistant_message,
                messages=messages,
                model_response=model_response,
                tool_role=tool_role,
        ) is not None:
            return await self.ahandle_post_tool_call_messages(messages=messages, model_response=model_response)
        
        return model_response
    
    def _update_stream_metrics(self, assistant_message: Message, metrics: Metrics):
        """Update metrics for streaming response."""
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
        if metrics.total_tokens:
            assistant_message.metrics["total_tokens"] = metrics.total_tokens
            self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + metrics.total_tokens
    
    def _handle_stream_tool_calls(
            self,
            assistant_message: Message,
            messages: List[Message],
            tool_role: str = "tool",
    ) -> Iterator[ModelResponse]:
        """Handle tool calls for streaming response."""
        if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0 and self.run_tools:
            function_calls_to_run: List[FunctionCall] = []
            function_call_results: List[Message] = []
            
            for tool_call in assistant_message.tool_calls:
                _tool_call_id = tool_call.get("id")
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(Message(
                        role=tool_role,
                        tool_call_id=_tool_call_id,
                        content="Could not find function to call.",
                    ))
                    continue
                if _function_call.error is not None:
                    messages.append(Message(
                        role=tool_role,
                        tool_call_id=_tool_call_id,
                        content=_function_call.error,
                    ))
                    continue
                function_calls_to_run.append(_function_call)
            
            if self.show_tool_calls:
                yield ModelResponse(content="\nRunning:")
                for _f in function_calls_to_run:
                    yield ModelResponse(content=f"\n - {_f.get_call_str()}")
                yield ModelResponse(content="\n\n")
            
            for function_call_response in self.run_function_calls(
                    function_calls=function_calls_to_run,
                    function_call_results=function_call_results,
                    tool_role=tool_role
            ):
                yield function_call_response
            
            if len(function_call_results) > 0:
                messages.extend(function_call_results)
    
    def _merge_tool_call_deltas(self, deltas: List[Any]) -> List[Dict[str, Any]]:
        """Merge streaming tool call deltas into complete tool calls."""
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        
        for delta in deltas:
            idx = getattr(delta, "index", 0)
            
            if idx not in tool_calls_by_index:
                tool_calls_by_index[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                }
            
            tc = tool_calls_by_index[idx]
            
            if hasattr(delta, "id") and delta.id:
                tc["id"] = delta.id
            if hasattr(delta, "function"):
                if hasattr(delta.function, "name") and delta.function.name:
                    tc["function"]["name"] += delta.function.name
                if hasattr(delta.function, "arguments") and delta.function.arguments:
                    tc["function"]["arguments"] += delta.function.arguments
        
        return list(tool_calls_by_index.values())
    
    def response_stream(self, messages: List[Message]) -> Iterator[ModelResponse]:
        """
        Generate a streaming response from the model.
        
        Args:
            messages: List of messages to send
            
        Yields:
            ModelResponse chunks with incremental content
        """
        self.sanitize_messages(messages)
        self._log_messages(messages)
        
        stream_data = StreamData()
        metrics = Metrics()
        
        metrics.response_timer.start()
        for chunk in self.invoke_stream(messages):
            if chunk.choices and len(chunk.choices) > 0:
                if metrics.completion_tokens == 0:
                    metrics.time_to_first_token = metrics.response_timer.elapsed
                metrics.completion_tokens += 1
                
                delta = chunk.choices[0].delta
                
                # Handle reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    stream_data.response_reasoning_content += delta.reasoning_content
                    yield ModelResponse(reasoning_content=delta.reasoning_content)
                
                # Handle content
                if hasattr(delta, "content") and delta.content:
                    stream_data.response_content += delta.content
                    yield ModelResponse(content=delta.content)
                
                # Handle tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    if stream_data.response_tool_calls is None:
                        stream_data.response_tool_calls = []
                    stream_data.response_tool_calls.extend(delta.tool_calls)
            
            # Handle usage in streaming
            if hasattr(chunk, "usage") and chunk.usage:
                metrics.input_tokens = getattr(chunk.usage, 'prompt_tokens', 0)
                metrics.output_tokens = getattr(chunk.usage, 'completion_tokens', 0)
                metrics.total_tokens = getattr(chunk.usage, 'total_tokens', 0)
        
        metrics.response_timer.stop()
        
        # Create assistant message
        assistant_message = Message(role="assistant")
        if stream_data.response_content:
            assistant_message.content = stream_data.response_content
        if stream_data.response_reasoning_content:
            assistant_message.reasoning_content = stream_data.response_reasoning_content
        if stream_data.response_tool_calls:
            assistant_message.tool_calls = self._merge_tool_call_deltas(stream_data.response_tool_calls)
        
        # Update metrics
        self._update_stream_metrics(assistant_message=assistant_message, metrics=metrics)
        
        # Add assistant message to messages
        messages.append(assistant_message)
        
        # Log response and metrics
        assistant_message.log()
        metrics.log()
        
        # Handle tool calls
        if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0 and self.run_tools:
            tool_role = "tool"
            yield from self._handle_stream_tool_calls(
                assistant_message=assistant_message,
                messages=messages,
                tool_role=tool_role
            )
            yield from self.handle_post_tool_call_messages_stream(messages=messages)
    
    async def aresponse_stream(self, messages: List[Message]) -> Any:
        """
        Generate an async streaming response from the model.
        
        Args:
            messages: List of messages to send
            
        Yields:
            ModelResponse chunks with incremental content
        """
        self.sanitize_messages(messages)
        self._log_messages(messages)
        
        stream_data = StreamData()
        metrics = Metrics()
        
        metrics.response_timer.start()
        async for chunk in self.ainvoke_stream(messages):
            if chunk.choices and len(chunk.choices) > 0:
                if metrics.completion_tokens == 0:
                    metrics.time_to_first_token = metrics.response_timer.elapsed
                metrics.completion_tokens += 1
                
                delta = chunk.choices[0].delta
                
                # Handle reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    stream_data.response_reasoning_content += delta.reasoning_content
                    yield ModelResponse(reasoning_content=delta.reasoning_content)
                
                # Handle content
                if hasattr(delta, "content") and delta.content:
                    stream_data.response_content += delta.content
                    yield ModelResponse(content=delta.content)
                
                # Handle tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    if stream_data.response_tool_calls is None:
                        stream_data.response_tool_calls = []
                    stream_data.response_tool_calls.extend(delta.tool_calls)
            
            # Handle usage in streaming
            if hasattr(chunk, "usage") and chunk.usage:
                metrics.input_tokens = getattr(chunk.usage, 'prompt_tokens', 0)
                metrics.output_tokens = getattr(chunk.usage, 'completion_tokens', 0)
                metrics.total_tokens = getattr(chunk.usage, 'total_tokens', 0)
        
        metrics.response_timer.stop()
        
        # Create assistant message
        assistant_message = Message(role="assistant")
        if stream_data.response_content:
            assistant_message.content = stream_data.response_content
        if stream_data.response_reasoning_content:
            assistant_message.reasoning_content = stream_data.response_reasoning_content
        if stream_data.response_tool_calls:
            assistant_message.tool_calls = self._merge_tool_call_deltas(stream_data.response_tool_calls)
        
        # Update metrics
        self._update_stream_metrics(assistant_message=assistant_message, metrics=metrics)
        
        # Add assistant message to messages
        messages.append(assistant_message)
        
        # Log response and metrics
        assistant_message.log()
        metrics.log()
        
        # Handle tool calls
        if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0 and self.run_tools:
            tool_role = "tool"
            for tool_call_response in self._handle_stream_tool_calls(
                    assistant_message=assistant_message,
                    messages=messages,
                    tool_role=tool_role
            ):
                yield tool_call_response
            async for post_tool_call_response in self.ahandle_post_tool_call_messages_stream(messages=messages):
                yield post_tool_call_response
