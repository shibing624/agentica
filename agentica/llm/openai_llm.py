# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
part of the code from https://github.com/phidatahq/phidata
"""

from os import getenv
from typing import Optional, List, Iterator, Dict, Any, Union, Tuple

import httpx
from openai import OpenAI as OpenAIClient, AsyncOpenAI as AsyncOpenAIClient
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall as ChatCompletionFunctionCall,
)
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion_usage import CompletionUsage

from agentica.config import FAST_LLM
from agentica.llm.base import LLM
from agentica.message import Message
from agentica.tools.base import FunctionCall, get_function_call, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer


class OpenAILLM(LLM):
    name: str = "OpenAILLM"
    model: str = FAST_LLM or "gpt-4o-mini"
    # -*- Request parameters
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Any] = None
    logprobs: Optional[bool] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    top_p: Optional[float] = None
    extra_headers: Optional[Any] = None
    extra_query: Optional[Any] = None
    request_params: Optional[Dict[str, Any]] = None
    # -*- Client parameters
    api_key: Optional[str] = getenv("OPENAI_API_KEY")
    organization: Optional[str] = None
    base_url: Optional[Union[str, httpx.URL]] = getenv("OPENAI_BASE_URL")
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    default_headers: Optional[Any] = None
    default_query: Optional[Any] = None
    http_client: Optional[httpx.Client] = None
    client_params: Optional[Dict[str, Any]] = None
    # -*- Provide the OpenAI client manually
    client: Optional[OpenAIClient] = None
    async_client: Optional[AsyncOpenAIClient] = None
    openai_client: Optional[OpenAIClient] = None

    def get_client(self) -> OpenAIClient:
        if self.client:
            return self.client

        if self.openai_client:
            return self.openai_client

        _client_params: Dict[str, Any] = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.organization:
            _client_params["organization"] = self.organization
        if self.base_url:
            _client_params["base_url"] = self.base_url
        if self.timeout:
            _client_params["timeout"] = self.timeout
        if self.max_retries:
            _client_params["max_retries"] = self.max_retries
        if self.default_headers:
            _client_params["default_headers"] = self.default_headers
        if self.default_query:
            _client_params["default_query"] = self.default_query
        if self.http_client:
            _client_params["http_client"] = self.http_client
        if self.client_params:
            _client_params.update(self.client_params)
        self.client = OpenAIClient(**_client_params)
        return self.client

    def get_async_client(self) -> AsyncOpenAIClient:
        if self.async_client:
            return self.async_client

        _client_params: Dict[str, Any] = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.organization:
            _client_params["organization"] = self.organization
        if self.base_url:
            _client_params["base_url"] = self.base_url
        if self.timeout:
            _client_params["timeout"] = self.timeout
        if self.max_retries:
            _client_params["max_retries"] = self.max_retries
        if self.default_headers:
            _client_params["default_headers"] = self.default_headers
        if self.default_query:
            _client_params["default_query"] = self.default_query
        if self.http_client:
            _client_params["http_client"] = self.http_client
        else:
            _client_params["http_client"] = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            )
        if self.client_params:
            _client_params.update(self.client_params)
        self.async_client = AsyncOpenAIClient(**_client_params)
        return self.async_client

    @property
    def api_kwargs(self) -> Dict[str, Any]:
        _request_params: Dict[str, Any] = {}
        if self.frequency_penalty:
            _request_params["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias:
            _request_params["logit_bias"] = self.logit_bias
        if self.logprobs:
            _request_params["logprobs"] = self.logprobs
        if self.max_tokens:
            _request_params["max_tokens"] = self.max_tokens
        if self.presence_penalty:
            _request_params["presence_penalty"] = self.presence_penalty
        if self.response_format:
            _request_params["response_format"] = self.response_format
        if self.seed:
            _request_params["seed"] = self.seed
        if self.stop:
            _request_params["stop"] = self.stop
        if self.temperature:
            _request_params["temperature"] = self.temperature
        if self.top_logprobs:
            _request_params["top_logprobs"] = self.top_logprobs
        if self.user:
            _request_params["user"] = self.user
        if self.top_p:
            _request_params["top_p"] = self.top_p
        if self.extra_headers:
            _request_params["extra_headers"] = self.extra_headers
        if self.extra_query:
            _request_params["extra_query"] = self.extra_query
        if self.tools:
            _request_params["tools"] = self.get_tools_for_api()
            if self.tool_choice is None:
                _request_params["tool_choice"] = "auto"
            else:
                _request_params["tool_choice"] = self.tool_choice
        if self.request_params:
            _request_params.update(self.request_params)
        return _request_params

    def to_dict(self) -> Dict[str, Any]:
        _dict = super().to_dict()
        if self.frequency_penalty:
            _dict["frequency_penalty"] = self.frequency_penalty
        if self.logit_bias:
            _dict["logit_bias"] = self.logit_bias
        if self.logprobs:
            _dict["logprobs"] = self.logprobs
        if self.max_tokens:
            _dict["max_tokens"] = self.max_tokens
        if self.presence_penalty:
            _dict["presence_penalty"] = self.presence_penalty
        if self.response_format:
            _dict["response_format"] = self.response_format
        if self.seed:
            _dict["seed"] = self.seed
        if self.stop:
            _dict["stop"] = self.stop
        if self.temperature:
            _dict["temperature"] = self.temperature
        if self.top_logprobs:
            _dict["top_logprobs"] = self.top_logprobs
        if self.user:
            _dict["user"] = self.user
        if self.top_p:
            _dict["top_p"] = self.top_p
        if self.extra_headers:
            _dict["extra_headers"] = self.extra_headers
        if self.extra_query:
            _dict["extra_query"] = self.extra_query
        if self.tools:
            _dict["tools"] = self.get_tools_for_api()
            if self.tool_choice is None:
                _dict["tool_choice"] = "auto"
            else:
                _dict["tool_choice"] = self.tool_choice
        return _dict

    def invoke(self, messages: List[Message]) -> ChatCompletion:
        return self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],  # type: ignore
            **self.api_kwargs,
        )

    async def ainvoke(self, messages: List[Message]) -> Any:
        return await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],  # type: ignore
            **self.api_kwargs,
        )

    def invoke_stream(self, messages: List[Message]) -> Iterator[ChatCompletionChunk]:
        yield from self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],  # type: ignore
            stream=True,
            stream_options={"include_usage": True},
            **self.api_kwargs,
        )  # type: ignore

    async def ainvoke_stream(self, messages: List[Message]) -> Any:
        async_stream = await self.get_async_client().chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],  # type: ignore
            stream=True,
            **self.api_kwargs,
        )
        async for chunk in async_stream:  # type: ignore
            yield chunk

    def run_function(self, function_call: Dict[str, Any]) -> Tuple[Message, Optional[FunctionCall]]:
        _function_name = function_call.get("name")
        _function_arguments_str = function_call.get("arguments")
        if _function_name is not None:

            # Get function call
            _function_call = get_function_call(
                name=_function_name,
                arguments=_function_arguments_str,
                functions=self.functions,
            )
            logger.debug(
                f"run_function: {_function_name}, \n_function_name: {_function_name}, \narguments: {_function_arguments_str}"
                f"\nfunctions: {self.functions}")

            if _function_call is None:
                return Message(role="function", content="Could not find function to call."), None
            if _function_call.error is not None:
                return Message(role="function", tool_call_error=True, content=_function_call.error), _function_call

            if self.function_call_stack is None:
                self.function_call_stack = []

            # -*- Check function call limit
            if len(self.function_call_stack) > self.function_call_limit:
                self.tool_choice = "none"
                return Message(
                    role="function",
                    content=f"Function call limit ({self.function_call_limit}) exceeded.",
                ), _function_call

            # -*- Run function call
            self.function_call_stack.append(_function_call)
            _function_call_timer = Timer()
            _function_call_timer.start()
            function_call_success = _function_call.execute()
            _function_call_timer.stop()
            _function_call_message = Message(
                role="function",
                name=_function_call.function.name,
                content=_function_call.result if function_call_success else _function_call.error,
                tool_call_error=not function_call_success,
                metrics={"time": _function_call_timer.elapsed},
            )
            if "function_call_times" not in self.metrics:
                self.metrics["function_call_times"] = {}
            if _function_call.function.name not in self.metrics["function_call_times"]:
                self.metrics["function_call_times"][_function_call.function.name] = []
            self.metrics["function_call_times"][_function_call.function.name].append(_function_call_timer.elapsed)
            return _function_call_message, _function_call
        return Message(role="function", content="Function name is None."), None

    def response(self, messages: List[Message]) -> str:
        logger.debug("---------- OpenAI Response Start ----------")
        # -*- Log messages for debugging
        for m in messages:
            m.log()

        t = Timer()
        t.start()
        response: ChatCompletion = self.invoke(messages=messages)
        t.stop()
        logger.debug(f"Time to generate response: {t.elapsed:.4f}s")

        # -*- Parse response
        response_message: ChatCompletionMessage = response.choices[0].message
        response_role = response_message.role
        response_content: Optional[str] = response_message.content
        response_function_call: Optional[ChatCompletionFunctionCall] = response_message.function_call
        response_tool_calls: Optional[List[ChatCompletionMessageToolCall]] = response_message.tool_calls

        # -*- Create assistant message
        assistant_message = Message(
            role=response_role or "assistant",
            content=response_content,
        )
        if response_function_call is not None:
            assistant_message.function_call = response_function_call.model_dump()
        if response_tool_calls is not None:
            assistant_message.tool_calls = [t.model_dump() for t in response_tool_calls]

        # -*- Update usage metrics
        # Add response time to metrics
        assistant_message.metrics["time"] = t.elapsed
        if "response_times" not in self.metrics:
            self.metrics["response_times"] = []
        self.metrics["response_times"].append(t.elapsed)

        # Add token usage to metrics
        response_usage: Optional[CompletionUsage] = response.usage
        if response_usage:
            prompt_tokens = response_usage.prompt_tokens
            completion_tokens = response_usage.completion_tokens
            total_tokens = response_usage.total_tokens

            if prompt_tokens is not None:
                assistant_message.metrics["prompt_tokens"] = prompt_tokens
                self.metrics["prompt_tokens"] = self.metrics.get("prompt_tokens", 0) + prompt_tokens
                assistant_message.metrics["input_tokens"] = prompt_tokens
                self.metrics["input_tokens"] = self.metrics.get("input_tokens", 0) + prompt_tokens
            if completion_tokens is not None:
                assistant_message.metrics["completion_tokens"] = completion_tokens
                self.metrics["completion_tokens"] = self.metrics.get("completion_tokens", 0) + completion_tokens
                assistant_message.metrics["output_tokens"] = completion_tokens
                self.metrics["output_tokens"] = self.metrics.get("output_tokens", 0) + completion_tokens
            if total_tokens is not None:
                assistant_message.metrics["total_tokens"] = total_tokens
                self.metrics["total_tokens"] = self.metrics.get("total_tokens", 0) + total_tokens

        # -*- Add assistant message to messages
        messages.append(assistant_message)
        assistant_message.log()

        # -*- Parse and run function call
        need_to_run_functions = assistant_message.function_call is not None or assistant_message.tool_calls is not None
        if need_to_run_functions and self.run_tools:
            if assistant_message.function_call is not None:
                function_call_message, function_call = self.run_function(function_call=assistant_message.function_call)
                messages.append(function_call_message)
                # -*- Get new response using result of function call
                final_response = ""
                if self.show_tool_calls and function_call is not None:
                    final_response += f"\n - Running: {function_call.get_call_str()}\n\n"
                final_response += self.response(messages=messages)
                return final_response
            elif assistant_message.tool_calls is not None:
                final_response = ""
                function_calls_to_run: List[FunctionCall] = []
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

                if self.show_tool_calls:
                    if len(function_calls_to_run) == 1:
                        final_response += f"\n - Running: {function_calls_to_run[0].get_call_str()}\n\n"
                    elif len(function_calls_to_run) > 1:
                        final_response += "\nRunning:"
                        for _f in function_calls_to_run:
                            final_response += f"\n - {_f.get_call_str()}"
                        final_response += "\n\n"

                function_call_results = self.run_function_calls(function_calls_to_run)
                if len(function_call_results) > 0:
                    messages.extend(function_call_results)
                # -*- Get new response using result of tool call
                final_response += self.response(messages=messages)
                return final_response
        logger.debug("---------- OpenAI Response End ----------")
        # -*- Return content if no function calls are present
        if assistant_message.content is not None:
            return assistant_message.get_content_string()
        return "Something went wrong, please try again."

    async def aresponse(self, messages: List[Message]) -> str:
        logger.debug("---------- OpenAI Async Response Start ----------")
        for m in messages:
            m.log()

        t = Timer()
        t.start()
        response: ChatCompletion = await self.ainvoke(messages=messages)
        t.stop()
        logger.debug(f"Time to generate response: {t.elapsed:.4f}s")

        # -*- Parse response
        response_message: ChatCompletionMessage = response.choices[0].message
        response_role = response_message.role
        response_content: Optional[str] = response_message.content
        response_function_call: Optional[ChatCompletionFunctionCall] = response_message.function_call
        response_tool_calls: Optional[List[ChatCompletionMessageToolCall]] = response_message.tool_calls

        # -*- Create assistant message
        assistant_message = Message(
            role=response_role or "assistant",
            content=response_content,
        )
        if response_function_call is not None:
            assistant_message.function_call = response_function_call.model_dump()
        if response_tool_calls is not None:
            assistant_message.tool_calls = [t.model_dump() for t in response_tool_calls]

        # -*- Update usage metrics
        # Add response time to metrics
        assistant_message.metrics["time"] = t.elapsed
        if "response_times" not in self.metrics:
            self.metrics["response_times"] = []
        self.metrics["response_times"].append(t.elapsed)

        # Add token usage to metrics
        response_usage: Optional[CompletionUsage] = response.usage
        prompt_tokens = response_usage.prompt_tokens if response_usage is not None else None
        if prompt_tokens is not None:
            assistant_message.metrics["prompt_tokens"] = prompt_tokens
            if "prompt_tokens" not in self.metrics:
                self.metrics["prompt_tokens"] = prompt_tokens
            else:
                self.metrics["prompt_tokens"] += prompt_tokens
        completion_tokens = response_usage.completion_tokens if response_usage is not None else None
        if completion_tokens is not None:
            assistant_message.metrics["completion_tokens"] = completion_tokens
            if "completion_tokens" not in self.metrics:
                self.metrics["completion_tokens"] = completion_tokens
            else:
                self.metrics["completion_tokens"] += completion_tokens
        total_tokens = response_usage.total_tokens if response_usage is not None else None
        if total_tokens is not None:
            assistant_message.metrics["total_tokens"] = total_tokens
            if "total_tokens" not in self.metrics:
                self.metrics["total_tokens"] = total_tokens
            else:
                self.metrics["total_tokens"] += total_tokens

        # -*- Add assistant message to messages
        messages.append(assistant_message)
        assistant_message.log()

        # -*- Parse and run function call
        need_to_run_functions = assistant_message.function_call is not None or assistant_message.tool_calls is not None
        if need_to_run_functions and self.run_tools:
            if assistant_message.function_call is not None:
                call_message, function_call = self.run_function(function_call=assistant_message.function_call)
                messages.append(call_message)
                # -*- Get new response using result of function call
                final_response = ""
                if self.show_tool_calls and function_call is not None:
                    final_response += f"\n - Running: {function_call.get_call_str()}\n\n"
                final_response += self.response(messages=messages)
                return final_response
            elif assistant_message.tool_calls is not None:
                final_response = ""
                function_calls_to_run: List[FunctionCall] = []
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

                if self.show_tool_calls:
                    if len(function_calls_to_run) == 1:
                        final_response += f"\n - Running: {function_calls_to_run[0].get_call_str()}\n\n"
                    elif len(function_calls_to_run) > 1:
                        final_response += "\nRunning:"
                        for _f in function_calls_to_run:
                            final_response += f"\n - {_f.get_call_str()}"
                        final_response += "\n\n"

                function_call_results = self.run_function_calls(function_calls_to_run)
                if len(function_call_results) > 0:
                    messages.extend(function_call_results)
                # -*- Get new response using result of tool call
                final_response += await self.aresponse(messages=messages)
                return final_response
        logger.debug("---------- OpenAI Async Response End ----------")
        # -*- Return content if no function calls are present
        if assistant_message.content is not None:
            return assistant_message.get_content_string()
        return "Something went wrong, please try again."

    def response_stream(self, messages: List[Message]) -> Iterator[str]:
        logger.debug("---------- OpenAI Response Start ----------")
        # -*- Log messages for debugging
        for m in messages:
            m.log()

        assistant_message_content = ""
        assistant_message_function_name = ""
        assistant_message_function_arguments_str = ""
        assistant_message_tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
        prompt_tokens = sum([self.calculate_tokens(i.get_content_string()) for i in messages if i])
        completion_tokens = 0
        response_prompt_tokens = 0
        response_completion_tokens = 0
        response_total_tokens = 0
        time_to_first_token = None
        t = Timer()
        t.start()
        for response in self.invoke_stream(messages=messages):
            response_content: Optional[str] = None
            response_function_call: Optional[ChoiceDeltaFunctionCall] = None
            response_tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
            if len(response.choices) > 0:
                # -*- Parse response
                response_delta: ChoiceDelta = response.choices[0].delta
                response_content = response_delta.content
                response_function_call = response_delta.function_call
                response_tool_calls = response_delta.tool_calls

            # -*- Return content if present, otherwise get function call
            if response_content is not None:
                assistant_message_content += response_content
                completion_tokens += 1
                if completion_tokens == 1:
                    time_to_first_token = t.elapsed
                yield response_content

            # -*- Parse function call
            if response_function_call is not None:
                _function_name_stream = response_function_call.name
                if _function_name_stream is not None:
                    assistant_message_function_name += _function_name_stream
                _function_args_stream = response_function_call.arguments
                if _function_args_stream is not None:
                    assistant_message_function_arguments_str += _function_args_stream

            # -*- Parse tool calls
            if response_tool_calls is not None:
                if assistant_message_tool_calls is None:
                    assistant_message_tool_calls = []
                assistant_message_tool_calls.extend(response_tool_calls)
            if response.usage:
                response_usage: Optional[CompletionUsage] = response.usage
                response_prompt_tokens = response_usage.prompt_tokens
                response_completion_tokens = response_usage.completion_tokens
                response_total_tokens = response_usage.total_tokens

        t.stop()
        if completion_tokens > 0:
            logger.debug(f"Time to first token: {time_to_first_token:.4f}s, "
                         f"Time to generate response: {t.elapsed:.4f}s, "
                         f"Prompt tokens size: {prompt_tokens}, "
                         f"Completion tokens size: {completion_tokens}")

        # -*- Create assistant message
        assistant_message = Message(role="assistant")
        # -*- Add content to assistant message
        if assistant_message_content != "":
            assistant_message.content = assistant_message_content
        # -*- Add function call to assistant message
        if assistant_message_function_name != "":
            assistant_message.function_call = {
                "name": assistant_message_function_name,
                "arguments": assistant_message_function_arguments_str,
            }
        # -*- Add tool calls to assistant message
        if assistant_message_tool_calls is not None:
            # Build tool calls
            tool_calls: List[Dict[str, Any]] = []
            for _tool_call in assistant_message_tool_calls:
                _index = _tool_call.index
                _tool_call_id = _tool_call.id
                _tool_call_type = _tool_call.type
                _tool_call_function_name = _tool_call.function.name if _tool_call.function is not None else None
                _tool_call_function_arguments_str = (
                    _tool_call.function.arguments if _tool_call.function is not None else None
                )

                tool_call_at_index = tool_calls[_index] if len(tool_calls) > _index else None
                if tool_call_at_index is None:
                    tool_call_at_index_function_dict = {}
                    if _tool_call_function_name is not None:
                        tool_call_at_index_function_dict["name"] = _tool_call_function_name
                    if _tool_call_function_arguments_str is not None:
                        tool_call_at_index_function_dict["arguments"] = _tool_call_function_arguments_str
                    tool_call_at_index_dict = {
                        "id": _tool_call.id,
                        "type": _tool_call_type,
                        "function": tool_call_at_index_function_dict,
                    }
                    tool_calls.insert(_index, tool_call_at_index_dict)
                else:
                    if _tool_call_function_name is not None:
                        if "name" not in tool_call_at_index["function"]:
                            tool_call_at_index["function"]["name"] = _tool_call_function_name
                        else:
                            tool_call_at_index["function"]["name"] += _tool_call_function_name
                    if _tool_call_function_arguments_str is not None:
                        if "arguments" not in tool_call_at_index["function"]:
                            tool_call_at_index["function"]["arguments"] = _tool_call_function_arguments_str
                        else:
                            tool_call_at_index["function"]["arguments"] += _tool_call_function_arguments_str
                    if _tool_call_id is not None:
                        tool_call_at_index["id"] = _tool_call_id
                    if _tool_call_type is not None:
                        tool_call_at_index["type"] = _tool_call_type
            assistant_message.tool_calls = tool_calls

        # -*- Update usage metrics
        # Add response time to assistant metrics
        assistant_message.metrics["time"] = t.elapsed
        if time_to_first_token is not None:
            assistant_message.metrics["time_to_first_token"] = f"{time_to_first_token:.4f}s"
        if completion_tokens > 0:
            assistant_message.metrics["time_per_output_token"] = f"{t.elapsed / completion_tokens:.4f}s"

        # Add response time to LLM metrics
        if "response_times" not in self.metrics:
            self.metrics["response_times"] = []
        self.metrics["response_times"].append(t.elapsed)
        if time_to_first_token is not None:
            if "time_to_first_token" not in self.metrics:
                self.metrics["time_to_first_token"] = []
            self.metrics["time_to_first_token"].append(f"{time_to_first_token:.4f}s")
        if completion_tokens > 0:
            if "tokens_per_second" not in self.metrics:
                self.metrics["tokens_per_second"] = []
            self.metrics["tokens_per_second"].append(f"{completion_tokens / t.elapsed:.4f}")

        # Add token usage to metrics
        assistant_message.metrics["prompt_tokens"] = response_prompt_tokens
        if "prompt_tokens" not in self.metrics:
            self.metrics["prompt_tokens"] = response_prompt_tokens
        else:
            self.metrics["prompt_tokens"] += response_prompt_tokens
        assistant_message.metrics["completion_tokens"] = response_completion_tokens
        if "completion_tokens" not in self.metrics:
            self.metrics["completion_tokens"] = response_completion_tokens
        else:
            self.metrics["completion_tokens"] += response_completion_tokens
        assistant_message.metrics["input_tokens"] = response_prompt_tokens
        if "input_tokens" not in self.metrics:
            self.metrics["input_tokens"] = response_prompt_tokens
        else:
            self.metrics["input_tokens"] += response_prompt_tokens
        assistant_message.metrics["output_tokens"] = response_completion_tokens
        if "output_tokens" not in self.metrics:
            self.metrics["output_tokens"] = response_completion_tokens
        else:
            self.metrics["output_tokens"] += response_completion_tokens
        assistant_message.metrics["total_tokens"] = response_total_tokens
        if "total_tokens" not in self.metrics:
            self.metrics["total_tokens"] = response_total_tokens
        else:
            self.metrics["total_tokens"] += response_total_tokens

        # -*- Add assistant message to messages
        messages.append(assistant_message)
        assistant_message.log()

        # -*- Parse and run function call
        need_to_run_functions = assistant_message.function_call is not None or assistant_message.tool_calls is not None
        if need_to_run_functions and self.run_tools:
            if assistant_message.function_call is not None:
                function_call_message, function_call = self.run_function(function_call=assistant_message.function_call)
                messages.append(function_call_message)
                if self.show_tool_calls and function_call is not None:
                    yield f"\n - Running: {function_call.get_call_str()}\n\n"
                # -*- Yield new response using result of function call
                yield from self.response_stream(messages=messages)
            elif assistant_message.tool_calls is not None:
                function_calls_to_run: List[FunctionCall] = []
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

                if self.show_tool_calls:
                    if len(function_calls_to_run) == 1:
                        yield f"\n - Running: {function_calls_to_run[0].get_call_str()}\n\n"
                    elif len(function_calls_to_run) > 1:
                        yield "\nRunning:"
                        for _f in function_calls_to_run:
                            yield f"\n - {_f.get_call_str()}"
                        yield "\n\n"

                function_call_results = self.run_function_calls(function_calls_to_run)
                if len(function_call_results) > 0:
                    messages.extend(function_call_results)
                    # Code to show function call results
                    # for f in function_call_results:
                    #     yield "\n"
                    #     yield f.get_content_string()
                    #     yield "\n"
                # -*- Yield new response using results of tool calls
                yield from self.response_stream(messages=messages)
        logger.debug("---------- OpenAI Response End ----------")

    @staticmethod
    def calculate_tokens(content: str) -> int:
        """
        Calculate the number of tokens in the content.
            English chars: every 3 chars count as 1 token
            Chinese chars: every 1 char count as 1 token
        :param content: str
        :return: int
        """
        english_chars = sum(1 for c in content if c.isascii())
        chinese_chars = len(content) - english_chars
        return english_chars // 3 + chinese_chars

    async def aresponse_stream(self, messages: List[Message]) -> Any:
        logger.debug("---------- OpenAI Async Response Start ----------")
        for m in messages:
            m.log()

        assistant_message_content = ""
        assistant_message_function_name = ""
        assistant_message_function_arguments_str = ""
        assistant_message_tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
        completion_tokens = 0
        t = Timer()
        t.start()
        async_stream = self.ainvoke_stream(messages=messages)
        async for response in async_stream:
            response_content: Optional[str] = None
            response_function_call: Optional[ChoiceDeltaFunctionCall] = None
            response_tool_calls: Optional[List[ChoiceDeltaToolCall]] = None
            if len(response.choices) > 0:
                # -*- Parse response
                response_delta: ChoiceDelta = response.choices[0].delta
                response_content = response_delta.content
                response_function_call = response_delta.function_call
                response_tool_calls = response_delta.tool_calls

            # -*- Return content if present, otherwise get function call
            if response_content is not None:
                assistant_message_content += response_content
                completion_tokens += 1
                yield response_content

            # -*- Parse function call
            if response_function_call is not None:
                _function_name_stream = response_function_call.name
                if _function_name_stream is not None:
                    assistant_message_function_name += _function_name_stream
                _function_args_stream = response_function_call.arguments
                if _function_args_stream is not None:
                    assistant_message_function_arguments_str += _function_args_stream

            # -*- Parse tool calls
            if response_tool_calls is not None:
                if assistant_message_tool_calls is None:
                    assistant_message_tool_calls = []
                assistant_message_tool_calls.extend(response_tool_calls)

        t.stop()
        logger.debug(f"Time to generate response: {t.elapsed:.4f}s")

        # -*- Create assistant message
        assistant_message = Message(role="assistant")
        # -*- Add content to assistant message
        if assistant_message_content != "":
            assistant_message.content = assistant_message_content
        # -*- Add function call to assistant message
        if assistant_message_function_name != "":
            assistant_message.function_call = {
                "name": assistant_message_function_name,
                "arguments": assistant_message_function_arguments_str,
            }
        # -*- Add tool calls to assistant message
        if assistant_message_tool_calls is not None:
            # Build tool calls
            tool_calls: List[Dict[str, Any]] = []
            for _tool_call in assistant_message_tool_calls:
                _index = _tool_call.index
                _tool_call_id = _tool_call.id
                _tool_call_type = _tool_call.type
                _tool_call_function_name = _tool_call.function.name if _tool_call.function is not None else None
                _tool_call_function_arguments_str = (
                    _tool_call.function.arguments if _tool_call.function is not None else None
                )

                tool_call_at_index = tool_calls[_index] if len(tool_calls) > _index else None
                if tool_call_at_index is None:
                    tool_call_at_index_function_dict = {}
                    if _tool_call_function_name is not None:
                        tool_call_at_index_function_dict["name"] = _tool_call_function_name
                    if _tool_call_function_arguments_str is not None:
                        tool_call_at_index_function_dict["arguments"] = _tool_call_function_arguments_str
                    tool_call_at_index_dict = {
                        "id": _tool_call.id,
                        "type": _tool_call_type,
                        "function": tool_call_at_index_function_dict,
                    }
                    tool_calls.insert(_index, tool_call_at_index_dict)
                else:
                    if _tool_call_function_name is not None:
                        if "name" not in tool_call_at_index["function"]:
                            tool_call_at_index["function"]["name"] = _tool_call_function_name
                        else:
                            tool_call_at_index["function"]["name"] += _tool_call_function_name
                    if _tool_call_function_arguments_str is not None:
                        if "arguments" not in tool_call_at_index["function"]:
                            tool_call_at_index["function"]["arguments"] = _tool_call_function_arguments_str
                        else:
                            tool_call_at_index["function"]["arguments"] += _tool_call_function_arguments_str
                    if _tool_call_id is not None:
                        tool_call_at_index["id"] = _tool_call_id
                    if _tool_call_type is not None:
                        tool_call_at_index["type"] = _tool_call_type
            assistant_message.tool_calls = tool_calls

        # -*- Update usage metrics
        # Add response time to metrics
        assistant_message.metrics["time"] = t.elapsed
        if "response_times" not in self.metrics:
            self.metrics["response_times"] = []
        self.metrics["response_times"].append(t.elapsed)

        # Add token usage to metrics
        prompt_tokens = sum([self.calculate_tokens(i.get_content_string()) for i in messages if i])
        assistant_message.metrics["prompt_tokens"] = prompt_tokens
        if "prompt_tokens" not in self.metrics:
            self.metrics["prompt_tokens"] = prompt_tokens
        else:
            self.metrics["prompt_tokens"] += prompt_tokens
        logger.debug(f"Estimated completion tokens: {completion_tokens}")
        assistant_message.metrics["completion_tokens"] = completion_tokens
        if "completion_tokens" not in self.metrics:
            self.metrics["completion_tokens"] = completion_tokens
        else:
            self.metrics["completion_tokens"] += completion_tokens
        total_tokens = prompt_tokens + completion_tokens
        assistant_message.metrics["total_tokens"] = total_tokens
        if "total_tokens" not in self.metrics:
            self.metrics["total_tokens"] = total_tokens
        else:
            self.metrics["total_tokens"] += total_tokens

        # -*- Add assistant message to messages
        messages.append(assistant_message)
        assistant_message.log()

        # -*- Parse and run function call
        run_functions = assistant_message.function_call is not None or assistant_message.tool_calls is not None
        if run_functions and self.run_tools:
            if assistant_message.function_call is not None:
                call_message, function_call = self.run_function(function_call=assistant_message.function_call)
                messages.append(call_message)
                if self.show_tool_calls and function_call is not None:
                    yield f"\n - Running: {function_call.get_call_str()}\n\n"
                # -*- Yield new response using result of function call
                fc_stream = self.aresponse_stream(messages=messages)
                async for fc in fc_stream:
                    yield fc
                # yield from self.response_stream(messages=messages)
            elif assistant_message.tool_calls is not None:
                function_calls_to_run: List[FunctionCall] = []
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

                if self.show_tool_calls:
                    if len(function_calls_to_run) == 1:
                        yield f"\n - Running: {function_calls_to_run[0].get_call_str()}\n\n"
                    elif len(function_calls_to_run) > 1:
                        yield "\nRunning:"
                        for _f in function_calls_to_run:
                            yield f"\n - {_f.get_call_str()}"
                        yield "\n\n"

                function_call_results = self.run_function_calls(function_calls_to_run)
                if len(function_call_results) > 0:
                    messages.extend(function_call_results)
                    # Code to show function call results
                    # for f in function_call_results:
                    #     yield "\n"
                    #     yield f.get_content_string()
                    #     yield "\n"
                # -*- Yield new response using results of tool calls
                fc_stream = self.aresponse_stream(messages=messages)
                async for fc in fc_stream:
                    yield fc
        logger.debug("---------- OpenAI Async Response End ----------")
