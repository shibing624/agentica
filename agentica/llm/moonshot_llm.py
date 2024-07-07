# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
kimi api refer: https://platform.moonshot.cn/docs/api/tool-use#%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8
"""
from os import getenv
from typing import Optional, List, Iterator, Dict, Any

from openai import OpenAI as OpenAIClient

from agentica.llm.base import LLM
from agentica.message import Message
from agentica.tool import FunctionCall, get_function_call_for_tool_call
from agentica.utils.log import logger
from agentica.utils.timer import Timer


class MoonshotLLM(LLM):
    name: str = "Moonshot"
    model: str = "moonshot-v1-8k"
    api_key: Optional[str] = getenv("MOONSHOT_API_KEY")
    base_url: str = "https://api.moonshot.cn/v1"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    # Deactivate tool calls after 1 tool call
    deactivate_tools_after_use: bool = False
    request_params: Optional[Dict[str, Any]] = None
    client_params: Optional[Dict[str, Any]] = None
    # -*- Provide the client manually
    client: Optional[OpenAIClient] = None

    def get_client(self) -> OpenAIClient:
        if self.client:
            return self.client

        _client_params: Dict[str, Any] = {}
        if self.api_key:
            _client_params["api_key"] = self.api_key
        if self.base_url:
            _client_params["base_url"] = self.base_url
        if self.client_params:
            _client_params.update(self.client_params)
        self.client = OpenAIClient(**_client_params)
        return self.client

    @property
    def api_kwargs(self) -> Dict[str, Any]:
        _request_params: Dict[str, Any] = {}
        if self.max_tokens:
            _request_params["max_tokens"] = self.max_tokens
        if self.temperature:
            _request_params["temperature"] = self.temperature
        if self.top_p:
            _request_params["top_p"] = self.top_p
        if self.top_k:
            _request_params["top_k"] = self.top_k
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
        if self.max_tokens:
            _dict["max_tokens"] = self.max_tokens
        if self.temperature:
            _dict["temperature"] = self.temperature
        if self.top_p:
            _dict["top_p"] = self.top_p
        if self.tools:
            _dict["tools"] = self.get_tools_for_api()
            if self.tool_choice is None:
                _dict["tool_choice"] = "auto"
            else:
                _dict["tool_choice"] = self.tool_choice
        return _dict

    def invoke(self, messages: List[Message]):
        return self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            **self.api_kwargs,
        )

    def invoke_stream(self, messages: List[Message]):
        yield from self.get_client().chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            stream=True,
            **self.api_kwargs,
        )

    def response(self, messages: List[Message]) -> str:
        logger.debug("---------- Moonshot Response Start ----------")
        # -*- Log messages for debugging
        for m in messages:
            m.log()

        t = Timer()
        t.start()
        response = self.invoke(messages=messages)
        t.stop()
        logger.debug(f"Time to generate response: {t.elapsed:.4f}s")

        # -*- Parse response
        response_message = response.choices[0].message
        response_role = response_message.role
        response_content: Optional[str] = response_message.content
        response_tool_calls = response_message.tool_calls

        tool_calls = []
        if response_tool_calls is not None:
            tool_calls = [t.model_dump() for t in response_tool_calls]
            if not response_content:
                response_content = f"Running tool calls: {tool_calls}"

        # -*- Create assistant message
        assistant_message = Message(
            role=response_role or "assistant",
            content=response_content,
        )

        # -*- Update usage metrics
        # Add response time to metrics
        assistant_message.metrics["time"] = t.elapsed
        if "response_times" not in self.metrics:
            self.metrics["response_times"] = []
        self.metrics["response_times"].append(t.elapsed)

        # Add token usage to metrics
        response_usage = response.usage
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
        if tool_calls is not None and self.run_tools:
            final_response = ""
            function_calls_to_run: List[FunctionCall] = []
            for tool_call in tool_calls:
                _function_call = get_function_call_for_tool_call(tool_call, self.functions)
                if _function_call is None:
                    messages.append(Message(role="user", content="Could not find function to call."))
                    continue
                if _function_call.error is not None:
                    messages.append(Message(role="user", content=_function_call.error))
                    continue
                function_calls_to_run.append(_function_call)

            if self.show_tool_calls:
                if len(function_calls_to_run) == 1:
                    final_response += f" - Running: {function_calls_to_run[0].get_call_str()}\n\n"
                elif len(function_calls_to_run) > 1:
                    final_response += "Running:"
                    for _f in function_calls_to_run:
                        final_response += f"\n - {_f.get_call_str()}"
                    final_response += "\n\n"

            function_call_results = self.run_function_calls(function_calls_to_run, role="user")
            if len(function_call_results) > 0:
                fc_responses = "<tool_results>"

                for _fc_message in function_call_results:
                    fc_responses += "<result>"
                    fc_responses += "<tool_name>" + _fc_message.tool_call_name + "</tool_name>"  # type: ignore
                    fc_responses += "<stdout>" + _fc_message.content + "</stdout>"  # type: ignore
                    fc_responses += "</result>"
                fc_responses += "</tool_results>"

                messages.append(Message(role="user", content=fc_responses))

            # Deactivate tool calls after 1 tool call
            if self.deactivate_tools_after_use:
                self.deactivate_function_calls()

            # -*- Yield new response using results of tool calls
            last_message = messages[-1]
            if last_message.role == "user" and last_message.content is not None:
                final_response += self.response(messages=messages)
                return final_response
        logger.debug("---------- Moonshot Response End ----------")
        # -*- Return content if no function calls are present
        if assistant_message.content is not None:
            return assistant_message.get_content_string()
        return "Something went wrong, please try again."

    def response_stream(self, messages: List[Message]) -> Iterator[str]:
        logger.debug("MoonshotLLM tool use not support stream, use response instead.")
        r = self.response(messages)
        for i in r:
            yield i
