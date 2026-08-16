# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: OpenAI Responses API model adapter with reasoning and function tools.
"""

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel

from agentica.model.base import Model, NativeCompactionResult
from agentica.model.message import Message
from agentica.model.metrics import Metrics
from agentica.model.openai.chat import OpenAIChat
from agentica.model.response import ModelResponse
from agentica.model.stream_retry import stream_with_retry
from agentica.utils.tokens import count_schema_tokens, count_text_tokens, count_tool_tokens


@dataclass
class OpenAIResponses(OpenAIChat):
    """OpenAI Responses API adapter.

    ``reasoning`` is the effort name stored in Agentica config. It is serialized
    to the Responses request shape ``{"effort": <value>}``.
    """

    id: str = "gpt-5.6-sol"
    name: str = "OpenAIResponses"
    provider: str = "OpenAI"
    supports_native_compaction: bool = True

    reasoning: Optional[str] = None
    max_output_tokens: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    truncation: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.reasoning_effort is not None:
            raise ValueError("OpenAIResponses uses 'reasoning', not Chat Completions' 'reasoning_effort'.")

    def describe_thinking_mode(self) -> str:
        if self.reasoning is None:
            return "off"
        return f"on(reasoning={self.reasoning})"

    def to_dict(self) -> Dict[str, Any]:
        model_dict = Model.to_dict(self)
        model_dict.update(
            {
                "reasoning": self.reasoning,
                "max_output_tokens": self.max_output_tokens if self.max_output_tokens is not None else self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "store": self.store,
            }
        )
        tools = self._tools_for_responses()
        if tools:
            model_dict["tools"] = tools
            choice = self._tool_choice_for_responses()
            model_dict["tool_choice"] = choice if choice is not None else "auto"
        return {key: value for key, value in model_dict.items() if value is not None}

    @staticmethod
    def _dump(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        return value.model_dump(exclude_none=True)

    @staticmethod
    def _output_text(output_items: List[Any]) -> str:
        parts: List[str] = []
        for item in output_items:
            item_type = item.get("type") if isinstance(item, dict) else item.type
            if item_type != "message":
                continue
            content = item.get("content", []) if isinstance(item, dict) else item.content
            for block in content:
                block_type = block.get("type") if isinstance(block, dict) else block.type
                if block_type == "output_text":
                    parts.append(block.get("text", "") if isinstance(block, dict) else block.text)
        return "".join(parts)

    @staticmethod
    def _reasoning_text(output_items: List[Any]) -> str:
        parts: List[str] = []
        for item in output_items:
            item_type = item.get("type") if isinstance(item, dict) else item.type
            if item_type != "reasoning":
                continue
            summary = item.get("summary", []) if isinstance(item, dict) else item.summary
            for block in summary:
                text = block.get("text") if isinstance(block, dict) else block.text
                if text:
                    parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def _function_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            return dict(tool)
        formatted = {"type": "function", **function}
        formatted.pop("cache_control", None)
        return formatted

    def _tools_for_responses(self) -> List[Dict[str, Any]]:
        return [self._function_tool(tool) for tool in (self.get_tools_for_api() or [])]

    def _tool_choice_for_responses(self) -> Any:
        choice = self.get_tool_choice()
        if not isinstance(choice, dict):
            return choice
        function = choice.get("function")
        if choice.get("type") == "function" and isinstance(function, dict):
            return {"type": "function", "name": function["name"]}
        return choice

    @staticmethod
    def _content_for_responses(content: Any) -> Any:
        if not isinstance(content, list):
            return content
        blocks: List[Any] = []
        for block in content:
            if not isinstance(block, dict):
                blocks.append(block)
                continue
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "input_text", "text": block.get("text", "")})
            elif block_type == "image_url":
                image_url = block.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                blocks.append({"type": "input_image", "image_url": url})
            else:
                blocks.append(dict(block))
        return blocks

    def _assistant_items(self, message: Message) -> List[Dict[str, Any]]:
        provider_data = message.provider_data
        raw_output = []
        if isinstance(provider_data, dict) and provider_data.get("object") == "response":
            raw_output = provider_data.get("output") or []

        current_calls = {call["id"]: call for call in (message.tool_calls or [])}
        emitted_call_ids = set()
        items: List[Dict[str, Any]] = []
        raw_text = self._output_text(raw_output)

        for raw_item in raw_output:
            item = self._dump(raw_item)
            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                call = current_calls.get(call_id)
                if call is None:
                    continue
                function = call["function"]
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": function["name"],
                        "arguments": function.get("arguments", "{}"),
                    }
                )
                emitted_call_ids.add(call_id)
            elif item_type == "message":
                if raw_text == message.get_content_string():
                    items.append(item)
            else:
                items.append(item)

        if message.content is not None and (not raw_output or raw_text != message.get_content_string()):
            items.append(
                {
                    "role": "assistant",
                    "content": self._content_for_responses(message.content),
                }
            )

        for call_id, call in current_calls.items():
            if call_id in emitted_call_ids:
                continue
            function = call["function"]
            items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": function["name"],
                    "arguments": function.get("arguments", "{}"),
                }
            )
        return items

    def _checkpoint_identity(self) -> Dict[str, str]:
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        return {
            "provider": self.provider,
            "model": self.id,
            "base_url": str(base_url).rstrip("/"),
        }

    def _compatible_checkpoint(self, message: Message) -> Optional[Dict[str, Any]]:
        checkpoint = message.provider_checkpoint
        if not isinstance(checkpoint, dict):
            return None
        if checkpoint.get("type") != "openai_responses_compaction":
            return None
        identity = self._checkpoint_identity()
        if any(checkpoint.get(key) != value for key, value in identity.items()):
            return None
        output = checkpoint.get("output")
        return checkpoint if isinstance(output, list) else None

    def has_compatible_native_checkpoint(self, messages: List[Message]) -> bool:
        return any(self._compatible_checkpoint(message) is not None for message in messages)

    def _append_formatted_message(self, formatted: List[Dict[str, Any]], message: Message) -> None:
        if message.role == "assistant":
            formatted.extend(self._assistant_items(message))
        elif message.role == "tool":
            if not message.tool_call_id:
                raise ValueError("Tool result must be a response to a preceding message with a tool call id.")
            formatted.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.get_content_string(),
                }
            )
        else:
            if message.role == "user" and message.images is not None:
                message = self.add_images_to_message(message=message, images=message.images)
            formatted.append(
                {
                    "role": message.role,
                    "content": self._content_for_responses(message.content),
                }
            )

    def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        checkpoint_index = -1
        checkpoint: Optional[Dict[str, Any]] = None
        for index in range(len(messages) - 1, -1, -1):
            checkpoint = self._compatible_checkpoint(messages[index])
            if checkpoint is not None:
                checkpoint_index = index
                break

        if checkpoint is not None:
            # Standalone compact output is the canonical next context window.
            # System instructions are re-applied separately because compacted
            # output contains retained user items plus the opaque checkpoint.
            for message in messages:
                if message.role == "system":
                    self._append_formatted_message(formatted, message)
            formatted.extend(dict(item) for item in checkpoint["output"])
            for message in messages[checkpoint_index + 1:]:
                if message.role != "system":
                    self._append_formatted_message(formatted, message)
            return formatted

        for message in messages:
            self._append_formatted_message(formatted, message)
        return formatted

    def native_compaction_token_limit(self) -> int:
        """Largest input worth sending to native compaction.

        Reserve room for the compaction response (output budget + slack), but
        never below 80% of the window: on small windows the headroom term goes
        negative and the previous ``max(1, ...)`` floor collapsed the limit to
        1, making ``should_native_compact`` fire every single turn. The 80%
        floor also matches the ``compress_token_limit`` cap ``min()``-ed in
        downstream, so the collapse point no longer changes behaviour.
        """
        output_limit = self.max_output_tokens if self.max_output_tokens is not None else self.max_tokens
        if output_limit is None:
            output_limit = 16_384
        return max(int(self.context_window * 0.8), self.context_window - output_limit - 8_192)

    def estimate_native_compaction_tokens(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
    ) -> int:
        formatted = self.format_messages(messages)
        total = count_text_tokens(json.dumps(formatted, ensure_ascii=False), self.id)
        if tools:
            total += count_tool_tokens(tools, self.id)
        if self.response_format is not None:
            total += count_schema_tokens(self.response_format, self.id)
        return total

    async def compact_context(
        self,
        messages: List[Message],
        instructions: Optional[str] = None,
    ) -> NativeCompactionResult:
        """Create an opaque Responses checkpoint for subsequent requests."""
        request: Dict[str, Any] = {
            "model": self.id,
            "input": self.format_messages(messages),
        }
        if instructions is not None:
            request["instructions"] = instructions
        for name, value in (
            ("extra_headers", self.extra_headers),
            ("extra_query", self.extra_query),
            ("extra_body", self.extra_body),
        ):
            if value is not None:
                request[name] = value

        metrics = Metrics()
        metrics.response_timer.start()
        try:
            response = await self.get_client().responses.compact(**request)
        finally:
            metrics.response_timer.stop()

        if response.object != "response.compaction":
            raise RuntimeError(f"Responses compact returned unexpected object: {response.object!r}")

        usage = response.usage.model_dump(exclude_none=True)
        accounting_message = Message(role="assistant")
        self.update_usage_metrics(accounting_message, metrics, self._completion_usage(response.usage))
        checkpoint = {
            "type": "openai_responses_compaction",
            **self._checkpoint_identity(),
            "id": response.id,
            "created_at": response.created_at,
            "output": [self._dump(item) for item in response.output],
            "usage": usage,
        }
        return NativeCompactionResult(checkpoint=checkpoint, usage=usage)

    @property
    def request_kwargs(self) -> Dict[str, Any]:
        request: Dict[str, Any] = {}
        passthrough = {
            "store": self.store,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "user": self.user,
            "metadata": self.metadata,
            "extra_headers": self.extra_headers,
            "extra_query": self.extra_query,
            "extra_body": self.extra_body,
        }
        request.update({name: value for name, value in passthrough.items() if value is not None})

        output_limit = self.max_output_tokens if self.max_output_tokens is not None else self.max_tokens
        if output_limit is not None:
            request["max_output_tokens"] = output_limit
        if self.reasoning is not None:
            request["reasoning"] = {"effort": self.reasoning}
        if self.parallel_tool_calls is not None:
            request["parallel_tool_calls"] = self.parallel_tool_calls
        if self.truncation is not None:
            request["truncation"] = self.truncation

        text: Dict[str, Any] = {}
        if self.verbosity is not None:
            text["verbosity"] = self.verbosity
        if isinstance(self.response_format, dict):
            response_format = dict(self.response_format)
            if response_format.get("type") == "json_schema" and isinstance(response_format.get("json_schema"), dict):
                schema = response_format.pop("json_schema")
                response_format = {"type": "json_schema", **schema}
            text["format"] = response_format
        if text:
            request["text"] = text

        tools = self._tools_for_responses()
        if tools:
            request["tools"] = tools
            choice = self._tool_choice_for_responses()
            request["tool_choice"] = choice if choice is not None else "auto"
        if self.request_params is not None:
            request.update(self.request_params)
        if self.reasoning is not None and self.store is not True:
            include = list(request.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            request["include"] = include
        return request

    async def invoke(self, messages: List[Message]) -> Any:
        """Send one non-streaming Responses API request."""
        request = self.request_kwargs
        formatted = self.format_messages(messages)
        try:
            if (
                self.response_format is not None
                and self.use_structured_outputs
                and isinstance(self.response_format, type)
                and issubclass(self.response_format, BaseModel)
            ):
                request.pop("text", None)
                return await self.get_client().responses.parse(
                    model=self.id,
                    input=formatted,
                    text_format=self.response_format,
                    **request,
                )
            return await self.get_client().responses.create(model=self.id, input=formatted, **request)
        except Exception as error:
            self._learn_context_limit_from_error(str(error))
            raise

    async def invoke_stream(self, messages: List[Message]) -> AsyncIterator[Any]:
        """Open and iterate one streaming Responses API request."""
        request = self.request_kwargs
        formatted = self.format_messages(messages)

        async def _open() -> AsyncIterator[Any]:
            try:
                return await self.get_client().responses.create(
                    model=self.id,
                    input=formatted,
                    stream=True,
                    **request,
                )
            except Exception as error:
                self._learn_context_limit_from_error(str(error))
                raise

        async for event in stream_with_retry(
            _open,
            extra_substrings=self.extra_retryable_substrings,
            max_retries=self.max_retries or 0,
            provider_label=f"openai-responses/{self.id}",
        ):
            yield event

    @staticmethod
    def _finish_reason(response: Any) -> str:
        if response.status == "incomplete" and response.incomplete_details is not None:
            reason = response.incomplete_details.reason
            if reason == "max_output_tokens":
                return "length"
            if reason == "content_filter":
                return "content_filter"
            return reason or "incomplete"
        if any(item.type == "function_call" for item in response.output):
            return "tool_calls"
        return "stop"

    @staticmethod
    def _completion_usage(response_usage: Any) -> Any:
        if response_usage is None:
            return None
        return SimpleNamespace(
            prompt_tokens=response_usage.input_tokens,
            completion_tokens=response_usage.output_tokens,
            total_tokens=response_usage.total_tokens,
            prompt_tokens_details=response_usage.input_tokens_details,
            completion_tokens_details=response_usage.output_tokens_details,
        )

    @staticmethod
    def _replay_provider_data(response: Any) -> Dict[str, Any]:
        """Keep only the parts of the Response that replay actually reads back.

        A full ``response.model_dump()`` is mostly an echo of the REQUEST, and
        the tool schema dominates it: measured over one 54MB transcript corpus,
        ``tools`` was 89% of these blobs (11.7MB across 395 assistant entries)
        for 3 distinct schemas repeated verbatim on every entry. None of the
        echo is ever consumed:

        - ``_assistant_items`` reads ``object`` and ``output``, nothing else
        - ``provider_data`` is local-only and never reaches the wire
          (``Message.to_dict``'s allowlist, pinned by
          ``tests/model/test_wire_payload_allowlist.py``)
        - Responses stateful chaining rides on ``provider_checkpoint``, a
          separate field persisted separately

        So this is an allowlist, not a ``tools`` blacklist: a future fat
        request-echo key cannot silently reintroduce the bloat.
        """
        return response.model_dump(exclude_none=True, include={"object", "output"})

    def _assistant_message(self, response: Any, metrics: Metrics) -> Message:
        output = response.output
        content = self._output_text(output).lstrip("\n")
        reasoning_content = self._reasoning_text(output)
        tool_calls: List[Dict[str, Any]] = []
        for item in output:
            if item.type != "function_call":
                continue
            tool_calls.append(
                {
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                }
            )

        assistant = Message(
            role="assistant",
            content=content or None,
            reasoning_content=reasoning_content or None,
            tool_calls=tool_calls or None,
            provider_data=self._replay_provider_data(response),
        )
        assistant.finish_reason = self._finish_reason(response)
        self.update_usage_metrics(assistant, metrics, self._completion_usage(response.usage))
        return assistant

    @staticmethod
    def _raise_failed_response(response: Any) -> None:
        if response.status != "failed":
            return
        error = response.error
        if error is None:
            raise RuntimeError("Responses API request failed without an error payload.")
        raise RuntimeError(f"Responses API request failed: {error.code}: {error.message}")

    async def response(self, messages: List[Message]) -> ModelResponse:
        """Generate one Agentica response from the Responses API."""
        self.sanitize_messages(messages)
        self._log_messages(messages)
        metrics = Metrics()
        metrics.response_timer.start()
        response = await self.invoke(messages)
        metrics.response_timer.stop()
        self._raise_failed_response(response)

        assistant = self._assistant_message(response, metrics)
        messages.append(assistant)
        assistant.log()
        metrics.log()

        model_response = ModelResponse(
            content=assistant.get_content_string() if assistant.content is not None else None,
            reasoning_content=assistant.reasoning_content,
            finish_reason=assistant.finish_reason,
        )
        if (
            self.response_format is not None
            and self.use_structured_outputs
            and isinstance(self.response_format, type)
            and issubclass(self.response_format, BaseModel)
        ):
            parsed = response.output_parsed
            if parsed is not None:
                model_response.parsed = parsed

        self.last_finish_reason = assistant.finish_reason
        if (
            await self.handle_tool_calls(
                assistant_message=assistant,
                messages=messages,
                model_response=model_response,
                tool_role="tool",
            )
            is not None
        ):
            return model_response
        return model_response

    async def response_stream(self, messages: List[Message]) -> AsyncIterator[ModelResponse]:
        """Generate one streaming Agentica response from Responses events."""
        self.sanitize_messages(messages)
        self._log_messages(messages)
        metrics = Metrics()
        final_response = None
        streamed_content = ""
        streamed_reasoning = ""

        metrics.response_timer.start()
        async for event in self.invoke_stream(messages):
            if event.type == "response.output_text.delta":
                if metrics.time_to_first_token is None:
                    metrics.time_to_first_token = metrics.response_timer.elapsed
                streamed_content += event.delta
                yield ModelResponse(content=event.delta)
            elif event.type in (
                "response.reasoning_summary_text.delta",
                "response.reasoning_text.delta",
            ):
                if metrics.time_to_first_token is None:
                    metrics.time_to_first_token = metrics.response_timer.elapsed
                streamed_reasoning += event.delta
                yield ModelResponse(reasoning_content=event.delta)
            elif event.type in ("response.completed", "response.incomplete"):
                final_response = event.response
            elif event.type == "response.failed":
                self._raise_failed_response(event.response)
            elif event.type == "error":
                raise RuntimeError(f"Responses API stream error: {event.code}: {event.message}")
        metrics.response_timer.stop()

        if final_response is None:
            raise RuntimeError("Responses API stream ended without a terminal response event.")
        self._raise_failed_response(final_response)

        final_content = self._output_text(final_response.output).lstrip("\n")
        final_reasoning = self._reasoning_text(final_response.output)
        if final_content and not streamed_content:
            yield ModelResponse(content=final_content)
        if final_reasoning and not streamed_reasoning:
            yield ModelResponse(reasoning_content=final_reasoning)

        assistant = self._assistant_message(final_response, metrics)
        messages.append(assistant)
        assistant.log()
        metrics.log()
        self.last_finish_reason = assistant.finish_reason

        if assistant.tool_calls and self.run_tools:
            async for tool_response in self.handle_stream_tool_calls(
                assistant_message=assistant,
                messages=messages,
                tool_role="tool",
            ):
                yield tool_response
