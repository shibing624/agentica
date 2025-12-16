# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: React Agent implementing the MultiTurn ReAct paradigm.

Supports two tool calling modes:
1. OpenAI Function Calling (default): Uses OpenAI-style tools parameter, works with OpenAI/DeepSeek/etc.
2. XML Protocol: Uses <tool_call> tags, compatible with all LLMs including local models
"""
from typing import Any, Callable, Dict, List, Optional
import asyncio
import datetime
import inspect
import json
import json5
import random
import time
import re

from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)

from webresearcher.base import today_date, Message, build_text_completion_prompt, count_tokens as count_tokens_base
from webresearcher.log import logger
from webresearcher.prompt import get_react_system_prompt_xml, TOOL_DESCRIPTIONS, get_react_system_prompt_fc
from webresearcher.tool_file import FileParser
from webresearcher.tool_scholar import Scholar
from webresearcher.tool_python import PythonInterpreter
from webresearcher.tool_search import Search
from webresearcher.tool_visit import Visit
from webresearcher.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    OBS_START,
    OBS_END,
    MAX_LLM_CALL_PER_RUN,
    FILE_DIR,
    LLM_MODEL_NAME,
)


TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


class ReactAgent:
    """
    A lightweight MultiTurn ReAct-style agent compatible with local vLLM or OpenAI endpoints.
    
    Supports two tool calling modes:
    - use_xml_protocol=False (default): OpenAI-style function calling, works with OpenAI/DeepSeek/etc.
    - use_xml_protocol=True: XML-based <tool_call> tags, compatible with all LLMs including local models
    """

    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        function_list: Optional[List[str]] = None, # "search", "visit", "python" defult can be use without api key
        instruction: str = "",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_xml_protocol: bool = False,
    ) -> None:
        llm_config = dict(llm_config or {})
        if api_key:
            llm_config["api_key"] = api_key
        if base_url:
            llm_config["base_url"] = base_url

        self.llm_config = llm_config
        self.model = model or self.llm_config.get("model", LLM_MODEL_NAME)
        self.generate_cfg = self.llm_config.get("generate_cfg", {"temperature": 0.6, "top_p": 0.95})
        self.api_key = self.llm_config.get("api_key", LLM_API_KEY)
        self.base_url = self.llm_config.get("base_url", LLM_BASE_URL)
        self.llm_timeout = self.llm_config.get("llm_timeout", 600.0)
        self.agent_timeout = self.llm_config.get("agent_timeout", 1800.0)

        self.function_list = function_list or list(TOOL_MAP.keys())
        self.instruction = instruction
        self.use_xml_protocol = use_xml_protocol

    def _get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions in OpenAI function calling format."""
        tools = []
        for tool_name in self.function_list:
            if tool_name in TOOL_DESCRIPTIONS:
                tools.append(TOOL_DESCRIPTIONS[tool_name])
            else:
                # Fallback for custom tools
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Custom tool '{tool_name}'",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                })
        return tools

    def count_tokens(self, messages: List[Dict]) -> int:
        try:
            full_message: List[Message] = []
            for x in messages:
                if isinstance(x, dict):
                    full_message.append(Message(**x))
                else:
                    full_message.append(x)
            full_prompt = build_text_completion_prompt(full_message, allow_special=True)
            return count_tokens_base(full_prompt, self.model)
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}. Using simple split.")
            return sum(len(str(x).split()) for x in messages)

    async def call_server(
        self, 
        msgs: List[Dict], 
        stop_sequences: Optional[List[str]] = None, 
        max_tries: int = 5,
        tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Call LLM server with support for both XML-based and native function calling.
        Uses AsyncOpenAI for true async concurrency.
        
        Returns:
            Dict with keys:
            - content: str, the text content
            - reasoning_content: Optional[str], thinking process (for models like DeepSeek)
            - tool_calls: Optional[List], native tool calls (when use_native_tools=True)
            - raw_message: the original message object
        """
        client = AsyncOpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=self.llm_timeout,
        )
        base_sleep_time = 1
        stop_sequences = stop_sequences or ([OBS_START] if self.use_xml_protocol else None)

        for attempt in range(max_tries):
            try:
                request_params = {
                    "model": self.model,
                    "messages": msgs,
                    "temperature": self.generate_cfg.get("temperature", 0.6),
                    "top_p": self.generate_cfg.get("top_p", 0.95),
                }
                
                # Add stop sequences only for XML mode
                if stop_sequences and self.use_xml_protocol:
                    request_params["stop"] = stop_sequences
                
                # Add tools for function calling mode (non-XML)
                if tools and not self.use_xml_protocol:
                    request_params["tools"] = tools
                
                # Add extra_body for thinking mode (DeepSeek R1 etc.)
                model_thinking_type = self.generate_cfg.get("model_thinking_type", "")
                if model_thinking_type:
                    request_params["extra_body"] = {
                        "thinking": {"type": model_thinking_type}
                    }
                
                # Use native async call
                chat_response = await client.chat.completions.create(**request_params)
                
                message = chat_response.choices[0].message
                content = message.content or ""
                
                # Extract reasoning_content if available (DeepSeek R1, etc.)
                reasoning_content = getattr(message, 'reasoning_content', None)
                
                # Extract native tool_calls if available
                tool_calls = getattr(message, 'tool_calls', None)
                
                logger.debug(
                    f"Input messages: {msgs}, \n"
                    f"Reasoning_content: {reasoning_content}, \n"
                    f"Tool_calls: {tool_calls}, \n"
                    f"LLM Response: {content}"
                )
                
                return {
                    "content": content.strip() if content else "",
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                    "raw_message": message,
                }
                
            except RateLimitError as e:
                logger.warning(f"Attempt {attempt + 1} rate limit error: {e}")
            except AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                break  # Don't retry auth errors
            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.warning(f"Attempt {attempt + 1} API error: {e}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                logger.warning(f"Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
        
        return {
            "content": "LLM server error.",
            "reasoning_content": None,
            "tool_calls": None,
            "raw_message": None,
        }

    @staticmethod
    def _strip_after_tool_response(content: str) -> str:
        if OBS_START in content:
            return content.split(OBS_START, 1)[0].strip()
        return content

    async def _execute_function_call(self, tool_call) -> str:
        """Execute an OpenAI-style function call."""
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
        
        logger.debug(f"Native tool call: {func_name}({args_str})")
        
        if func_name not in TOOL_MAP:
            return f"Error: Tool {func_name} not found"
        
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return f"Error: Failed to decode arguments: {args_str}"
        
        tool = TOOL_MAP[func_name]
        try:
            # Special handling for python tool: extract code from arguments
            if func_name == "python":
                code = args.get("code", "")
                if not code:
                    return "[Python Interpreter Error]: Empty code. Please provide code in arguments.code"
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.call, code)
                return result if isinstance(result, str) else str(result)
            
            if asyncio.iscoroutinefunction(tool.call):
                if func_name == "parse_file":
                    params = {"files": args.get("files")}
                    result = await tool.call(params, file_root_path=FILE_DIR)
                else:
                    result = await tool.call(args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.call, args)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: Tool execution failed. {e}"

    async def _execute_xml_tool(self, tool_call_block: str) -> str:
        """Execute a tool call from XML <tool_call> block."""
        # Python inline code path
        if "<code>" in tool_call_block and "</code>" in tool_call_block and "python" in tool_call_block.lower():
            code_raw = tool_call_block.split("<code>", 1)[1].split("</code>", 1)[0].strip()
            result = TOOL_MAP["python"].call(code_raw)
            return result if isinstance(result, str) else str(result)

        # JSON tool path
        try:
            tool_call = json5.loads(tool_call_block)
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
        except Exception:
            return 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'

        if tool_name not in TOOL_MAP:
            return f"Error: Tool {tool_name} not found"

        tool = TOOL_MAP[tool_name]
        # handle async tool (parse_file) with file root
        try:
            if asyncio.iscoroutinefunction(tool.call):
                if tool_name == "parse_file":
                    params = {"files": tool_args.get("files")}
                    result = await tool.call(params, file_root_path=FILE_DIR)
                else:
                    result = await tool.call(tool_args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, tool.call, tool_args)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: Tool execution failed. {e}"

    def _parse_answer(self, content: str) -> Dict[str, Optional[str]]:
        ans = {
            "answer": None,
            "terminate": False,
        }
        # Prefer <answer> as a termination signal; if both exist, <answer> wins
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            ans["answer"] = answer_match.group(1).strip()
            ans["terminate"] = True  # Treat <answer> as a terminate signal
            return ans
        term_match = re.search(r"<terminate>(.*?)</terminate>", content, re.DOTALL)
        if term_match:
            ans["terminate"] = True
            body = term_match.group(1)
            if body:
                ans["answer"] = body.strip()
        return ans

    async def run(self, question: str, progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None) -> Dict[str, str]:
        """
        Run the agent loop.
        
        Args:
            question: User's question
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with question, prediction, termination, and trajectory
        """
        async def emit(event: Dict[str, Any]):
            if not callable(progress_callback):
                return
            event.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
            try:
                maybe = progress_callback(event)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as callback_err:
                logger.warning(f"progress_callback raised error: {callback_err}")

        # Build system prompt (XML mode uses detailed prompt, function calling uses simpler one)
        if self.use_xml_protocol:
            system_prompt = get_react_system_prompt_xml(today_date(), self.function_list, self.instruction, question=question)
        else:
            system_prompt = get_react_system_prompt_fc(today_date(), self.function_list, self.instruction, question=question)
        
        messages: List[Dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        # Get tool definitions for function calling mode
        tool_definitions = self._get_tool_definitions() if not self.use_xml_protocol else None

        start_time = time.time()
        remaining = MAX_LLM_CALL_PER_RUN
        round_num = 0

        while remaining > 0:
            if time.time() - start_time > self.agent_timeout:
                best_effort = "Final answer generated by agent (timeout)."
                await emit({"type": "final", "answer": best_effort, "termination": "timeout"})
                return {
                    "question": question,
                    "prediction": best_effort,
                    "termination": "timeout",
                    "trajectory": messages,
                }

            remaining -= 1
            round_num += 1
            
            response = await self.call_server(messages, tools=tool_definitions)
            content = response["content"]
            reasoning_content = response["reasoning_content"]
            tool_calls = response["tool_calls"]
            raw_message = response["raw_message"]
            
            # Log reasoning content if available
            if reasoning_content:
                logger.info(f"[Round {round_num}] Thinking: {reasoning_content[:500]}...")
                await emit({"type": "thinking", "round": round_num, "content": reasoning_content})
            
            # === OpenAI Function Calling Mode ===
            if not self.use_xml_protocol and tool_calls:
                # Append assistant message with tool_calls (preserving reasoning_content)
                msg_dict = raw_message.model_dump(exclude_none=True) if raw_message else {"role": "assistant", "content": content}
                if reasoning_content:
                    msg_dict['reasoning_content'] = reasoning_content
                messages.append(msg_dict)
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_result = await self._execute_function_call(tool_call)
                    logger.debug(f"Tool {tool_call.function.name} result: {tool_result[:200]}...")
                    
                    await emit({
                        "type": "tool",
                        "round": round_num,
                        "tool_name": tool_call.function.name,
                        "tool_args": tool_call.function.arguments,
                        "observation": tool_result[:1000],
                    })
                    
                    # Append tool result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                continue
            
            # === XML Protocol Mode ===
            content = self._strip_after_tool_response(content)
            
            if "<tool_call>" in content and "</tool_call>" in content:
                tool_block = content.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0]
                tool_result = await self._execute_xml_tool(tool_block)
                
                await emit({
                    "type": "tool",
                    "round": round_num,
                    "tool_call": tool_block,
                    "observation": tool_result[:1000],
                })
                
                # Combine tool_call and response into a single 'user' message
                messages.append({
                    "role": "user",
                    "content": (
                        f"<tool_call>\n{tool_block}\n</tool_call>\n"
                        f"{OBS_START}\n{tool_result}\n{OBS_END}"
                    )
                })
                continue

            # === Normal Response Path (no tool call) ===
            messages.append({"role": "assistant", "content": content})

            # Check for termination
            final = self._parse_answer(content)
            logger.debug(f"Final answer check: {final}")
            
            if final["answer"]:
                await emit({"type": "final", "round": round_num, "answer": final["answer"], "termination": "terminated with answer"})
                return {
                    "question": question,
                    "prediction": final["answer"],
                    "termination": "terminated with answer",
                    "trajectory": messages,
                }
            
            if final["terminate"]:
                best_effort = content.strip() or "Final answer generated by agent."
                await emit({"type": "final", "round": round_num, "answer": best_effort, "termination": "terminated without answer"})
                return {
                    "question": question,
                    "prediction": best_effort,
                    "termination": "terminated without answer",
                    "trajectory": messages,
                }
            
            # For function calling mode without tool calls, treat content as final answer
            if not self.use_xml_protocol and not tool_calls:
                await emit({"type": "final", "round": round_num, "answer": content, "termination": "function calling completed"})
                return {
                    "question": question,
                    "prediction": content,
                    "termination": "function calling completed",
                    "trajectory": messages,
                }
            
            # Prompt to continue (XML protocol mode)
            messages.append({
                "role": "user",
                "content": "Please continue your analysis or provide the final answer using <answer> tags."
            })

            # Last round fallback
            if remaining == 0:
                forced_prompt = (
                    "You have reached the limit. Stop tool calls. Provide the final response using "
                    "<answer> only. Do NOT include <tool_call> or <think>."
                )
                if self.instruction:
                    forced_prompt = f"{forced_prompt}\n\nRemember the task-specific instruction:\n{self.instruction}"
                
                messages.append({"role": "user", "content": forced_prompt})
                response = await self.call_server(messages, tools=None)  # No tools for final call
                content = response["content"]
                messages.append({"role": "assistant", "content": content})
                
                final = self._parse_answer(content)
                if final["answer"]:
                    await emit({"type": "final", "round": round_num, "answer": final["answer"], "termination": "terminated with answer (forced)"})
                    return {
                        "question": question,
                        "prediction": final["answer"],
                        "termination": "terminated with answer (forced)",
                        "trajectory": messages,
                    }
                
                fallback_text = content.strip() or "Final answer generated by agent."
                await emit({"type": "final", "round": round_num, "answer": fallback_text, "termination": "finalized without answer tag"})
                return {
                    "question": question,
                    "prediction": fallback_text,
                    "termination": "finalized without answer tag",
                    "trajectory": messages,
                }

        # Exhausted LLM calls
        forced_prompt = (
            "You have reached the limit. Stop tool calls. Provide the final response using "
            "<answer> only. Do NOT include <tool_call> or <think>."
        )
        if self.instruction:
            forced_prompt = f"{forced_prompt}\n\nRemember the task-specific instruction:\n{self.instruction}"
        
        messages.append({"role": "user", "content": forced_prompt})
        response = await self.call_server(messages, tools=None)
        content = response["content"]
        messages.append({"role": "assistant", "content": content})
        
        final = self._parse_answer(content)
        if final["answer"]:
            await emit({"type": "final", "answer": final["answer"], "termination": "terminated with answer (forced)"})
            return {
                "question": question,
                "prediction": final["answer"],
                "termination": "terminated with answer (forced)",
                "trajectory": messages,
            }
        
        fallback_text = content.strip() or "Final answer generated by agent."
        await emit({"type": "final", "answer": fallback_text, "termination": "exceed available llm calls"})
        return {
            "question": question,
            "prediction": fallback_text,
            "termination": "exceed available llm calls (finalized without answer tag)",
            "trajectory": messages,
        }

async def main():
    agent = ReactAgent()
    result = await agent.run("What is the capital of France?")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())