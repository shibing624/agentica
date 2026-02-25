# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Shared metrics and stream data classes for all model providers.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any

from agentica.utils.log import logger
from agentica.utils.timer import Timer


@dataclass
class Metrics:
    """Unified metrics for tracking LLM response performance across all providers."""
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
        """Log metrics with safe divide-by-zero protection."""
        if self.time_to_first_token is not None:
            logger.debug(f"* Time to first token:         {self.time_to_first_token:.4f}s")
        elapsed = self.response_timer.elapsed or 0
        logger.debug(f"* Time to generate response:   {elapsed:.4f}s")
        output_tokens = self.output_tokens or self.completion_tokens or 0
        if elapsed > 0 and output_tokens > 0:
            tokens_per_second = output_tokens / elapsed
            logger.debug(f"* Tokens per second:           {tokens_per_second:.4f} tokens/s")
        else:
            logger.debug("* Tokens per second:           N/A")
        input_tokens = self.input_tokens or self.prompt_tokens or 0
        logger.debug(f"* Input tokens:                {input_tokens}")
        logger.debug(f"* Output tokens:               {output_tokens}")
        total = self.total_tokens or (input_tokens + output_tokens) or 0
        logger.debug(f"* Total tokens:                {total}")


@dataclass
class StreamData:
    """Data accumulated during streaming response (OpenAI-compatible format)."""
    response_content: str = ""
    response_reasoning_content: str = ""
    response_audio: Optional[dict] = None
    response_tool_calls: Optional[List[Any]] = None
