# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Compression manager for compressing tool call results to save context space.

This module provides functionality to automatically compress tool call results
when the context size exceeds certain thresholds, helping to manage token limits
in long conversations with many tool calls.
"""
import asyncio
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from agentica.utils.log import logger

DEFAULT_COMPRESSION_PROMPT = dedent("""\
    You are compressing tool call results to save context space while preserving critical information.
    
    Your goal: Extract only the essential information from the tool output.
    
    ALWAYS PRESERVE:
    • Specific facts: numbers, statistics, amounts, prices, quantities, metrics
    • Temporal data: dates, times, timestamps (use short format: "Oct 21 2025")
    • Entities: people, companies, products, locations, organizations
    • Identifiers: URLs, IDs, codes, technical identifiers, versions
    • Key quotes, citations, sources (if relevant to agent's task)
    
    COMPRESS TO ESSENTIALS:
    • Descriptions: keep only key attributes
    • Explanations: distill to core insight
    • Lists: focus on most relevant items based on agent context
    • Background: minimal context only if critical
    
    REMOVE ENTIRELY:
    • Introductions, conclusions, transitions
    • Hedging language ("might", "possibly", "appears to")
    • Meta-commentary ("According to", "The results show")
    • Formatting artifacts (markdown, HTML, JSON structure)
    • Redundant or repetitive information
    • Generic background not relevant to agent's task
    • Promotional language, filler words
    
    EXAMPLE:
    Input: "According to recent market analysis and industry reports, OpenAI has made several significant announcements in the technology sector. The company revealed ChatGPT Atlas on October 21, 2025, which represents a new AI-powered browser application that has been specifically designed for macOS users. This browser is strategically positioned to compete with traditional search engines in the market. Additionally, on October 6, 2025, OpenAI launched Apps in ChatGPT, which includes a comprehensive software development kit (SDK) for developers. The company has also announced several initial strategic partners who will be integrating with this new feature, including well-known companies such as Spotify, the popular music streaming service, Zillow, which is a real estate marketplace platform, and Canva, a graphic design platform."
    
    Output: "OpenAI - Oct 21 2025: ChatGPT Atlas (AI browser, macOS, search competitor); Oct 6 2025: Apps in ChatGPT + SDK; Partners: Spotify, Zillow, Canva"
    
    Be concise while retaining all critical facts.
    """)


@dataclass
class CompressionManager:
    """
    Manager for compressing tool call results to save context space.
    
    Compression can be triggered by:
    1. Token threshold: When context tokens exceed `compress_token_limit`
    2. Tool count threshold: When uncompressed tool calls exceed `compress_tool_results_limit`
    
    Args:
        model: The model used for compression (can be a lighter model like gpt-4o-mini)
        compress_tool_results: Whether to enable tool result compression
        compress_tool_results_limit: Number of tool results before triggering compression
        compress_token_limit: Token count threshold for triggering compression
        compress_tool_call_instructions: Custom compression prompt
    
    Example:
        ```python
        from agentica.compression import CompressionManager
        from agentica.model.openai import OpenAIChat
        
        # Simple usage
        compression_manager = CompressionManager(
            model=OpenAIChat(id="gpt-4o-mini"),
            compress_tool_results_limit=5,
        )
        
        # With token limit
        compression_manager = CompressionManager(
            model=OpenAIChat(id="gpt-4o-mini"),
            compress_token_limit=10000,
        )
        ```
    """
    model: Optional[Any] = None  # Model used for compression
    compress_tool_results: bool = True
    compress_tool_results_limit: Optional[int] = None
    compress_token_limit: Optional[int] = None
    compress_tool_call_instructions: Optional[str] = None

    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Default to 3 tool results if no limit is specified
        if self.compress_tool_results_limit is None and self.compress_token_limit is None:
            self.compress_tool_results_limit = 3

    def _is_tool_result_message(self, msg: "Message") -> bool:
        """Check if a message is a tool result message."""
        return msg.role == "tool"

    def should_compress(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """
        Check if tool results should be compressed.

        Args:
            messages: List of messages to check
            tools: List of tools for token counting
            model: The Agent model (for token counting)
            response_format: Output schema for accurate token counting
        
        Returns:
            True if compression should be triggered
        """
        if not self.compress_tool_results:
            return False

        # Token-based threshold check
        if self.compress_token_limit is not None:
            try:
                from agentica.utils.tokens import count_tokens
                model_id = getattr(model, 'id', 'gpt-4o') if model else 'gpt-4o'
                tokens = count_tokens(messages, tools, model_id, response_format)
                if tokens >= self.compress_token_limit:
                    logger.info(f"Token limit hit: {tokens} >= {self.compress_token_limit}")
                    return True
            except Exception as e:
                logger.warning(f"Error counting tokens: {e}")

        # Count-based threshold check
        if self.compress_tool_results_limit is not None:
            uncompressed_tools_count = len(
                [m for m in messages if self._is_tool_result_message(m) and not getattr(m, 'compressed_content', None)]
            )
            if uncompressed_tools_count >= self.compress_tool_results_limit:
                logger.info(f"Tool count limit hit: {uncompressed_tools_count} >= {self.compress_tool_results_limit}")
                return True

        return False

    def _compress_tool_result(self, tool_result: "Message") -> Optional[str]:
        """Compress a single tool result message."""
        if not tool_result:
            return None

        tool_name = getattr(tool_result, 'tool_name', None) or 'unknown'
        tool_content = f"Tool: {tool_name}\n{tool_result.content}"

        if not self.model:
            logger.warning("No compression model available")
            return None

        compression_prompt = self.compress_tool_call_instructions or DEFAULT_COMPRESSION_PROMPT
        compression_message = "Tool Results to Compress: " + tool_content + "\n"

        try:
            from agentica.model.message import Message
            response = self.model.response(
                messages=[
                    Message(role="system", content=compression_prompt),
                    Message(role="user", content=compression_message),
                ]
            )
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"Error compressing tool result: {e}")
            return tool_content

    def compress(self, messages: List["Message"]) -> None:
        """
        Compress uncompressed tool results in place.
        
        Args:
            messages: List of messages to process (modified in place)
        """
        if not self.compress_tool_results:
            return

        uncompressed_tools = [
            msg for msg in messages 
            if msg.role == "tool" and not getattr(msg, 'compressed_content', None)
        ]

        if not uncompressed_tools:
            return

        for tool_msg in uncompressed_tools:
            original_len = len(str(tool_msg.content)) if tool_msg.content else 0
            compressed = self._compress_tool_result(tool_msg)
            if compressed:
                tool_msg.compressed_content = compressed
                # Update stats
                tool_results_count = len(tool_msg.tool_calls) if tool_msg.tool_calls else 1
                self.stats["tool_results_compressed"] = (
                    self.stats.get("tool_results_compressed", 0) + tool_results_count
                )
                self.stats["original_size"] = self.stats.get("original_size", 0) + original_len
                self.stats["compressed_size"] = self.stats.get("compressed_size", 0) + len(compressed)
                logger.debug(f"Compressed tool result: {original_len} -> {len(compressed)} chars")
            else:
                logger.warning(f"Compression failed for {getattr(tool_msg, 'tool_name', 'unknown')}")

    # =============================================================================
    # Async Methods
    # =============================================================================

    async def ashould_compress(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """
        Async check if tool results should be compressed.

        Args:
            messages: List of messages to check
            tools: List of tools for token counting
            model: The Agent model (for token counting)
            response_format: Output schema for accurate token counting
        
        Returns:
            True if compression should be triggered
        """
        if not self.compress_tool_results:
            return False

        # Token-based threshold check
        if self.compress_token_limit is not None:
            try:
                from agentica.utils.tokens import count_tokens
                model_id = getattr(model, 'id', 'gpt-4o') if model else 'gpt-4o'
                tokens = count_tokens(messages, tools, model_id, response_format)
                if tokens >= self.compress_token_limit:
                    logger.info(f"Token limit hit: {tokens} >= {self.compress_token_limit}")
                    return True
            except Exception as e:
                logger.warning(f"Error counting tokens: {e}")

        # Count-based threshold check
        if self.compress_tool_results_limit is not None:
            uncompressed_tools_count = len(
                [m for m in messages if self._is_tool_result_message(m) and not getattr(m, 'compressed_content', None)]
            )
            if uncompressed_tools_count >= self.compress_tool_results_limit:
                logger.info(f"Tool count limit hit: {uncompressed_tools_count} >= {self.compress_tool_results_limit}")
                return True

        return False

    async def _acompress_tool_result(self, tool_result: "Message") -> Optional[str]:
        """Async compress a single tool result."""
        if not tool_result:
            return None

        tool_name = getattr(tool_result, 'tool_name', None) or 'unknown'
        tool_content = f"Tool: {tool_name}\n{tool_result.content}"

        if not self.model:
            logger.warning("No compression model available")
            return None

        compression_prompt = self.compress_tool_call_instructions or DEFAULT_COMPRESSION_PROMPT
        compression_message = "Tool Results to Compress: " + tool_content + "\n"

        try:
            from agentica.model.message import Message
            # Check if model has async response method
            if hasattr(self.model, 'aresponse'):
                response = await self.model.aresponse(
                    messages=[
                        Message(role="system", content=compression_prompt),
                        Message(role="user", content=compression_message),
                    ]
                )
            else:
                # Fallback to sync method
                response = self.model.response(
                    messages=[
                        Message(role="system", content=compression_prompt),
                        Message(role="user", content=compression_message),
                    ]
                )
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            logger.error(f"Error compressing tool result: {e}")
            return tool_content

    async def acompress(self, messages: List["Message"]) -> None:
        """
        Async compress uncompressed tool results using parallel processing.
        
        Args:
            messages: List of messages to process (modified in place)
        """
        if not self.compress_tool_results:
            return

        uncompressed_tools = [
            msg for msg in messages 
            if msg.role == "tool" and not getattr(msg, 'compressed_content', None)
        ]

        if not uncompressed_tools:
            return

        # Track original sizes before compression
        original_sizes = [len(str(msg.content)) if msg.content else 0 for msg in uncompressed_tools]

        # Parallel compression using asyncio.gather
        tasks = [self._acompress_tool_result(msg) for msg in uncompressed_tools]
        results = await asyncio.gather(*tasks)

        # Apply results and track stats
        for msg, compressed, original_len in zip(uncompressed_tools, results, original_sizes):
            if compressed:
                msg.compressed_content = compressed
                tool_results_count = len(msg.tool_calls) if msg.tool_calls else 1
                self.stats["tool_results_compressed"] = (
                    self.stats.get("tool_results_compressed", 0) + tool_results_count
                )
                self.stats["original_size"] = self.stats.get("original_size", 0) + original_len
                self.stats["compressed_size"] = self.stats.get("compressed_size", 0) + len(compressed)
                logger.debug(f"Compressed tool result: {original_len} -> {len(compressed)} chars")
            else:
                logger.warning(f"Compression failed for {getattr(msg, 'tool_name', 'unknown')}")

    def get_compression_ratio(self) -> float:
        """Get the compression ratio achieved so far."""
        original = self.stats.get("original_size", 0)
        compressed = self.stats.get("compressed_size", 0)
        if original == 0:
            return 1.0
        return compressed / original

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self.stats,
            "compression_ratio": self.get_compression_ratio(),
        }
