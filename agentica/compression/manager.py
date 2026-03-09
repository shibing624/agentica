# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Compression manager for compressing tool call results to save context space.

Supports two compression strategies (applied in order):
1. Rule-based compression (default, free):
   - Truncate oldest tool results to head N characters
   - If still over limit, drop oldest messages keeping only recent N rounds
2. LLM-based compression (optional, costs money):
   - Use a lightweight LLM to intelligently summarize tool results
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
    - Specific facts: numbers, statistics, amounts, prices, quantities, metrics
    - Temporal data: dates, times, timestamps (use short format: "Oct 21 2025")
    - Entities: people, companies, products, locations, organizations
    - Identifiers: URLs, IDs, codes, technical identifiers, versions
    - Key quotes, citations, sources (if relevant to agent's task)
    
    COMPRESS TO ESSENTIALS:
    - Descriptions: keep only key attributes
    - Explanations: distill to core insight
    - Lists: focus on most relevant items based on agent context
    - Background: minimal context only if critical
    
    REMOVE ENTIRELY:
    - Introductions, conclusions, transitions
    - Hedging language ("might", "possibly", "appears to")
    - Meta-commentary ("According to", "The results show")
    - Formatting artifacts (markdown, HTML, JSON structure)
    - Redundant or repetitive information
    - Generic background not relevant to agent's task
    - Promotional language, filler words
    
    Be concise while retaining all critical facts.
    """)

# Truncation marker appended to truncated content
_TRUNCATED_MARKER = "\n...[truncated]"


@dataclass
class CompressionManager:
    """
    Manager for compressing tool call results to save context space.
    
    Two-stage compression strategy (applied in order):
    
    Stage 1 - Rule-based (free, always runs first):
        a) Truncate oldest uncompressed tool results to `truncate_head_chars` characters
        b) If still over token limit, drop oldest messages keeping `keep_recent_rounds` rounds
    
    Stage 2 - LLM-based (optional, costs money):
        Use `model` to intelligently summarize tool results
    
    Args:
        model: The model used for LLM compression (e.g. gpt-4o-mini). None = disable LLM compression
        compress_tool_results: Whether to enable compression
        compress_token_limit: Token count threshold for triggering compression (e.g. context_window * 0.8).
            If None, auto-resolved from model.context_window * 0.8 at runtime
        compress_target_token_limit: Target token count after compression (e.g. context_window * 0.5).
            If None, auto-resolved from model.context_window * 0.5, or compress_token_limit * 0.6
        compress_tool_results_limit: Number of uncompressed tool results before triggering compression
        truncate_head_chars: Max characters to keep per tool result in rule-based truncation (default 150)
        keep_recent_rounds: Number of recent assistant-tool rounds to preserve when dropping old messages (default 3)
        use_llm_compression: Whether to enable LLM-based compression as Stage 2 (default False)
        compress_tool_call_instructions: Custom LLM compression prompt
    
    Example:
        ```python
        from agentica.compression import CompressionManager
        from agentica.model.openai import OpenAIChat
        
        # Zero config: auto-resolve from model.context_window (80% trigger, 50% target)
        cm = CompressionManager()
        
        # Explicit limits: trigger at 80k tokens, compress down to 50k
        cm = CompressionManager(
            compress_token_limit=80000,
            compress_target_token_limit=50000,
        )
        
        # Rule-based + LLM fallback
        cm = CompressionManager(
            model=OpenAIChat(id="gpt-4o-mini"),
            compress_token_limit=80000,
            compress_target_token_limit=50000,
            use_llm_compression=True,
        )
        ```
    """
    model: Optional[Any] = None
    compress_tool_results: bool = True
    compress_tool_results_limit: Optional[int] = None
    compress_token_limit: Optional[int] = None
    compress_target_token_limit: Optional[int] = None
    truncate_head_chars: int = 150
    keep_recent_rounds: int = 3
    use_llm_compression: bool = False
    compress_tool_call_instructions: Optional[str] = None

    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.compress_tool_results_limit is None and self.compress_token_limit is None:
            self.compress_tool_results_limit = 3
        # Default target: 60% of trigger threshold
        if self.compress_target_token_limit is None and self.compress_token_limit is not None:
            self.compress_target_token_limit = int(self.compress_token_limit * 0.6)

    def _resolve_limits(self, model: Optional[Any] = None) -> None:
        """Auto-resolve compress_token_limit and target from model.context_window if not set."""
        if self.compress_token_limit is not None:
            return
        context_window = getattr(model, 'context_window', None)
        if context_window:
            self.compress_token_limit = int(context_window * 0.8)
            self.compress_target_token_limit = int(context_window * 0.5)
            logger.info(
                f"Auto-set compress limits from context_window={context_window}: "
                f"trigger={self.compress_token_limit}, target={self.compress_target_token_limit}"
            )

    def should_compress(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Check if compression should be triggered."""
        if not self.compress_tool_results:
            return False

        # Auto-resolve limits from model context_window
        self._resolve_limits(model)

        # Token-based threshold
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

        # Count-based threshold
        if self.compress_tool_results_limit is not None:
            uncompressed_count = sum(
                1 for m in messages
                if m.role == "tool" and not getattr(m, 'compressed_content', None)
            )
            if uncompressed_count >= self.compress_tool_results_limit:
                logger.info(f"Tool count limit hit: {uncompressed_count} >= {self.compress_tool_results_limit}")
                return True

        return False

    # -------------------------------------------------------------------------
    # Stage 1: Rule-based compression (free)
    # -------------------------------------------------------------------------

    def _truncate_oldest_tool_results(self, messages: List["Message"]) -> int:
        """
        Truncate oldest uncompressed tool result messages to `truncate_head_chars`.
        
        Processes from oldest to newest, skipping the most recent `keep_recent_rounds`
        assistant-tool round groups.
        
        Returns:
            Number of messages truncated.
        """
        # Identify tool result indices (oldest first)
        tool_indices = [i for i, m in enumerate(messages) if m.role == "tool" and not getattr(m, 'compressed_content', None)]
        if not tool_indices:
            return 0

        # Identify recent round boundaries: count assistant messages from the end
        # Each "round" = one assistant message + its following tool messages
        assistant_count = 0
        protect_from_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and messages[i].tool_calls:
                assistant_count += 1
                if assistant_count >= self.keep_recent_rounds:
                    protect_from_idx = i
                    break

        truncated = 0
        for idx in tool_indices:
            if idx >= protect_from_idx:
                break  # Don't truncate recent rounds
            msg = messages[idx]
            content_str = str(msg.content) if msg.content else ""
            if len(content_str) <= self.truncate_head_chars:
                continue
            original_len = len(content_str)
            truncated_content = content_str[:self.truncate_head_chars] + _TRUNCATED_MARKER
            msg.content = truncated_content
            msg.compressed_content = truncated_content
            truncated += 1
            self.stats["rule_truncated"] = self.stats.get("rule_truncated", 0) + 1
            self.stats["rule_truncated_saved_chars"] = (
                self.stats.get("rule_truncated_saved_chars", 0) + original_len - len(truncated_content)
            )
            logger.debug(f"Truncated tool result [{idx}]: {original_len} -> {len(truncated_content)} chars")

        return truncated

    def _drop_old_messages(self, messages: List["Message"]) -> int:
        """
        Drop old messages (assistant + tool pairs) keeping only the most recent
        `keep_recent_rounds` rounds plus system and first user message.
        
        Modifies the list in-place.
        
        Returns:
            Number of messages dropped.
        """
        # Identify round boundaries from the end
        # A "round" starts with an assistant message that has tool_calls
        round_start_indices = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and messages[i].tool_calls:
                round_start_indices.append(i)

        if len(round_start_indices) <= self.keep_recent_rounds:
            return 0  # Not enough rounds to drop

        # Determine the cutoff: keep from this index onwards
        keep_from_idx = round_start_indices[self.keep_recent_rounds - 1] if self.keep_recent_rounds > 0 else len(messages)

        # Always preserve: system messages at the start, and the first user message
        preserved_head = []
        first_user_found = False
        drop_start_idx = 0
        for i, msg in enumerate(messages):
            if msg.role == "system":
                preserved_head.append(msg)
                drop_start_idx = i + 1
            elif msg.role == "user" and not first_user_found:
                preserved_head.append(msg)
                first_user_found = True
                drop_start_idx = i + 1
                break
            else:
                break

        if keep_from_idx <= drop_start_idx:
            return 0  # Nothing to drop

        # Messages to drop: between preserved_head and keep_from_idx
        dropped_count = keep_from_idx - drop_start_idx
        if dropped_count <= 0:
            return 0

        # Build new message list
        kept_tail = messages[keep_from_idx:]
        messages.clear()
        messages.extend(preserved_head)
        messages.extend(kept_tail)

        self.stats["messages_dropped"] = self.stats.get("messages_dropped", 0) + dropped_count
        logger.info(f"Dropped {dropped_count} old messages, kept {len(messages)} messages")
        return dropped_count

    # -------------------------------------------------------------------------
    # Stage 2: LLM-based compression (optional)
    # -------------------------------------------------------------------------

    async def _compress_tool_result_llm(self, tool_result: "Message") -> Optional[str]:
        """Compress a single tool result using LLM."""
        if not tool_result or not self.model:
            return None

        tool_name = getattr(tool_result, 'tool_name', None) or 'unknown'
        tool_content = f"Tool: {tool_name}\n{tool_result.content}"
        compression_prompt = self.compress_tool_call_instructions or DEFAULT_COMPRESSION_PROMPT

        try:
            from agentica.model.message import Message
            response = await self.model.response(
                messages=[
                    Message(role="system", content=compression_prompt),
                    Message(role="user", content="Tool Results to Compress: " + tool_content),
                ]
            )
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM compression failed: {e}")
            return None

    async def _llm_compress_old_tool_results(self, messages: List["Message"]) -> int:
        """
        LLM-compress uncompressed tool results (oldest first, skip recent rounds).
        
        Returns:
            Number of messages compressed by LLM.
        """
        if not self.model:
            return 0

        # Collect uncompressed tool messages, skip recent rounds
        assistant_count = 0
        protect_from_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "assistant" and messages[i].tool_calls:
                assistant_count += 1
                if assistant_count >= self.keep_recent_rounds:
                    protect_from_idx = i
                    break

        targets = [
            msg for i, msg in enumerate(messages)
            if msg.role == "tool" and not getattr(msg, 'compressed_content', None) and i < protect_from_idx
        ]
        if not targets:
            return 0

        original_sizes = [len(str(msg.content)) if msg.content else 0 for msg in targets]
        results = await asyncio.gather(*[self._compress_tool_result_llm(msg) for msg in targets])

        compressed_count = 0
        for msg, compressed, original_len in zip(targets, results, original_sizes):
            if compressed:
                msg.compressed_content = compressed
                msg.content = compressed
                compressed_count += 1
                self.stats["llm_compressed"] = self.stats.get("llm_compressed", 0) + 1
                self.stats["llm_original_size"] = self.stats.get("llm_original_size", 0) + original_len
                self.stats["llm_compressed_size"] = self.stats.get("llm_compressed_size", 0) + len(compressed)
                logger.debug(f"LLM compressed tool result: {original_len} -> {len(compressed)} chars")

        return compressed_count

    # -------------------------------------------------------------------------
    # Main compress entry point
    # -------------------------------------------------------------------------

    async def compress(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> None:
        """
        Run compression pipeline on messages (in-place).
        
        Stage 1: Rule-based (always runs)
          a) Truncate oldest tool results to head chars
          b) If still over limit, drop old messages
        Stage 2: LLM-based (only if use_llm_compression=True and still over limit)
        """
        if not self.compress_tool_results:
            return

        # Stage 1a: Truncate oldest tool results
        truncated = self._truncate_oldest_tool_results(messages)
        if truncated:
            logger.info(f"Stage 1a: Truncated {truncated} old tool results")

        # Check if still over limit
        if self._still_over_limit(messages, tools, model, response_format):
            # Stage 1b: Drop old messages
            dropped = self._drop_old_messages(messages)
            if dropped:
                logger.info(f"Stage 1b: Dropped {dropped} old messages")

        # Stage 2: LLM compression (optional)
        if self.use_llm_compression and self._still_over_limit(messages, tools, model, response_format):
            llm_count = await self._llm_compress_old_tool_results(messages)
            if llm_count:
                logger.info(f"Stage 2: LLM compressed {llm_count} tool results")

    def _still_over_limit(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Check if messages are still over the target token limit after compression."""
        target = self.compress_target_token_limit or self.compress_token_limit
        if target is None:
            return False
        try:
            from agentica.utils.tokens import count_tokens
            model_id = getattr(model, 'id', 'gpt-4o') if model else 'gpt-4o'
            tokens = count_tokens(messages, tools, model_id, response_format)
            over = tokens >= target
            if over:
                logger.debug(f"Still over target: {tokens} >= {target}")
            return over
        except Exception as e:
            logger.warning(f"Error counting tokens in limit check: {e}")
            return False

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_compression_ratio(self) -> float:
        """Get the overall compression ratio."""
        original = self.stats.get("llm_original_size", 0)
        compressed = self.stats.get("llm_compressed_size", 0)
        if original == 0:
            return 1.0
        return compressed / original

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self.stats,
            "compression_ratio": self.get_compression_ratio(),
        }
