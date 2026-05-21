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
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Type, Union

from pydantic import BaseModel

from agentica.compression.tool_call_args import shrink_tool_call_arguments_json
from agentica.compression.tool_result_storage import maybe_persist_result
from agentica.model.message import Message
from agentica.prompts.compression import (
    DEFAULT_COMPRESSION_PROMPT,
    ITERATIVE_COMPRESSION_PROMPT,
)
from agentica.security.redact import redact_sensitive_text
from agentica.utils.log import logger
from agentica.utils.tokens import count_tokens

if TYPE_CHECKING:
    from agentica.workspace import Workspace


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
    compress_token_limit: Optional[int] = None
    compress_target_token_limit: Optional[int] = None
    truncate_head_chars: int = 150
    keep_recent_rounds: int = 3
    use_llm_compression: bool = False
    compress_tool_call_instructions: Optional[str] = None
    workspace: Optional["Workspace"] = None  # Workspace instance for archiving dropped messages

    stats: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Auto-compact state (Layer 2 -- mirrors CC's circuit-breaker)
    # ------------------------------------------------------------------
    _consecutive_auto_compact_failures: int = field(init=False, default=0)
    _max_auto_compact_failures: int = field(init=False, default=3)
    # Buffer tokens reserved for the compaction summary output (CC uses 13_000).
    _auto_compact_buffer_tokens: int = field(init=False, default=13_000)

    # Iterative summary for *conversation-level* auto_compact only.
    # Tool-result compression does NOT use iterative summaries because it runs
    # concurrently via asyncio.gather() — sharing mutable state would race.
    # Pattern borrowed from hermes-agent ContextCompressor.
    _conversation_previous_summary: Optional[str] = field(init=False, default=None, repr=False)

    def reset_run_state(self) -> None:
        """Reset per-run state. Call at the start of each agent run to prevent
        circuit breaker and stats from leaking across runs."""
        self._consecutive_auto_compact_failures = 0
        self._conversation_previous_summary = None

    def __post_init__(self):
        # Default target: 60% of trigger threshold
        if self.compress_target_token_limit is None and self.compress_token_limit is not None:
            self.compress_target_token_limit = int(self.compress_token_limit * 0.6)

    def _resolve_limits(self, model: Optional[Any] = None) -> None:
        """Auto-resolve compress_token_limit and target from model.context_window if not set."""
        if self.compress_token_limit is not None:
            return
        context_window = model.context_window if model is not None else None
        if context_window:
            self.compress_token_limit = int(context_window * 0.8)
            self.compress_target_token_limit = int(context_window * 0.5)

    def should_compress(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Check if compression should be triggered (pure token-based)."""
        if not self.compress_tool_results:
            return False

        # Auto-resolve limits from model context_window
        self._resolve_limits(model)

        # Token-based threshold only
        if self.compress_token_limit is not None:
            model_id = model.id if model else 'gpt-4o'
            tokens = count_tokens(messages, tools, model_id, response_format)
            if tokens >= self.compress_token_limit:
                _window = model.context_window if model is not None else None
                _window_str = f"{_window:,}" if _window else "?"
                logger.debug(
                    f"Compression triggered: {tokens:,}/{_window_str} tokens "
                    f"(threshold={self.compress_token_limit:,})"
                )
                return True

        return False

    def _shrink_assistant_tool_call_arguments(self, messages: List["Message"]) -> int:
        """Shrink large assistant tool-call argument strings without invalidating JSON."""
        shrunk = 0
        for msg in messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue
            for tool_call in msg.tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict):
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        next_arguments = shrink_tool_call_arguments_json(
                            arguments, max_string_chars=self.truncate_head_chars,
                        )
                        if next_arguments != arguments:
                            function["arguments"] = next_arguments
                            shrunk += 1
                arguments = tool_call.get("arguments")
                if isinstance(arguments, str):
                    next_arguments = shrink_tool_call_arguments_json(
                        arguments, max_string_chars=self.truncate_head_chars,
                    )
                    if next_arguments != arguments:
                        tool_call["arguments"] = next_arguments
                        shrunk += 1
        if shrunk:
            self.stats["tool_call_args_shrunk"] = self.stats.get("tool_call_args_shrunk", 0) + shrunk
        return shrunk

    # -------------------------------------------------------------------------
    # Stage 1: Rule-based compression (free)
    # -------------------------------------------------------------------------

    def _truncate_oldest_tool_results(self, messages: List["Message"], user_id: Optional[str] = None) -> int:
        """Persist oldest uncompressed tool results to disk, replacing with preview.

        Uses tool_result_storage.maybe_persist_result() so the full content
        is saved to ~/.agentica/projects/.../ and the context keeps a 2KB
        preview + file path.  Falls back to simple truncation only if
        persistence fails.

        Processes from oldest to newest, skipping the most recent
        `keep_recent_rounds` assistant-tool round groups.

        Returns:
            Number of messages truncated/persisted.
        """
        # Identify tool result indices (oldest first)
        tool_indices = [i for i, m in enumerate(messages) if m.role == "tool" and not m.compressed_content]
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

            # Persist to disk + replace with preview (preserves full content)
            tool_use_id = msg.tool_call_id or f"rule_compact_{idx}"
            tool_name = msg.tool_name or "tool"
            new_content = maybe_persist_result(
                tool_name=tool_name,
                tool_use_id=tool_use_id,
                content=content_str,
                max_result_size_chars=self.truncate_head_chars,
                user_id=user_id,
            )

            msg.content = new_content
            msg.compressed_content = new_content
            truncated += 1
            self.stats["rule_truncated"] = self.stats.get("rule_truncated", 0) + 1
            self.stats["rule_truncated_saved_chars"] = (
                self.stats.get("rule_truncated_saved_chars", 0) + original_len - len(new_content)
            )
            logger.debug(f"Persisted tool result [{idx}]: {original_len} -> {len(new_content)} chars")

        return truncated

    async def _drop_old_messages(self, messages: List["Message"]) -> int:
        """
        Drop old messages (assistant + tool pairs) keeping only the most recent
        `keep_recent_rounds` rounds plus system and first user message.
        
        Before dropping, archives the messages to workspace if configured.
        
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

        # Archive dropped messages to workspace before deletion
        if self.workspace is not None:
            dropped_messages = messages[drop_start_idx:keep_from_idx]
            await self._archive_dropped_messages(dropped_messages)

        # Build new message list
        kept_tail = messages[keep_from_idx:]
        messages.clear()
        messages.extend(preserved_head)
        messages.extend(kept_tail)

        self.stats["messages_dropped"] = self.stats.get("messages_dropped", 0) + dropped_count
        logger.debug(f"Dropped {dropped_count} old messages, kept {len(messages)} messages")
        return dropped_count

    async def _archive_dropped_messages(self, dropped_messages: List["Message"]) -> None:
        """Archive messages that are about to be dropped to the workspace conversation archive.

        Awaits the archive call directly to prevent silent cancellation in
        run_stream_sync() scenarios (C-01 fix).
        """
        if self.workspace is None:
            return
        try:
            archive_msgs = []
            for msg in dropped_messages:
                content = msg.content if msg.content else ""
                if not isinstance(content, str):
                    content = str(content)
                if content:
                    archive_msgs.append({
                        "role": msg.role or "unknown",
                        "content": content[:1000],  # Limit size for archive
                    })
            if archive_msgs:
                await self.workspace.archive_conversation(
                    archive_msgs, session_id="compression-archive"
                )
                self.stats["messages_archived"] = (
                    self.stats.get("messages_archived", 0) + len(archive_msgs)
                )
                logger.debug(f"Archived {len(archive_msgs)} dropped messages to workspace")
        except Exception as e:
            logger.warning(f"Failed to archive dropped messages: {e}")

    # -------------------------------------------------------------------------
    # Stage 2: LLM-based compression (optional)
    # -------------------------------------------------------------------------

    async def _compress_tool_result_llm(self, tool_result: "Message") -> Optional[str]:
        """Compress a single tool result using LLM.

        This runs concurrently (asyncio.gather) for multiple tool results,
        so it must be stateless — no shared mutable summary state.
        Always uses the user-provided or default compression prompt.
        """
        if not tool_result or not self.model:
            return None

        tool_name = tool_result.tool_name or 'unknown'
        content = str(tool_result.content or "")
        tool_content = f"Tool: {tool_name}\n{redact_sensitive_text(content)}"

        compression_prompt = self.compress_tool_call_instructions or DEFAULT_COMPRESSION_PROMPT

        try:
            response = await self.model.response(
                messages=[
                    Message(role="system", content=compression_prompt),
                    Message(role="user", content="Tool Results to Compress: " + tool_content),
                ]
            )
            summary_text = response.content if hasattr(response, 'content') else str(response)
            return redact_sensitive_text(str(summary_text))
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
            if msg.role == "tool" and not msg.compressed_content and i < protect_from_idx
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
    # Tool-pair sanitization
    # -------------------------------------------------------------------------

    def _sanitize_tool_pairs(self, messages: List["Message"]) -> List["Message"]:
        """Ensure every tool_call has a matching tool result and vice versa.

        Rebuilds the message list in-order: for each assistant message with
        tool_calls, inserts any existing tool results (matched by call_id) or
        a placeholder, in the *original tool_calls order*. This preserves the
        provider's hard constraint that tool results immediately follow their
        assistant tool_call message.

        Orphan tool results (no matching assistant tool_call) are dropped.
        """
        # Index existing tool results by call_id for O(1) lookup
        result_by_id: Dict[str, "Message"] = {}
        for msg in messages:
            if msg.role == "tool" and msg.tool_call_id:
                result_by_id[msg.tool_call_id] = msg

        # Collect all call_ids that assistant messages reference
        all_call_ids: set = set()
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id:
                        all_call_ids.add(tc_id)

        rebuilt: List["Message"] = []
        placeholder_count = 0
        orphan_count = 0

        for msg in messages:
            if msg.role == "tool":
                # Tool results are placed by the assistant-loop below;
                # drop orphans (result whose call_id has no matching assistant)
                if msg.tool_call_id not in all_call_ids:
                    orphan_count += 1
                # Skip here — results are re-inserted after their assistant msg
                continue

            rebuilt.append(msg)

            # After an assistant with tool_calls, insert results in call order
            if msg.role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if not tc_id:
                        continue
                    if tc_id in result_by_id:
                        rebuilt.append(result_by_id[tc_id])
                    else:
                        rebuilt.append(Message(
                            role="tool",
                            tool_call_id=tc_id,
                            content="[Tool result removed during compression]",
                        ))
                        placeholder_count += 1

        if orphan_count:
            logger.debug(f"Removed {orphan_count} orphan tool results")
        if placeholder_count:
            logger.debug(f"Added {placeholder_count} placeholder tool results")

        return rebuilt

    # -------------------------------------------------------------------------
    # Main compress entry point
    # -------------------------------------------------------------------------

    async def compress(
        self,
        messages: List["Message"],
        tools: Optional[List] = None,
        model: Optional[Any] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        trigger: str = "manual",
        task_anchor: Optional[Any] = None,
        user_id: Optional[str] = None,
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

        # Count tokens before compression
        _model_id = model.id if model else 'gpt-4o'
        _before_tokens = count_tokens(messages, tools, _model_id, response_format)
        _messages_before = len(messages)
        _stats_before = dict(self.stats)

        # Keep provider-facing tool call arguments valid while reducing huge payloads.
        tool_call_args_shrunk = self._shrink_assistant_tool_call_arguments(messages)

        # Stage 1a: Truncate oldest tool results
        truncated = self._truncate_oldest_tool_results(messages, user_id=user_id)
        if truncated:
            logger.debug(f"Stage 1a: Truncated {truncated} old tool results")

        # Check if still over limit
        if self._still_over_limit(messages, tools, model, response_format):
            # Stage 1b: Drop old messages
            dropped = await self._drop_old_messages(messages)
            if dropped:
                logger.debug(f"Stage 1b: Dropped {dropped} old messages")

        # Stage 2: LLM compression (optional)
        if self.use_llm_compression and self._still_over_limit(messages, tools, model, response_format):
            llm_count = await self._llm_compress_old_tool_results(messages)
            if llm_count:
                logger.debug(f"Stage 2: LLM compressed {llm_count} tool results")

        # Sanitize orphaned tool_call/result pairs after compression
        sanitized = self._sanitize_tool_pairs(list(messages))
        messages.clear()
        messages.extend(sanitized)

        # Log compression result
        _after_tokens = count_tokens(messages, tools, _model_id, response_format)
        task_anchor_preserved: Optional[bool] = None
        if task_anchor is not None and task_anchor.source_query:
            task_anchor_preserved = any(
                task_anchor.source_query in str(msg.content or "")
                for msg in messages
            )
        report = {
            "trigger": trigger,
            "messages_before": _messages_before,
            "messages_after": len(messages),
            "tokens_before": _before_tokens,
            "tokens_after": _after_tokens,
            "tool_results_pruned": truncated,
            "messages_dropped": self.stats.get("messages_dropped", 0) - _stats_before.get("messages_dropped", 0),
            "llm_summary_used": self.stats.get("llm_compressed", 0) > _stats_before.get("llm_compressed", 0),
            "task_anchor_preserved": task_anchor_preserved,
            "tool_call_args_shrunk": tool_call_args_shrunk,
        }
        self.stats["last_report"] = report
        _window = model.context_window if model is not None else None
        if _after_tokens < _before_tokens:
            _window_str = f"{_window:,}" if _window else "?"
            logger.debug(
                f"Compression: {_before_tokens:,} → {_after_tokens:,} tokens "
                f"(context_window={_window_str})"
            )

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
        model_id = model.id if model else 'gpt-4o'
        tokens = count_tokens(messages, tools, model_id, response_format)
        over = tokens >= target
        if over:
            logger.debug(f"Still over target: {tokens} >= {target}")
        return over

    # -------------------------------------------------------------------------
    # Layer 2: auto_compact — LLM-summarise when context approaches the limit
    # -------------------------------------------------------------------------

    def _should_auto_compact(self, messages: List["Message"], model: Optional[Any] = None) -> bool:
        """Return True when token count is within _auto_compact_buffer_tokens of the context window."""
        context_window = model.context_window if model is not None else None
        if context_window is None:
            return False
        threshold = context_window - self._auto_compact_buffer_tokens
        model_id = model.id if model else 'gpt-4o'
        tokens = count_tokens(messages, None, model_id, None)
        over = tokens >= threshold
        if over:
            logger.debug(
                f"Auto-compact threshold hit: {tokens:,} tokens "
                f">= {threshold:,} (window={context_window:,})"
            )
        return over

    async def _summarise_conversation(
        self,
        messages: List["Message"],
        model: Optional[Any] = None,
        custom_instructions: Optional[str] = None,
    ) -> Optional[str]:
        """Use an LLM to summarise the conversation for continuity.

        Args:
            messages: Conversation messages to summarise.
            model: LLM instance to use (falls back to self.model).
            custom_instructions: Optional user-provided instructions appended to prompt.
        """
        active_model = model or self.model
        if active_model is None:
            return None

        # Adaptive limits based on model context window.
        # Reserve ~50% of context for the summary input (the other 50% for
        # prompt overhead + output tokens).  Fallback: 80K chars (~20K tokens).
        context_window = active_model.context_window or 200_000
        # 1 token ~ 4 chars; use half of context window for the conversation dump
        max_total_chars = min(context_window * 4 // 2, 400_000)
        # Per-message truncation: distribute budget across messages, floor 1000, cap 8000
        per_msg_chars = max(1000, min(8000, max_total_chars // max(len(messages), 1)))

        text = json.dumps(
            [
                {
                    "role": m.role,
                    "content": str(redact_sensitive_text(str(m.content or "")))[:per_msg_chars],
                }
                for m in messages
            ],
            ensure_ascii=False,
        )[:max_total_chars]

        prompt_parts = []

        # Iterative summary: if we have a previous summary, ask LLM to UPDATE
        # it with new turns rather than regenerating from scratch. This preserves
        # accumulated knowledge across multiple compressions.
        if self._conversation_previous_summary:
            prompt_parts.extend([
                "You are updating an existing conversation summary with new turns.",
                "",
                "## Previous Summary",
                redact_sensitive_text(self._conversation_previous_summary),
                "",
                "## New Turns to Integrate",
                text,
                "",
                "Update the summary to incorporate new information.",
                "Preserve: key decisions, file paths, progress, next steps.",
                "Remove: outdated progress, superseded decisions.",
                "",
                "Your updated summary MUST include:",
            ])
        else:
            prompt_parts.extend([
                "Create a detailed summary of the conversation so far for continuity.",
                "",
                "Your summary MUST include:",
            ])

        prompt_parts.extend([
            "1. Primary Request and Intent: the user's explicit goals and requirements",
            "2. Key Technical Concepts: important technical details, APIs, patterns discussed",
            "3. Files and Code: specific files, functions, code sections mentioned or modified",
            "4. Completed Steps: what has been done, decisions made, problems solved",
            "5. Pending Tasks: remaining work, next steps, open questions",
            "6. Important Facts: numbers, URLs, IDs, configurations, error messages discovered",
            "",
            "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
        ])
        if custom_instructions:
            prompt_parts.append("")
            prompt_parts.append(f"Additional instructions: {redact_sensitive_text(custom_instructions)}")

        if not self._conversation_previous_summary:
            prompt_parts.append("")
            prompt_parts.append("Conversation to summarise:")
            prompt_parts.append(text)

        try:
            resp = await active_model.invoke([
                Message(role="user", content="\n".join(prompt_parts))
            ])
        except Exception as e:
            logger.warning(f"Summarisation LLM call failed: {e}")
            return None
        # Extract text from common response shapes
        summary: Optional[str] = None
        if hasattr(resp, "choices") and resp.choices:
            try:
                summary = resp.choices[0].message.content
            except (AttributeError, IndexError):
                pass
        if summary is None and hasattr(resp, "content") and isinstance(resp.content, str):
            summary = resp.content
        if summary is None and isinstance(resp, str):
            summary = resp
        if summary is None and resp:
            summary = str(resp)

        # Store for iterative updates on next compression
        if summary:
            summary = redact_sensitive_text(summary)
            self._conversation_previous_summary = summary
        return summary

    async def auto_compact(
        self,
        messages: List["Message"],
        model: Optional[Any] = None,
        force: bool = False,
        working_memory: Optional[Any] = None,
        custom_instructions: Optional[str] = None,
    ) -> bool:
        """Layer 2 compaction: LLM-summarise when context is near the limit.

        Mirrors CC's autoCompactIfNeeded():
        - Circuit breaker: stops after _max_auto_compact_failures consecutive
          failures to avoid wasting API calls.
        - If WorkingMemory has an existing session summary, reuses it directly
          without calling LLM (SM-compact optimization: faster + cheaper).
        - Saves a transcript to .transcripts/ before replacing messages.
        - Replaces all messages with a two-message [compressed] context.

        Args:
            messages: Current message list (mutated in-place on success).
            model:    The active LLM instance (used for token counting + summary).
            force:    If True, bypass threshold check (reactive compact path).
            working_memory: Optional WorkingMemory instance. When its session
                summary is available, it is used directly instead of calling LLM.
            custom_instructions: Optional user-provided instructions for summarisation.

        Returns:
            True if compaction occurred, False otherwise.
        """
        # Circuit breaker
        if self._consecutive_auto_compact_failures >= self._max_auto_compact_failures:
            logger.debug(
                f"Auto-compact circuit breaker: "
                f"{self._consecutive_auto_compact_failures} consecutive failures, skipping"
            )
            return False

        if not force and not self._should_auto_compact(messages, model):
            return False

        logger.debug("Auto-compact triggered: summarising conversation")


        # SM-compact optimization: reuse existing WorkingMemory session summary
        # when available (avoids LLM call, faster + cheaper).
        # Skip SM-compact when custom_instructions are provided (user wants custom summary).
        summary: Optional[str] = None
        if not custom_instructions and working_memory is not None and working_memory.summary is not None:
            sm = working_memory.summary
            summary = sm.summary
            if sm.topics:
                summary += f"\n\nTopics covered: {', '.join(sm.topics)}"
            logger.debug("Auto-compact: reusing WorkingMemory session summary (SM-compact)")

        if summary is None:
            summary = await self._summarise_conversation(messages, model, custom_instructions)

        if not summary:
            self._consecutive_auto_compact_failures += 1
            logger.warning(
                f"Auto-compact: summarisation failed "
                f"({self._consecutive_auto_compact_failures}/{self._max_auto_compact_failures})"
            )
            return False

        # Replace message list in-place
        messages.clear()
        messages.append(Message(role="user",
                                content=f"[Context compressed]\n\n{summary}"))
        messages.append(Message(role="assistant",
                                content="Understood. I have the conversation context. Continuing."))

        self._consecutive_auto_compact_failures = 0
        self.stats["auto_compact_count"] = self.stats.get("auto_compact_count", 0) + 1
        logger.debug(f"Auto-compact complete, messages reduced to {len(messages)}")

        # Write compact boundary to JSONL session log (if configured)
        try:
            if model is not None:
                _agent_ref = model._agent_ref
                _agent = _agent_ref() if _agent_ref else None
                if _agent is not None:
                    _slog = _agent._session_log
                    if _slog is not None:
                        _slog.append_compact_boundary(summary)
                        logger.debug("Compact boundary written to session log")
        except Exception as cb_err:
            logger.warning(f"Failed to write compact boundary: {cb_err}")

        return True

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
