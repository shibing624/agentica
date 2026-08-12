# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Layer 2 compaction — LLM summarisation of the whole conversation.

Layer 1 (``agentica.compression.evict``) shrinks the request for free by
evicting old tool results. This is what happens when that is not enough: the
history is replaced by a summary, which costs an LLM call and is irreversible,
so it is deliberately the last thing tried.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.compression.evict import trailing_user_turn_start
from agentica.model.message import Message
from agentica.security.redact import redact_sensitive_text
from agentica.utils.log import logger
from agentica.utils.tokens import count_tokens


_ANTHROPIC_STREAMING_REQUIRED = "Streaming is required for operations that may take longer than 10 minutes"

# Fraction of the context window at which auto-compact fires. A ratio, not an
# absolute buffer: the codebase serves windows from 8K (gpt-4) to 1M, and any
# absolute value is wrong at both ends (negative on small windows — every turn
# "over threshold"; far too little headroom on 1M). 0.95 also buys time for
# tool results to land in prompt cache before the one paid summary resets it.
AUTO_COMPACT_THRESHOLD_RATIO = 0.95


def _covered_prefix_hash(msgs: List["Message"]) -> str:
    """Stable hash over the messages a summary replaces (role + content).

    Role and content only: timestamps and tool metadata churn between runs and
    would make the same conversation hash differently each time. Truncated to
    16 hex chars — it is a lineage marker, not a checksum adversaries attack.
    """
    h = hashlib.sha256()
    for m in msgs:
        try:
            content = m.get_content_string() or ""
        except Exception:
            content = str(m.content or "")
        h.update((m.role or "").encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(content.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]

@dataclass
class CompressionManager:
    """Summarise the conversation when the context approaches the window.

    Args:
        model: Model used for summarisation. Optional — callers normally pass
            the active model to :meth:`auto_compact`, which takes precedence.
        compress_token_limit: Token threshold. If None, auto-resolved from
            ``model.context_window * 0.8`` at runtime.
        compress_target_token_limit: Target after compaction. If None,
            auto-resolved from ``model.context_window * 0.5``.

    Example:
        ```python
        from agentica.compression import CompressionManager

        # Zero config: auto-resolve from model.context_window
        cm = CompressionManager()
        await cm.auto_compact(messages, model=model)
        ```
    """
    model: Optional[Any] = None
    compress_token_limit: Optional[int] = None
    compress_target_token_limit: Optional[int] = None

    stats: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Auto-compact state (mirrors CC's circuit-breaker)
    # ------------------------------------------------------------------
    _consecutive_auto_compact_failures: int = field(init=False, default=0)
    _max_auto_compact_failures: int = field(init=False, default=3)

    # Carried across compactions so each one updates the previous summary
    # instead of regenerating from an already-summarised transcript.
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

    def should_native_compact(
        self,
        messages: List["Message"],
        model: Any,
        tools: Optional[List] = None,
    ) -> bool:
        """Trigger provider compaction before local destructive stages."""
        if not model.supports_native_compaction:
            return False
        self._resolve_limits(model)
        provider_limit = model.native_compaction_token_limit()
        threshold = provider_limit
        if self.compress_token_limit is not None:
            threshold = min(threshold, self.compress_token_limit)
        tokens = model.estimate_native_compaction_tokens(messages, tools)
        if tokens < threshold:
            return False
        logger.debug(
            f"Native compact threshold hit: {tokens:,} >= {threshold:,} "
            f"(provider_limit={provider_limit:,})"
        )
        return True

    # -------------------------------------------------------------------------
    # Layer 2: auto_compact — LLM-summarise when context approaches the limit
    # -------------------------------------------------------------------------

    def should_auto_compact(
        self,
        messages: List["Message"],
        model: Optional[Any] = None,
        context_tokens: Optional[int] = None,
    ) -> bool:
        """Return True once token count reaches AUTO_COMPACT_THRESHOLD_RATIO of the window.

        Public because the decision has a side effect outside this class: the
        runner must fire ``on_pre_compact`` (which flushes memory buffers
        through an auxiliary LLM) only on turns that actually compact. Pass
        ``context_tokens`` when the caller already counted the request, so the
        gate and the compaction agree on one number instead of tokenising the
        transcript twice per turn.
        """
        context_window = model.context_window if model is not None else None
        if context_window is None:
            return False
        threshold = int(context_window * AUTO_COMPACT_THRESHOLD_RATIO)
        model_id = model.id if model else 'gpt-4o'
        tokens = context_tokens if context_tokens is not None else count_tokens(messages, None, model_id, None)
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
            if _ANTHROPIC_STREAMING_REQUIRED not in str(e):
                logger.warning(f"Summarisation LLM call failed: {e}")
                return None
            try:
                summary = await self._summarise_conversation_stream(
                    active_model,
                    "\n".join(prompt_parts),
                )
            except Exception as stream_error:
                logger.warning(f"Summarisation streaming LLM call failed: {stream_error}")
                return None
            if summary:
                summary = redact_sensitive_text(summary)
                self._conversation_previous_summary = summary
            return summary
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

    async def _summarise_conversation_stream(self, active_model: Any, prompt: str) -> Optional[str]:
        """Collect a streaming summary for Anthropic long-output requests."""
        chunks: List[str] = []
        async for chunk in active_model.response_stream([
            Message(role="user", content=prompt)
        ]):
            content = chunk.content
            if isinstance(content, str):
                chunks.append(content)
        summary = "".join(chunks).strip()
        return summary or None

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

        if not force and not self.should_auto_compact(messages, model):
            return False

        # INFO: auto-compact is a once-per-many-rounds transition that
        # changes how every later turn looks (different message stack, often
        # slower first reply). Operators need it visible without DEBUG noise.
        logger.info("Auto-compact triggered: summarising conversation")


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
            # Summarisation is a full LLM round-trip and routinely takes 10-20s
            # on a large context. Bracket it with events so the CLI can say why
            # the spinner is sitting still instead of looking hung. The SM-compact
            # branch above is instant and needs no announcement.
            agent = self._agent_of(model)
            cb = agent._event_callback if agent is not None else None
            agent_name = (agent.name or "Agent") if agent is not None else "Agent"
            if cb is not None:
                cb({"type": "compact.start", "agent_name": agent_name, "stage": "auto"})
            try:
                summary = await self._summarise_conversation(messages, model, custom_instructions)
            finally:
                if cb is not None:
                    cb({"type": "compact.end", "agent_name": agent_name, "stage": "auto"})

        if not summary:
            self._consecutive_auto_compact_failures += 1
            logger.warning(
                f"Auto-compact: summarisation failed "
                f"({self._consecutive_auto_compact_failures}/{self._max_auto_compact_failures})"
            )
            return False

        # Replace message list in-place, preserving two things a blind clear
        # would destroy:
        #
        # * The system prompt — it carries the instructions, tool guidance and
        #   workspace context, and the rest of the run would otherwise be sent
        #   with none of them. Stage 3 (`compress`) keeps it via
        #   `preserved_head`; stage 4 must match.
        # * The trailing turn, from the last user message on. That message is
        #   the request still awaiting an answer; dropping it leaves the
        #   conversation ending on an assistant turn, which the model is then
        #   asked to continue from (providers reject the prefill outright).
        #   Keeping the whole tail also keeps tool_calls paired with their
        #   results. The summary covers this span too, so nothing is lost when
        #   the tail turns out to be an injected notice rather than the question.
        #
        #   "User message" here means one the *user* sent. Anthropic delivers a
        #   tool round as a user message full of tool_result blocks, and cutting
        #   there would keep results whose tool_use block sits in the assistant
        #   message we just deleted — which that API rejects outright.
        preserved_system = [m for m in messages if m.role == "system"]
        tail_start = trailing_user_turn_start(messages)
        preserved_tail = [m for m in messages[tail_start:] if m.role != "system"]

        # Hash the span the summary replaces, before it is gone: boundary
        # entries carry it so a resume can tell which canonical prefix this
        # projection covered (observability today, hard validation later).
        covered_hash = _covered_prefix_hash(
            [m for m in messages[:tail_start] if m.role != "system"]
        )

        messages.clear()
        messages.extend(preserved_system)
        messages.append(Message(role="user",
                                content=f"[Context compressed]\n\n{summary}"))
        messages.append(Message(role="assistant",
                                content="Understood. I have the conversation context. Continuing."))
        messages.extend(preserved_tail)

        self._consecutive_auto_compact_failures = 0
        self.stats["auto_compact_count"] = self.stats.get("auto_compact_count", 0) + 1
        logger.info(f"Auto-compact complete, messages reduced to {len(messages)}")

        # Write compact boundary to JSONL session log (if configured), then
        # re-append the preserved tail. load()/fork() replay from the last
        # boundary only, and rebuild the summary turn from the boundary itself —
        # so only the tail needs writing. Without it, /resume and /fork after
        # /compact keep the summary and silently drop the pending turn.
        try:
            _agent = self._agent_of(model)
            if _agent is not None:
                _slog = _agent._session_log
                if _slog is not None:
                    _slog.append_compact_boundary(
                        summary,
                        model=model.id if model is not None else None,
                        covered_prefix_hash=covered_hash,
                    )
                    _slog.append_post_compact_messages(preserved_tail)
                    logger.debug("Compact boundary + preserved tail written to session log")
        except Exception as cb_err:
            logger.warning(f"Failed to write compact boundary: {cb_err}")

        return True

    @staticmethod
    def _agent_of(model: Optional[Any]) -> Optional[Any]:
        """Resolve the Agent behind a Model, or None if it's gone."""
        ref = model._agent_ref if model is not None else None
        return ref() if ref is not None else None

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return dict(self.stats)
