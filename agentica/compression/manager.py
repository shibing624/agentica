# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Layer 2 compaction — start a fresh context window (no LLM summary).

Layer 1 (``agentica.compression.evict``) shrinks the request for free by
evicting old tool results. When that is not enough, the activity window is
replaced by initial context + session-notes pointer. Prior turns stay in the
session JSONL and are retrieved with ``search_session`` / ``read_session_item``.

This is Codex TokenBudget compact (#29743): same lifecycle name, no summarizer.
"""
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentica.compression.evict import trailing_user_turn_start
from agentica.compression.new_window import notes_path_for, start_new_context_window
from agentica.compression.notes import ensure_rollover_notes
from agentica.model.message import Message
from agentica.utils.log import logger
from agentica.utils.tokens import count_tokens


# Fraction of the context window at which auto-compact fires. A ratio, not an
# absolute buffer: the codebase serves windows from 8K (gpt-4) to 1M, and any
# absolute value is wrong at both ends. 0.95 also buys time for tool results
# to land in prompt cache before the window reset.
AUTO_COMPACT_THRESHOLD_RATIO = 0.95


def parse_compact_token_limit(raw: Any) -> Optional[int]:
    """Positive token cap, or None when unset / non-positive / unparseable."""
    if raw is None or raw == "":
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def working_context_window(
    context_window: int,
    compact_token_limit: Optional[int] = None,
) -> int:
    """Window Layer 1 treats as full. The provider window is the hard cap."""
    if context_window <= 0:
        return 0
    cap = parse_compact_token_limit(compact_token_limit)
    if cap is None:
        return context_window
    return min(cap, context_window)


def auto_compact_threshold(
    context_window: int,
    compact_token_limit: Optional[int] = None,
) -> int:
    """Layer 2 trigger: min(compact_token_limit or ∞, int(window × 0.95)).

    A user cap is an absolute working budget (Codex-style 300k on a 1M
    model), not 95% of that budget. Unset → 95% of the window.
    A cap above the window is ignored; the ratio still leaves headroom.
    """
    if context_window <= 0:
        return 0
    ratio = int(context_window * AUTO_COMPACT_THRESHOLD_RATIO)
    cap = parse_compact_token_limit(compact_token_limit)
    if cap is None:
        return ratio
    return min(cap, ratio)


def _covered_prefix_hash(msgs: List["Message"]) -> str:
    """Stable hash over the messages a window cut replaces (role + content)."""
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
    """Open a fresh context window when the activity window is full.

    Args:
        model: Unused. Kept so existing ``CompressionManager(model=...)``
            construction does not break. Layer 2 no longer calls an LLM.
        compress_token_limit: Legacy native-compact threshold field.
        compact_token_limit: Optional user working cap (absolute tokens).
        compress_target_token_limit: Unused legacy field.

    Example:
        ```python
        from agentica.compression import CompressionManager

        cm = CompressionManager()
        await cm.auto_compact(messages, model=model)
        ```
    """
    model: Optional[Any] = None
    compress_token_limit: Optional[int] = None
    compress_target_token_limit: Optional[int] = None
    compact_token_limit: Optional[int] = None

    stats: Dict[str, Any] = field(default_factory=dict)
    window_id: int = 0
    reminder_claimed: bool = False
    fallback_claimed: bool = False
    compact_token_floor: Optional[int] = None

    def reset_run_state(self) -> None:
        """Per-run hook. Window id and notes-fallback state survive the turn."""
        return

    def __post_init__(self):
        if self.compress_target_token_limit is None and self.compress_token_limit is not None:
            self.compress_target_token_limit = int(self.compress_token_limit * 0.6)

    def _resolve_limits(self, model: Optional[Any] = None) -> None:
        """Auto-resolve compress_token_limit from model.context_window if not set."""
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
        """Always False. Native summarizer compact is no longer a Layer 2 path."""
        return False

    def should_auto_compact(
        self,
        messages: List["Message"],
        model: Optional[Any] = None,
        context_tokens: Optional[int] = None,
    ) -> bool:
        """Return True once token count reaches the Layer 2 threshold."""
        context_window = model.context_window if model is not None else None
        if context_window is None:
            return False
        threshold = auto_compact_threshold(context_window, self.compact_token_limit)
        model_id = model.id if model else "gpt-4o"
        tokens = context_tokens if context_tokens is not None else count_tokens(messages, None, model_id, None)
        if tokens < threshold:
            return False
        floor = self.compact_token_floor
        if floor is not None and tokens < floor:
            return False
        logger.debug(
            f"Auto-compact threshold hit: {tokens:,} tokens "
            f">= {threshold:,} (window={context_window:,}"
            f"{f', cap={self.compact_token_limit:,}' if self.compact_token_limit else ''})"
        )
        return True

    async def auto_compact(
        self,
        messages: List["Message"],
        model: Optional[Any] = None,
        force: bool = False,
        working_memory: Optional[Any] = None,
        custom_instructions: Optional[str] = None,
        keep_trailing_turn: bool = True,
    ) -> bool:
        """Layer 2: install a fresh context window. No LLM summary.

        ``working_memory`` and ``custom_instructions`` are accepted and ignored
        (legacy Layer 2 summariser knobs). ``keep_trailing_turn`` is True for
        auto-compact / ``/compact`` so the pending question survives.
        """
        if not force and not self.should_auto_compact(messages, model):
            return False

        logger.info("Auto-compact triggered: starting a new context window")

        self.window_id += 1
        self.reminder_claimed = False
        self.fallback_claimed = False
        self.compact_token_floor = None

        context_window = 0
        if model is not None and isinstance(model.context_window, int):
            context_window = model.context_window
        working = working_context_window(context_window, self.compact_token_limit)
        tokens_left = working if working else context_window

        agent = self._agent_of(model)
        slog = agent._session_log if agent is not None else None
        notes_path = notes_path_for(slog)

        if keep_trailing_turn:
            tail_start = trailing_user_turn_start(messages)
            covered = [m for m in messages[:tail_start] if m.role != "system"]
        else:
            covered = [m for m in messages if m.role != "system"]
        covered_hash = _covered_prefix_hash(covered)
        notes_text = ensure_rollover_notes(covered, notes_path, self.window_id)

        start_new_context_window(
            messages,
            window_id=self.window_id,
            tokens_left=tokens_left,
            notes_path=notes_path,
            notes_text=notes_text,
            keep_trailing_turn=keep_trailing_turn,
        )

        self.stats["auto_compact_count"] = self.stats.get("auto_compact_count", 0) + 1
        logger.info(
            f"New context window {self.window_id} installed, "
            f"{len(messages)} messages remain"
        )

        try:
            if slog is not None:
                slog.append_compact_boundary(
                    "",
                    model=model.id if model is not None else None,
                    covered_prefix_hash=covered_hash,
                    window_id=self.window_id,
                )
                tail = [m for m in messages if m.role != "system"]
                slog.append_post_compact_messages(tail)
                logger.debug("Empty compact boundary + new-window tail written")
        except Exception as cb_err:
            logger.warning(f"Failed to write compact boundary: {cb_err}")

        return True

    @staticmethod
    def _agent_of(model: Optional[Any]) -> Optional[Any]:
        """Resolve the Agent behind a Model, or None if it's gone."""
        ref = model._agent_ref if model is not None else None
        return ref() if ref is not None else None

    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return dict(self.stats)


async def apply_idle_compact(agent: Any) -> bool:
    """CLI / Web ``/compact``: empty window; preamble waits for the next turn.

    Mid-turn auto / reactive compact keeps the pending question
    (``keep_trailing_turn=True``). An idle slash/Web compact has no pending
    question — keeping the last answered turn would leave the oil-gauge on
    yesterday's user message, not on the next request.
    """
    wm = agent.working_memory
    if wm is None:
        return False
    messages = wm.messages
    if not messages:
        return False
    cm = agent.tool_config.compression_manager if agent.tool_config else None
    if cm is None:
        return False
    compacted = await cm.auto_compact(
        messages,
        model=agent.model,
        force=True,
        keep_trailing_turn=False,
    )
    if compacted:
        wm.collapse_runs(messages)
    return compacted
