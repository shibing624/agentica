# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Compiled experience card store.

CRUD for experience .md files with YAML frontmatter, lifecycle governance
(promotion/demotion/archive), relevance-scored retrieval, and global sync.
"""
import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional

from agentica.experience.compiler import CompiledCard
from agentica.utils.async_file import (
    async_read_text,
    async_write_text,
    extract_frontmatter_int,
    extract_frontmatter_list,
    extract_frontmatter_value,
    format_frontmatter_list,
    strip_frontmatter,
)


# Cap on stored source_tasks per card. Different runs can hit the same
# experience with different originating tasks; we keep a representative
# sample (most-recent-wins) so the spawn prompt doesn't blow its budget.
_MAX_SOURCE_TASKS = 5
# Per-task length cap to keep frontmatter compact.
_SOURCE_TASK_LEN = 200


class CompiledExperienceStore:
    """File-based store for compiled experience cards (.md with YAML frontmatter).

    Each experience is a .md file under exp_dir/ with frontmatter:
        ---
        title: execute_FileNotFoundError
        type: tool_error
        tool: execute
        repeat_count: 3
        first_seen: 2026-04-10
        last_seen: 2026-04-15
        tier: hot
        ---
        Tool `execute` failed...

    Provides: write (bump-not-duplicate), retrieval (scored), lifecycle, sync.

    Usage::

        store = CompiledExperienceStore(
            exp_dir=Path("/workspace/users/default/experiences"),
            index_path=Path("/workspace/users/default/EXPERIENCE.md"),
        )
        await store.write(card)
        text = await store.get_relevant(query="file ops", limit=5)
    """

    _INDEX_MAX_LINES = 200

    def __init__(
        self,
        exp_dir: Path,
        index_path: Path,
        relevance_scorer: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        """Initialize the compiled experience store.

        Args:
            exp_dir: Directory for experience .md files.
            index_path: Path to EXPERIENCE.md index file.
            relevance_scorer: Optional function(query, text) -> float for
                relevance scoring. If None, uses basic word overlap.
        """
        self._exp_dir = exp_dir
        self._index_path = index_path
        self._relevance_scorer = relevance_scorer

    # ── Write ──────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_source_tasks(existing: List[str], new_task: str) -> List[str]:
        """Merge a new source task into the existing list.

        Most-recent-wins ordering with dedup and a hard cap of
        ``_MAX_SOURCE_TASKS``. Empty new_task is a no-op (returns existing).
        """
        if not new_task:
            return existing[:_MAX_SOURCE_TASKS]
        # Move existing copy to the front (most-recent), preserve order otherwise.
        deduped = [t for t in existing if t != new_task]
        merged = [new_task] + deduped
        return merged[:_MAX_SOURCE_TASKS]

    @staticmethod
    def _build_frontmatter(
        card: CompiledCard,
        repeat_count: int,
        first_seen: str,
        last_seen: str,
        source_tasks: List[str],
    ) -> str:
        """Build the canonical frontmatter block for a compiled card.

        ``source_tasks`` and ``correction_key`` are both omitted when
        empty (not written as ``[]`` / blank) so legacy cards without
        the fields stay diff-clean on re-write.
        """
        lines = [
            "---",
            f"title: {card.title}",
            f"type: {card.experience_type}",
            f"tool: {card.tool_name}",
            f"repeat_count: {repeat_count}",
            f"first_seen: {first_seen}",
            f"last_seen: {last_seen}",
            "tier: hot",
        ]
        if card.correction_key:
            lines.append(f"correction_key: {card.correction_key}")
        if source_tasks:
            lines.append(f"source_tasks: {format_frontmatter_list(source_tasks)}")
        lines.append("---\n\n")
        return "\n".join(lines)

    async def write(self, card: CompiledCard) -> str:
        """Write an experience card. Bumps repeat_count if same title exists.

        Args:
            card: Compiled experience card.

        Returns:
            Absolute path to the written .md file.
        """
        self._exp_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r"[^\w\-]", "_", card.title.lower())[:50].strip("_")
        filename = f"{card.experience_type}_{safe_title}.md"
        filepath = self._exp_dir / filename
        today = date.today().isoformat()

        new_task = (card.source_task or "").strip()[:_SOURCE_TASK_LEN]

        if filepath.exists():
            existing = (await async_read_text(filepath)).strip()
            new_count = extract_frontmatter_int(existing, "repeat_count", 1) + 1
            first_seen = extract_frontmatter_value(existing, "first_seen") or today
            existing_tasks = extract_frontmatter_list(existing, "source_tasks")
            merged_tasks = self._merge_source_tasks(existing_tasks, new_task)
            # Preserve a previously-stamped correction_key if the new card
            # didn't bring one (e.g. a tool_error card landing on the same
            # file as a correction by collision — defensive only).
            effective_card = card
            if not card.correction_key:
                prior_key = extract_frontmatter_value(existing, "correction_key") or ""
                if prior_key:
                    effective_card = replace(card, correction_key=prior_key)
            frontmatter = self._build_frontmatter(
                card=effective_card,
                repeat_count=new_count,
                first_seen=first_seen,
                last_seen=today,
                source_tasks=merged_tasks,
            )
            # Rewrite the body from the incoming card rather than carrying the
            # stored one forward. A bump is a fresh occurrence of the same
            # experience, so the newest content is the accurate one, and the
            # card can never accumulate whatever a past parse left behind.
            await async_write_text(filepath, frontmatter + card.content)
        else:
            initial_tasks = [new_task] if new_task else []
            frontmatter = self._build_frontmatter(
                card=card,
                repeat_count=1,
                first_seen=today,
                last_seen=today,
                source_tasks=initial_tasks,
            )
            await async_write_text(filepath, frontmatter + card.content)

        hook = card.content[:100].replace("\n", " ").strip()
        await self._update_index(filename, card.title, hook, card.experience_type)
        return str(filepath)

    async def _update_index(
        self,
        filename: str,
        title: str,
        hook: str,
        experience_type: str,
    ) -> None:
        """Append or update an entry in EXPERIENCE.md index."""
        new_entry = f"- [{title}](experiences/{filename}) — [{experience_type}] {hook[:80]}"

        existing = ""
        if self._index_path.exists():
            existing = (await async_read_text(self._index_path)).strip()

        lines = [l for l in existing.splitlines() if l.strip()] if existing else []
        lines = [l for l in lines if f"(experiences/{filename})" not in l]
        lines.append(new_entry)

        while len(lines) > self._INDEX_MAX_LINES:
            lines.pop(0)

        await async_write_text(self._index_path, "\n".join(lines))

    # ── Retrieval ─────────────────────────────────────────────────────────

    async def get_relevant(
        self,
        query: str = "",
        limit: int = 5,
    ) -> str:
        """Retrieve top-k experiences ranked by relevance + recency + frequency.

        Scoring: relevance * 0.5 + recency * 0.3 + frequency * 0.2.
        Cold-tier experiences are excluded.

        Args:
            query: Current user query for relevance scoring.
            limit: Maximum number of experiences to return.

        Returns:
            Formatted markdown string for system prompt injection, or "".
        """
        if not self._index_path.exists() and not self._exp_dir.exists():
            return ""

        entries = await self._load_index_entries()
        if not entries:
            return ""

        today_ordinal = date.today().toordinal()
        scored_entries = []

        for entry in entries:
            filepath = self._exp_dir / entry["filename"]
            if not filepath.exists():
                continue

            try:
                raw = (await async_read_text(filepath)).strip()
            except (FileNotFoundError, OSError):
                continue

            # Capturing is not injecting. `tool_error` and `success_pattern`
            # cards exist for the skill-upgrade pipeline, which grounds a
            # generated SKILL.md's gotchas in real failures — but neither
            # belongs in a system prompt: "read_file failed: file not found"
            # is telemetry about our own tools, and it used to crowd out the
            # user corrections that actually change behaviour. The prompt gets
            # `correction` cards only; the files stay on disk for the pipeline.
            card_type = extract_frontmatter_value(raw, "type") or entry.get("type", "")
            if card_type in ("success_pattern", "tool_error"):
                continue

            repeat_count = extract_frontmatter_int(raw, "repeat_count", 1)
            last_seen_str = extract_frontmatter_value(raw, "last_seen")
            tier = extract_frontmatter_value(raw, "tier") or "hot"

            if tier == "cold":
                continue

            recency = 0.5
            if last_seen_str:
                try:
                    days_ago = today_ordinal - date.fromisoformat(last_seen_str).toordinal()
                    recency = max(0.0, 1.0 - days_ago / 90.0)
                except ValueError:
                    pass

            frequency = min(repeat_count / 10.0, 1.0)

            relevance = 0.5
            if query.strip():
                text = f"{entry['title']} {entry['hook']}".lower()
                if self._relevance_scorer:
                    relevance = self._relevance_scorer(query.lower(), text)
                else:
                    relevance = _basic_relevance(query.lower(), text)

            composite = relevance * 0.5 + recency * 0.3 + frequency * 0.2
            scored_entries.append({
                **entry,
                "_score": composite,
                "_raw": raw,
                "_tier": tier,
                "_repeat_count": repeat_count,
            })

        scored_entries.sort(key=lambda x: (
            -{"hot": 2, "warm": 1, "cold": 0}.get(x["_tier"], 0),
            -x["_score"],
        ))
        top = scored_entries[:limit]

        if not top:
            return ""

        parts = []
        for entry in top:
            body = strip_frontmatter(entry["_raw"])
            body = _compress_repeated_lines(body)
            body = _truncate_body(body)
            tier_badge = f"[{entry['_tier'].upper()}]" if entry["_tier"] != "hot" else ""
            count_badge = f"(seen {entry['_repeat_count']}x)" if entry["_repeat_count"] > 1 else ""
            header = f"### {entry['title']} {tier_badge} {count_badge}".strip()
            parts.append(f"{header}\n{body}")

        return "\n\n".join(parts)

    async def _load_index_entries(self) -> List[Dict]:
        """Load entries from EXPERIENCE.md index, with dir fallback."""
        entries: List[Dict] = []
        if self._index_path.exists():
            index_content = (await async_read_text(self._index_path)).strip()
            if index_content:
                entries = _parse_experience_index(index_content)

        if not entries and self._exp_dir.exists():
            for f in sorted(self._exp_dir.glob("*.md"), reverse=True):
                entries.append({
                    "title": f.stem,
                    "filename": f.name,
                    "hook": f.stem.replace("_", " "),
                    "type": "unknown",
                })

        return entries

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def run_lifecycle(
        self,
        promotion_count: int = 3,
        promotion_window_days: int = 7,
        demotion_days: int = 30,
        archive_days: int = 90,
    ) -> Dict[str, int]:
        """Run promotion/demotion/archive sweep.

        Rules:
        - repeat_count >= promotion_count AND within window -> tier=hot
        - last_seen > demotion_days ago -> tier=warm
        - last_seen > archive_days ago -> tier=cold

        Returns:
            Dict: {"promoted": N, "demoted": N, "archived": N}
        """
        if not self._exp_dir.exists():
            return {"promoted": 0, "demoted": 0, "archived": 0}

        today_ordinal = date.today().toordinal()
        stats = {"promoted": 0, "demoted": 0, "archived": 0}

        for filepath in self._exp_dir.glob("*.md"):
            try:
                raw = (await async_read_text(filepath)).strip()
            except (FileNotFoundError, OSError):
                continue

            repeat_count = extract_frontmatter_int(raw, "repeat_count", 1)
            last_seen_str = extract_frontmatter_value(raw, "last_seen")
            first_seen_str = extract_frontmatter_value(raw, "first_seen")
            current_tier = extract_frontmatter_value(raw, "tier") or "hot"

            days_since = 0
            if last_seen_str:
                try:
                    days_since = today_ordinal - date.fromisoformat(last_seen_str).toordinal()
                except ValueError:
                    pass

            span_days = 0
            if first_seen_str and last_seen_str:
                try:
                    span_days = (
                        date.fromisoformat(last_seen_str).toordinal()
                        - date.fromisoformat(first_seen_str).toordinal()
                    )
                except ValueError:
                    pass

            # Tier decision order:
            #   1. Archive/demotion checks FIRST — staleness always wins.
            #   2. Promotion only if the experience is still recently active
            #      (days_since <= demotion_days ensures we don't promote stale entries).
            #
            # Previous bug: promotion was checked first, so an experience that was
            # promoted early (high repeat_count, short span) but then went unused
            # for months would never be demoted or archived.
            new_tier = current_tier
            if days_since > archive_days:
                new_tier = "cold"
            elif days_since > demotion_days:
                new_tier = "warm"
            elif repeat_count >= promotion_count and span_days <= promotion_window_days:
                new_tier = "hot"

            if new_tier != current_tier:
                updated = re.sub(r"tier:\s*\w+", f"tier: {new_tier}", raw)
                await async_write_text(filepath, updated)

                if new_tier == "hot" and current_tier != "hot":
                    stats["promoted"] += 1
                elif new_tier == "warm" and current_tier == "hot":
                    stats["demoted"] += 1
                elif new_tier == "cold":
                    stats["archived"] += 1

        return stats

# ── Module-level helpers (experience-specific, not shared) ────────────────

# Cap on how much of one card's body reaches the system prompt. Bodies embed
# raw tool output (stderr, tracebacks) whose middle carries no signal, and the
# whole section is re-sent on every request.
_MAX_BODY_CHARS = 700


def _truncate_body(body: str) -> str:
    """Keep the head and tail of an over-long card body.

    Both ends matter: the head names the tool and the failing arguments, the
    tail carries the exception that was actually raised. Cutting either end
    alone drops the part that identifies the failure.
    """
    if len(body) <= _MAX_BODY_CHARS:
        return body
    head = _MAX_BODY_CHARS * 2 // 3
    tail = _MAX_BODY_CHARS - head
    return f"{body[:head].rstrip()}\n...\n{body[-tail:].lstrip()}"


def _compress_repeated_lines(text: str) -> str:
    """Collapse consecutive duplicate lines into ``<line> xN`` form.

    Defensive rendering layer for legacy experience cards written before the
    compiler started deduping. A success-pattern card with 76 raw ``- read_file``
    lines balloons the system prompt with zero added signal; this collapses
    such runs to a single ``- read_file x76`` row.
    """
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        if out and out[-1] == line and line.strip():
            out[-1] = f"{line} x2"
        elif out and line.strip() and out[-1].startswith(line + " x"):
            count = int(out[-1].rsplit(" x", 1)[1]) + 1
            out[-1] = f"{line} x{count}"
        else:
            out.append(line)
    return "\n".join(out)


def _parse_experience_index(index_content: str) -> List[Dict]:
    """Parse EXPERIENCE.md index lines into entry dicts.

    Expected format: `- [Title](experiences/filename.md) -- [type] hook`
    """
    entries = []
    for line in index_content.splitlines():
        m = re.match(r"-\s+\[(.+?)\]\(experiences/(.+?)\)\s*[—\-]\s*(?:\[(\w+)\]\s*)?(.+)", line)
        if m:
            entries.append({
                "title": m.group(1).strip(),
                "filename": m.group(2).strip(),
                "type": m.group(3).strip() if m.group(3) else "unknown",
                "hook": m.group(4).strip(),
            })
    return entries


def _basic_relevance(query: str, text: str) -> float:
    """Basic word-overlap relevance scoring (fallback when no scorer provided)."""
    query_words = set(query.split())
    if not query_words:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for w in query_words if w in text_lower)
    return matches / len(query_words)
