# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Experience -> Skill automatic upgrade pipeline (SDK core).

Two LLM touchpoints, one deterministic evidence layer:
1. ``maybe_spawn_skill()``  — judge candidates + generate SKILL.md in one call
2. ``maybe_update_skill_state()`` — at checkpoint, judge keep / promote / revise / rollback
3. ``record_episode()`` — deterministic append to episodes.jsonl

Multi-critic admission gates, append-only provenance audit logs and
LLM-driven repair-or-discard maintenance are NOT part of the SDK. They
live as research / paper-grade extensions in ``evaluation/vag/`` and are
injected via ``SkillLifecycleHooks`` (see ``skill_lifecycle_hooks.py``).
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.utils.log import logger
from agentica.model.message import Message
from agentica.utils.async_file import (
    async_read_text,
    async_write_text,
    extract_frontmatter_int,
    extract_frontmatter_list,
    extract_frontmatter_value,
)
from agentica.prompts.experience.skill_upgrade import (
    get_skill_judge_prompt,
    get_skill_spawn_prompt,
)
from agentica.experience.skill_lifecycle_hooks import (
    NoopSkillLifecycleHooks,
    SkillLifecycleHooks,
)


_FRONTMATTER_DASHES_RE = re.compile(r"^---\s*$", re.MULTILINE)
_FENCE_FRONTMATTER_RE = re.compile(
    r"\A\s*```(?:ya?ml)?\s*\n(?P<yaml>.*?)\n```\s*\n?",
    re.DOTALL,
)
_SECTION_HEADING_RE = re.compile(r"^##\s+", re.MULTILINE)


def _normalize_skill_md(text: str) -> str:
    """Coerce the two LLM-malformed SKILL.md headers we still see in practice
    into canonical ``--- ... ---`` form.

    Covered variants:
    1. ``--- ... ---`` already canonical (idempotent path).
    2. ```` ```yaml ... ``` ```` markdown code fence around the frontmatter.

    Other malformations (missing closing ``---``, bare ``name:`` start, stray
    ``-`` prefix lines) used to be handled with extra fallbacks, but were
    fragile. We now rely on a strict spawn prompt + structured-output JSON
    response; if the LLM still emits a malformed SKILL.md the spawn step
    rejects it on the next validate pass and we re-try at the next sweep.
    """
    text = text.lstrip("\ufeff").lstrip()

    matches = list(_FRONTMATTER_DASHES_RE.finditer(text))
    if len(matches) >= 2:
        first, second = matches[0], matches[1]
        yaml_body = text[first.end():second.start()].strip("\n")
        rest = text[second.end():].lstrip("\n")
        return f"---\n{yaml_body}\n---\n{rest}"

    m = _FENCE_FRONTMATTER_RE.match(text)
    if m:
        yaml_body = m.group("yaml").strip()
        rest = text[m.end():].lstrip()
        return f"---\n{yaml_body}\n---\n{rest}"

    return text


@dataclass
class _CandidateMatch:
    """Per-candidate slice of events.jsonl, computed once per spawn call."""
    recovery_count: int = 0
    matched_events: List[Dict[str, Any]] = None  # type: ignore[assignment]


class SkillEvolutionManager:
    """Manages experience -> skill upgrade lifecycle.

    Usage::

        manager = SkillEvolutionManager()
        skill_name = await manager.maybe_spawn_skill(
            model=agent.auxiliary_model,
            candidates=candidates,
            existing_skills=["slug-a"],
            generated_skills_dir=gen_dir,
        )
    """

    # ── Public API ────────────────────────────────────────────────────────

    async def maybe_spawn_skill(
        self,
        model: Any,
        candidates: List[Dict],
        existing_skills: List[str],
        generated_skills_dir: Path,
        event_store: Optional[Any] = None,
        min_success_applications: int = 0,
        hooks: Optional[SkillLifecycleHooks] = None,
    ) -> Optional[str]:
        """Judge candidates and generate SKILL.md in one LLM call.

        Args:
            model: LLM model with async ``response()`` method.
            candidates: List of dicts with title, content, repeat_count, type, tier.
            existing_skills: Names of already-generated skill slugs.
            generated_skills_dir: Directory for generated skills.
            event_store: Optional ``ExperienceEventStore`` — when provided,
                raw tool_error / tool_recovery events are appended to the
                prompt so the LLM can ground gotchas in real evidence.
            min_success_applications: If > 0, candidates must have at least
                this many ``tool_recovery`` events for their tool (or
                confirmed corrections for their key) before being eligible.
            hooks: Optional ``SkillLifecycleHooks`` — ``gate_admission``
                runs before installing the candidate as a shadow skill.

        Returns:
            Skill slug name if installed, None if no upgrade.
        """
        if not candidates:
            return None
        hooks = hooks or NoopSkillLifecycleHooks()

        # Pull raw events; let I/O errors propagate so callers see real bugs.
        all_events: List[Dict[str, Any]] = []
        if event_store is not None:
            all_events = await event_store.read_all()

        idx = self._index_events(all_events)

        # Per-candidate relevance gate: workspace-global recovery counts
        # would let unrelated tools unlock spawns. Filter to candidates
        # with their own evidence of working.
        if min_success_applications > 0 and event_store is not None:
            kept: List[Dict] = []
            for c in candidates:
                relevant = self._candidate_recovery_count(c, idx)
                if relevant >= min_success_applications:
                    kept.append(c)
                else:
                    logger.debug(
                        f"Skill spawn: candidate {c.get('title')!r} has only "
                        f"{relevant} relevant recovery/confirmation events "
                        f"(need {min_success_applications}); skipping."
                    )
            if not kept:
                logger.debug(
                    "Skill spawn: no candidate met per-candidate recovery "
                    "gate; deferring."
                )
                return None
            candidates = kept

        cards_text = "\n\n".join(
            self._format_card_for_prompt(c) for c in candidates
        )
        existing_text = ", ".join(existing_skills) if existing_skills else "(none)"
        evidence_text = self._build_evidence_text(candidates, idx)

        prompt = (
            get_skill_spawn_prompt()
            + f"Existing generated skills: {existing_text}\n\n"
            + f"Experience cards to evaluate:\n{cards_text}\n"
        )
        if evidence_text:
            prompt += (
                "\nRaw event evidence (use these and only these to write "
                "gotchas — quote the symptom verbatim):\n" + evidence_text + "\n"
            )

        response = await model.response([Message(role="user", content=prompt)])
        if not response or not response.content:
            logger.debug("Skill spawn: empty LLM response")
            return None

        text = _strip_code_fences(response.content)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(
                f"Skill spawn: LLM returned invalid JSON. "
                f"Raw response (first 300 chars): {response.content[:300]}"
            )
            return None
        if not isinstance(result, dict):
            return None

        action = result.get("action")
        if action != "install_shadow":
            logger.debug(
                f"Skill spawn: LLM decided action={action!r} "
                f"(reason={result.get('reason', 'n/a')!r}, "
                f"{len(candidates)} candidate(s) offered)"
            )
            return None

        skill_name = result.get("skill_name", "")
        skill_md = result.get("skill_md", "")
        source = result.get("source_experience", "")
        if not skill_name or not skill_md:
            return None

        slug = re.sub(r"[^\w\-]", "-", skill_name.lower())[:50].strip("-")
        if not slug or slug in existing_skills:
            return None

        skill_md = _normalize_skill_md(skill_md)

        # Aggregate originating tasks across all source candidates so the
        # SKILL.md Source block records every distinct user request that
        # produced this skill (capped). Most-recent-first ordering.
        merged_source_tasks: List[str] = []
        seen_tasks: set = set()
        for c in candidates:
            for t in (c.get("source_tasks") or []):
                clean = str(t).strip()
                if clean and clean not in seen_tasks:
                    seen_tasks.add(clean)
                    merged_source_tasks.append(clean)

        skill_md = self._append_source_section(
            skill_md,
            source=source,
            event_count=sum(
                1 for e in all_events
                if e.get("event_type") in ("tool_error", "tool_recovery")
            ),
            source_tasks=merged_source_tasks,
        )

        # No-Execution-No-Memory gate: skeletons / placeholders / missing
        # gotchas are rejected without ever touching disk.
        is_valid, reason = self._validate_skill_content(skill_md)
        if not is_valid:
            logger.info(f"Skill spawn rejected by validator: {reason}")
            return None

        skill_dir = generated_skills_dir / slug
        meta_for_hook = {
            "skill_name": slug,
            "source_experience": source,
            "stage": "spawn",
        }
        if not await hooks.gate_admission(skill_md, skill_dir, meta_for_hook):
            logger.info(f"Skill spawn rejected by lifecycle gate: {slug}")
            return None

        skill_dir.mkdir(parents=True, exist_ok=True)
        await async_write_text(skill_dir / "SKILL.md", skill_md)
        meta = {
            "skill_name": slug,
            "status": "shadow",
            "source_experience": source,
            "generated_at": date.today().isoformat(),
            "version": 1,
            "total_episodes": 0,
            "success_count": 0,
            "failure_count": 0,
            "consecutive_failures": 0,
            "gotchas_hit_count": 0,
            "new_gotchas_seen": 0,
            "last_judged_at": None,
        }
        self.write_meta(skill_dir / "meta.json", meta)
        self.rebuild_index(generated_skills_dir)
        logger.info(f"Installed shadow skill: {slug} from experience '{source}'")
        return slug

    async def maybe_update_skill_state(
        self,
        model: Any,
        skill_dir: Path,
        checkpoint_interval: int = 5,
        rollback_consecutive_failures: int = 2,
        hooks: Optional[SkillLifecycleHooks] = None,
    ) -> Optional[str]:
        """Judge skill state at checkpoint based on recent episodes.

        Runs only when ``total_episodes`` is a multiple of
        ``checkpoint_interval`` or ``consecutive_failures`` reaches
        ``rollback_consecutive_failures``.

        Args:
            model: LLM with async ``response()``.
            skill_dir: Path to ``generated_skills/{slug}/``.
            checkpoint_interval: Run judgment every N episodes.
            rollback_consecutive_failures: Auto-rollback threshold.
            hooks: Optional ``SkillLifecycleHooks``. ``gate_promotion``
                runs before shadow -> auto. ``on_failure_threshold`` runs
                when failures hit the rollback threshold; if it returns
                ``None`` the SDK falls back to deterministic rollback.

        Returns:
            Decision string, or None if not at checkpoint.
        """
        meta_path = skill_dir / "meta.json"
        episodes_path = skill_dir / "episodes.jsonl"
        skill_md_path = skill_dir / "SKILL.md"

        meta = self.read_meta(meta_path)
        if not meta or meta.get("status") in ("rolled_back", "disabled"):
            return None
        hooks = hooks or NoopSkillLifecycleHooks()

        total = meta.get("total_episodes", 0)
        consecutive_failures = meta.get("consecutive_failures", 0)

        if consecutive_failures >= rollback_consecutive_failures:
            outcome = await hooks.on_failure_threshold(skill_dir, meta, model)
            if outcome is not None:
                # Any non-None return means the hook fully handled the
                # transition (repair / discard / keep_shadow). Don't fall
                # through to default rollback.
                return outcome
            # Default deterministic rollback path.
            meta["status"] = "rolled_back"
            self.write_meta(meta_path, meta)
            self._disable_skill_md(skill_dir)
            logger.info(
                f"Auto-rolled back skill {meta.get('skill_name')} "
                f"after {consecutive_failures} consecutive failures"
            )
            return "rollback"

        if total < checkpoint_interval or total % checkpoint_interval != 0:
            return None

        episodes = self._read_recent_episodes(episodes_path, limit=checkpoint_interval)
        if not episodes:
            return None

        skill_content = ""
        if skill_md_path.exists():
            skill_content = await async_read_text(skill_md_path)
        skill_content_preview = skill_content[:2000]

        episodes_text = "\n".join(
            "- "
            f"[{e.get('outcome', '?')}] "
            f"tool_errors={e.get('tool_errors', 0)} "
            f"user_corrected={e.get('user_corrected', False)} "
            f"skill_followed={e.get('skill_followed', True)} "
            f"hit={e.get('skill_gotchas_hit', [])} "
            f"new={e.get('new_gotchas_found', [])} "
            f"{e.get('query', '')[:100]}"
            for e in episodes
        )

        prompt = (
            get_skill_judge_prompt()
            + f"Skill: {meta.get('skill_name', '?')}\n"
            + f"Status: {meta.get('status', '?')}\n"
            + f"Total episodes: {total}\n"
            + f"Success rate: {meta.get('success_count', 0)}/{total}\n"
            + f"Consecutive failures: {consecutive_failures}\n"
            + f"Gotchas the skill caught (cumulative): "
            + f"{meta.get('gotchas_hit_count', 0)}\n"
            + f"New gotchas not yet covered (cumulative): "
            + f"{meta.get('new_gotchas_seen', 0)}\n\n"
            + f"Recent episodes:\n{episodes_text}\n\n"
            + f"SKILL.md content:\n{skill_content_preview}\n"
        )
        response = await model.response([Message(role="user", content=prompt)])
        if not response or not response.content:
            return None

        text = _strip_code_fences(response.content)
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Skill judge: LLM returned invalid JSON")
            return None
        if not isinstance(result, dict):
            return None

        decision = result.get("decision", "keep_shadow")
        meta["last_judged_at"] = date.today().isoformat()

        if decision == "promote":
            meta_for_hook = dict(meta)
            meta_for_hook["stage"] = "promote"
            if await hooks.gate_promotion(skill_content, skill_dir, meta_for_hook):
                meta["status"] = "auto"
            else:
                decision = "keep_shadow"
        elif decision == "rollback":
            meta["status"] = "rolled_back"
            self._disable_skill_md(skill_dir)
        elif decision == "revise":
            revised_md = None
            section_updates = result.get("section_updates")
            if isinstance(section_updates, dict) and skill_md_path.exists():
                current_skill_md = await async_read_text(skill_md_path)
                revised_md = self._apply_section_updates(current_skill_md, section_updates)
            if revised_md is None:
                revised_md = result.get("revised_skill_md")
            if revised_md:
                revised_md = _normalize_skill_md(revised_md)
                revised_md = self._append_source_section(
                    revised_md,
                    source=meta.get("source_experience", ""),
                    event_count=meta.get("gotchas_hit_count", 0)
                    + meta.get("new_gotchas_seen", 0),
                )
                is_valid, reason = self._validate_skill_content(revised_md)
                if not is_valid:
                    logger.info(
                        f"Skill {meta.get('skill_name')}: revision rejected "
                        f"by validator ({reason}); keeping current version"
                    )
                    decision = "keep_shadow"
                else:
                    await async_write_text(skill_md_path, revised_md)
                    meta["version"] = meta.get("version", 1) + 1

        self.write_meta(meta_path, meta)
        logger.info(f"Skill {meta.get('skill_name')}: judge decision = {decision}")
        return decision

    # ── Episode recording / meta updates (deterministic) ──────────────────

    @staticmethod
    def record_episode(
        episodes_path: Path,
        outcome: str,
        query: str = "",
        tool_errors: int = 0,
        user_corrected: bool = False,
        skill_followed: bool = True,
        skill_gotchas_hit: Optional[List[str]] = None,
        new_gotchas_found: Optional[List[str]] = None,
    ) -> None:
        """Append a runtime episode to ``episodes.jsonl``."""
        episode = {
            "date": date.today().isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outcome": outcome,
            "query": query[:200],
            "tool_errors": tool_errors,
            "user_corrected": user_corrected,
            "skill_followed": skill_followed,
            "skill_gotchas_hit": list(skill_gotchas_hit or []),
            "new_gotchas_found": list(new_gotchas_found or []),
        }
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(episode, ensure_ascii=False) + "\n"
        with episodes_path.open("a", encoding="utf-8") as f:
            f.write(line)

    @staticmethod
    def update_meta_after_episode(
        meta_path: Path,
        outcome: str,
        skill_gotchas_hit: Optional[List[str]] = None,
        new_gotchas_found: Optional[List[str]] = None,
    ) -> Dict:
        """Update meta.json counters after one episode."""
        meta = SkillEvolutionManager.read_meta(meta_path)
        if not meta:
            return {}

        meta["total_episodes"] = meta.get("total_episodes", 0) + 1
        if outcome == "success":
            meta["success_count"] = meta.get("success_count", 0) + 1
            meta["consecutive_failures"] = 0
        elif outcome == "failure":
            meta["failure_count"] = meta.get("failure_count", 0) + 1
            meta["consecutive_failures"] = meta.get("consecutive_failures", 0) + 1

        if skill_gotchas_hit:
            meta["gotchas_hit_count"] = (
                meta.get("gotchas_hit_count", 0) + len(skill_gotchas_hit)
            )
        if new_gotchas_found:
            meta["new_gotchas_seen"] = (
                meta.get("new_gotchas_seen", 0) + len(new_gotchas_found)
            )

        SkillEvolutionManager.write_meta(meta_path, meta)
        return meta

    @staticmethod
    def read_meta(meta_path: Path) -> Dict:
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def write_meta(meta_path: Path, meta: Dict) -> None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def get_candidate_cards(
        exp_dir: Path,
        min_repeat_count: int = 3,
        min_tier: str = "hot",
    ) -> List[Dict]:
        """Scan experience .md files and return cards meeting upgrade threshold."""
        if not exp_dir.exists():
            return []

        allowed_tiers = {"hot"} if min_tier == "hot" else {"hot", "warm"}
        candidates = []

        for filepath in exp_dir.glob("*.md"):
            try:
                raw = filepath.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                continue

            repeat_count = extract_frontmatter_int(raw, "repeat_count", 1)
            tier = extract_frontmatter_value(raw, "tier") or "hot"
            title = extract_frontmatter_value(raw, "title") or filepath.stem
            exp_type = extract_frontmatter_value(raw, "type") or "unknown"
            tool_name = extract_frontmatter_value(raw, "tool") or ""
            correction_key = extract_frontmatter_value(raw, "correction_key") or ""
            source_tasks = extract_frontmatter_list(raw, "source_tasks")

            if repeat_count < min_repeat_count or tier not in allowed_tiers:
                continue

            content = re.sub(r"^---[\s\S]*?---\s*", "", raw, flags=re.MULTILINE).strip()
            candidates.append({
                "title": title,
                "content": content[:500],
                "repeat_count": repeat_count,
                "type": exp_type,
                "tier": tier,
                "tool": tool_name,
                "correction_key": correction_key,
                "filename": filepath.name,
                "source_tasks": source_tasks,
            })

        return candidates

    @staticmethod
    def list_generated_skills(generated_skills_dir: Path) -> List[Dict]:
        if not generated_skills_dir.exists():
            return []
        skills = []
        for skill_dir in sorted(generated_skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            meta = SkillEvolutionManager.read_meta(skill_dir / "meta.json")
            if meta:
                skills.append(meta)
        return skills

    # ── Validation ────────────────────────────────────────────────────────

    _PLACEHOLDER_RE = re.compile(
        r"#\s*TODO\b|#\s*FIXME\b|<your[_ ][^>]*>|<placeholder>|"
        r"\bpass\s*#\s*implement\b",
        re.IGNORECASE,
    )
    _CODE_BLOCK_RE = re.compile(r"```[\w-]*\n(.*?)\n```", re.DOTALL)
    _GOTCHA_RE = re.compile(r"⚠️|##\s*Gotchas|##\s*\u907f\u5751|##\s*\u5751\u70b9")
    _FORBIDDEN_HEADINGS_RE = re.compile(
        r"^#{1,3}\s*(Overview|When To Use|Workflow|Failure Recovery)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    @classmethod
    def _validate_skill_content(cls, skill_md: str) -> tuple:
        """No-Execution-No-Memory rules. Returns ``(is_valid, reason)``."""
        if not skill_md or not skill_md.strip():
            return False, "empty skill content"

        body = skill_md
        if body.startswith("---"):
            end = body.find("\n---", 3)
            if end != -1:
                body = body[end + len("\n---"):]

        if not cls._GOTCHA_RE.search(body):
            return False, "missing gotchas section (no ⚠️ markers or heading found)"

        m = cls._PLACEHOLDER_RE.search(body)
        if m:
            return False, f"contains placeholder/TODO marker: {m.group(0)!r}"

        m = cls._FORBIDDEN_HEADINGS_RE.search(body)
        if m:
            return False, (
                f"contains forbidden textbook heading {m.group(1)!r} "
                "(skill must be gotcha-first, not tutorial-style)"
            )

        for block in cls._CODE_BLOCK_RE.findall(body):
            non_empty = [ln for ln in block.split("\n") if ln.strip()]
            if not non_empty:
                continue
            avg_len = sum(len(ln) for ln in non_empty) / len(non_empty)
            if avg_len < 10:
                return False, (
                    f"code block looks like a skeleton "
                    f"(avg {avg_len:.1f} chars/line < 10)"
                )

        return True, ""

    # ── Prompt assembly helpers ───────────────────────────────────────────

    _SOURCE_TASKS_PER_CARD = 3
    _SOURCE_TASK_DISPLAY_LEN = 200

    @classmethod
    def _format_card_for_prompt(cls, card: Dict) -> str:
        header = (
            f"### {card['title']} (repeat: {card.get('repeat_count', 1)}, "
            f"type: {card.get('type', 'unknown')})"
        )
        body = card.get("content", "")
        sample_tasks = (card.get("source_tasks") or [])[:cls._SOURCE_TASKS_PER_CARD]
        if not sample_tasks:
            return f"{header}\n{body}"
        rendered = "\n".join(
            f"- {str(t)[:cls._SOURCE_TASK_DISPLAY_LEN]}" for t in sample_tasks
        )
        return f"{header}\n{body}\n\nSource tasks (samples):\n{rendered}"

    @staticmethod
    def _index_events(all_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Single O(N) scan into per-bucket indices used by both the recovery
        gate and the evidence builder during one spawn call.
        """
        events_by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        recovery_count_by_tool: Dict[str, int] = defaultdict(int)
        classifications_by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        correction_count_by_key: Dict[str, int] = defaultdict(int)

        for e in all_events:
            etype = e.get("event_type")
            if etype in ("tool_error", "tool_recovery"):
                tool = str(e.get("tool", ""))
                if tool:
                    events_by_tool[tool].append(e)
                    if etype == "tool_recovery":
                        recovery_count_by_tool[tool] += 1
            elif etype == "correction_classification":
                key = str(e.get("correction_key", ""))
                if key:
                    classifications_by_key[key].append(e)
                    if e.get("is_correction") and e.get("should_persist"):
                        correction_count_by_key[key] += 1

        return {
            "events_by_tool": events_by_tool,
            "recovery_count_by_tool": recovery_count_by_tool,
            "classifications_by_key": classifications_by_key,
            "correction_count_by_key": correction_count_by_key,
        }

    @staticmethod
    def _candidate_recovery_count(candidate: Dict, idx: Dict[str, Any]) -> int:
        """How many *relevant* recovery / confirmation events exist for one card."""
        ctype = candidate.get("type", "")
        if ctype == "correction":
            key = (candidate.get("correction_key") or "").strip()
            return idx["correction_count_by_key"].get(key, 0) if key else 0
        tool = (candidate.get("tool") or "").strip()
        return idx["recovery_count_by_tool"].get(tool, 0) if tool else 0

    @staticmethod
    def _build_evidence_text(
        candidates: List[Dict],
        idx: Dict[str, Any],
        per_candidate_limit: int = 5,
    ) -> str:
        """Render strict per-candidate raw-event evidence for the spawn prompt.

        Two distinct corrections in the same workspace will NEVER share
        evidence — matching is by ``correction_key`` (correction cards) or
        by ``tool`` (tool/success cards), strict equality.
        """
        sections: List[str] = []
        for c in candidates:
            ctype = c.get("type", "")
            title = c.get("title", "")
            if ctype == "correction":
                key = (c.get("correction_key") or "").strip()
                matches = idx["classifications_by_key"].get(key, []) if key else []
            else:
                tool = (c.get("tool") or "").strip()
                matches = idx["events_by_tool"].get(tool, []) if tool else []
            if not matches:
                continue

            recent = matches[-per_candidate_limit:]
            lines = [f"### {title}"]
            for e in recent:
                etype = e.get("event_type", "?")
                if etype == "correction_classification":
                    user = str(e.get("user_message", ""))[:200]
                    rule = str(e.get("rule", ""))[:120]
                    lines.append(f"- [correction] user={user!r} rule={rule!r}")
                else:
                    err = str(e.get("error", "") or e.get("note", ""))[:200]
                    lines.append(f"- [{etype}] {err}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def _append_source_section(
        skill_md: str,
        source: str,
        event_count: int,
        source_tasks: Optional[List[str]] = None,
    ) -> str:
        """Replace / append the trailing ``## Source`` section deterministically."""
        lines = [
            "\n\n## Source",
            f"- generated from experience card: `{source or 'unknown'}`",
            f"- raw events cited: {event_count}",
            f"- generated_at: {date.today().isoformat()}",
        ]
        if source_tasks:
            lines.append("- originating tasks:")
            for t in source_tasks[:5]:
                clean = str(t).strip().replace("\n", " ")[:200]
                if clean:
                    lines.append(f"  - {clean}")
        block = "\n".join(lines) + "\n"
        cleaned = re.sub(
            r"\n##\s*Source\b.*\Z", "", skill_md.rstrip(), flags=re.DOTALL,
        )
        return cleaned + block

    # ── Section update helpers (used by `revise` decision) ────────────────

    @staticmethod
    def _format_gotchas_block(gotchas: Any) -> Optional[str]:
        if isinstance(gotchas, str):
            text = gotchas.strip()
            return text or None
        if not isinstance(gotchas, list):
            return None
        lines: List[str] = []
        for item in gotchas:
            text = str(item).strip()
            if not text:
                continue
            if text.startswith("-"):
                lines.append(text)
            elif text.startswith("⚠️"):
                lines.append(f"- {text}")
            else:
                lines.append(f"- ⚠️ {text}")
        return "\n".join(lines) if lines else None

    @staticmethod
    def _replace_summary_block(skill_md: str, summary: str) -> str:
        normalized = _normalize_skill_md(skill_md)
        matches = list(_FRONTMATTER_DASHES_RE.finditer(normalized))
        if len(matches) < 2:
            return normalized
        body_start = matches[1].end()
        prefix = normalized[:body_start].rstrip() + "\n\n"
        body = normalized[body_start:].lstrip("\n")
        heading_match = _SECTION_HEADING_RE.search(body)
        tail = body[heading_match.start():].lstrip() if heading_match else ""
        summary_block = summary.strip()
        if not summary_block:
            return normalized
        if tail:
            return prefix + summary_block + "\n\n" + tail
        return prefix + summary_block + "\n"

    @staticmethod
    def _replace_named_section(skill_md: str, heading: str, body: str) -> str:
        replacement = f"## {heading}\n{body.strip()}\n\n"
        pattern = re.compile(
            rf"(?ms)^##\s*{re.escape(heading)}\s*\n.*?(?=^##\s|\Z)"
        )
        if pattern.search(skill_md):
            return pattern.sub(replacement, skill_md, count=1)
        return skill_md.rstrip() + "\n\n" + replacement

    @classmethod
    def _apply_section_updates(cls, skill_md: str, section_updates: Dict[str, Any]) -> Optional[str]:
        updated = _normalize_skill_md(skill_md)
        changed = False

        summary = section_updates.get("summary")
        if isinstance(summary, str) and summary.strip():
            updated = cls._replace_summary_block(updated, summary)
            changed = True

        gotchas_block = cls._format_gotchas_block(section_updates.get("gotchas"))
        if gotchas_block:
            updated = cls._replace_named_section(updated, "Gotchas", gotchas_block)
            changed = True

        minimal_example = section_updates.get("minimal_example")
        if isinstance(minimal_example, str) and minimal_example.strip():
            updated = cls._replace_named_section(updated, "Minimal Example", minimal_example)
            changed = True

        return updated if changed else None

    # ── Index / disable helpers ───────────────────────────────────────────

    _INDEX_HEADER = (
        "# Generated Skills Index (L1)\n\n"
        "Auto-generated from skill frontmatter. Do not edit by hand.\n"
        "One row per active skill: `keywords -> name (description)`.\n\n"
    )

    @classmethod
    def rebuild_index(cls, generated_skills_dir: Path) -> Optional[Path]:
        """Regenerate INDEX.md listing every active generated skill."""
        if not generated_skills_dir.exists():
            return None

        rows: List[str] = []
        for skill_dir in sorted(generated_skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            md_path = skill_dir / "SKILL.md"
            if not md_path.exists():
                continue
            meta = cls.read_meta(skill_dir / "meta.json")
            status = meta.get("status")
            if status in ("rolled_back", "disabled", "retired"):
                continue
            try:
                raw = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            name = extract_frontmatter_value(raw, "name") or skill_dir.name
            desc = extract_frontmatter_value(raw, "description") or ""
            keywords = extract_frontmatter_value(raw, "when-to-use") or ""
            rows.append(f"- `{keywords}` -> **{name}** — {desc}")

        if not rows:
            return None

        index_path = generated_skills_dir / "INDEX.md"
        index_path.write_text(
            cls._INDEX_HEADER + "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        return index_path

    @staticmethod
    def _disable_skill_md(skill_dir: Path) -> None:
        """Rename SKILL.md to SKILL.md.disabled so SkillLoader won't discover it."""
        skill_md = skill_dir / "SKILL.md"
        if skill_md.exists():
            skill_md.rename(skill_dir / "SKILL.md.disabled")

    @staticmethod
    def _read_recent_episodes(episodes_path: Path, limit: int = 10) -> List[Dict]:
        if not episodes_path.exists():
            return []
        try:
            lines = episodes_path.read_text(encoding="utf-8").strip().splitlines()
        except (OSError, UnicodeDecodeError):
            return []
        episodes = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return episodes


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text
