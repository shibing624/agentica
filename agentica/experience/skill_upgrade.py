# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Experience → Skill automatic upgrade pipeline.

Two LLM touchpoints:
1. maybe_spawn_skill(): Judge candidates + generate SKILL.md in one call
2. maybe_update_skill_state(): At checkpoint, judge keep/promote/revise/rollback

All runtime evidence (episodes) is recorded deterministically — no LLM.
LLM is only invoked for semantic judgment at spawn time and at checkpoints.
"""
import json
import re
from dataclasses import dataclass, field
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
from agentica.skills.evolution import SkillAdmissionGate
from agentica.skills.provenance import append_provenance_event


@dataclass
class _EventIndex:
    """One-shot bucketed view of events.jsonl used by the spawn pipeline.

    Built once per ``maybe_spawn_skill`` call and consumed by both the
    per-candidate recovery gate and the per-candidate evidence builder,
    so each candidate is O(1) instead of O(N) re-scans.
    """
    # tool -> count of tool_recovery events
    recovery_count_by_tool: Dict[str, int] = field(default_factory=dict)
    # tool -> tool_error + tool_recovery events, in original log order
    events_by_tool: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # correction_key -> all correction_classification events (any verdict)
    classifications_by_key: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # correction_key -> count of confirmed-and-persisted classifications
    correction_count_by_key: Dict[str, int] = field(default_factory=dict)


_FENCE_FRONTMATTER_RE = re.compile(
    r"\A\s*```(?:ya?ml)?\s*\n(?P<yaml>.*?)\n```\s*\n?",
    re.DOTALL,
)
_FRONTMATTER_DASHES_RE = re.compile(r"^---\s*$", re.MULTILINE)
_SECTION_HEADING_RE = re.compile(r"^##\s+", re.MULTILINE)


def _normalize_skill_md(text: str) -> str:
    """Coerce common LLM-malformed SKILL.md headers into ``--- ... ---``.

    The parser strictly requires the file to start with ``---\\n``. LLMs
    routinely violate this in a few ways:

    1. ``\u0060\u0060\u0060yaml ... \u0060\u0060\u0060`` markdown code fence around the frontmatter.
    2. ``\u0060\u0060\u0060yaml`` opening fence but ``---`` closing (mixed form).
    3. Stray characters ahead of the real ``---`` line (e.g. a single ``-``
       leaked because the model misread "first character must be '-'").
    4. Missing the opening ``---`` line entirely (frontmatter starts with
       ``name:``).
    5. Opening ``---`` present but closing ``---`` missing (frontmatter
       ends at the first markdown heading or blank-then-heading).

    Strategy: find the *first* line that is exactly ``---`` and the *next*
    line that is exactly ``---``. Everything between them is the YAML body;
    everything after the second ``---`` is the markdown body. Anything
    before the first ``---`` is preamble noise and is dropped. If we cannot
    find two ``---`` lines, fall back to the older variant-specific path
    so canonical-but-fenced inputs still get rewritten.
    """
    text = text.lstrip("\ufeff").lstrip()

    # Drop stray noise lines before the first proper `---`. LLMs sometimes
    # emit a lone `-` at the top because they misread "first char must be -".
    lines = text.split("\n")
    while lines and lines[0].strip() in ("-", "--"):
        lines = lines[1:]
    text = "\n".join(lines).lstrip()

    matches = list(_FRONTMATTER_DASHES_RE.finditer(text))
    if len(matches) >= 2:
        first, second = matches[0], matches[1]
        yaml_body = text[first.end():second.start()].strip("\n")
        rest = text[second.end():].lstrip("\n")
        return f"---\n{yaml_body}\n---\n{rest}"

    # Variant: ```yaml opening fence + ``` closing fence, no `---` anywhere.
    m = _FENCE_FRONTMATTER_RE.match(text)
    if m:
        yaml_body = m.group("yaml").strip()
        rest = text[m.end():].lstrip()
        return f"---\n{yaml_body}\n---\n{rest}"

    # Variant: ```yaml opening fence + `---` closing line (mixed form).
    fence_prefix = None
    for prefix in ("```yaml", "```YAML", "```yml"):
        if text.startswith(prefix):
            fence_prefix = prefix
            break
    if fence_prefix and matches:
        first = matches[0]
        # Body between the line right after the opening fence and the first ---.
        first_nl = text.find("\n")
        yaml_body = text[first_nl + 1:first.start()].strip("\n")
        rest = text[first.end():].lstrip("\n")
        return f"---\n{yaml_body}\n---\n{rest}"

    # Variant: bare YAML keys at the top, single closing `---`.
    if text.startswith("name:"):
        end = text.find("\n---")
        if end != -1:
            yaml_body = text[:end].strip()
            rest = text[end + len("\n---"):].lstrip()
            return f"---\n{yaml_body}\n---\n{rest}"

    # Variant: opening `---` present, closing `---` missing. Frontmatter is
    # everything from the first `---` up to (but not including) the first
    # markdown heading line. LLMs love this when forced into a strict format.
    #
    # The body-start signal must avoid catching YAML inline comments, which
    # are also `#`-prefixed. We require either:
    #   (a) ``##``+ heading (level >= 2), since YAML keys never start with
    #       ``##``, OR
    #   (b) a level-1 `#` heading that is preceded by a blank line — YAML
    #       frontmatter never has blank lines between keys, but markdown
    #       bodies almost always start after one.
    if len(matches) == 1:
        first = matches[0]
        body_start_re = re.compile(
            r"(?:\n\s*\n|\A)(#{2,6}\s)|"   # any heading after blank line, or
            r"(?:\n\s*\n)(#\s)",            # # heading after blank line
            re.MULTILINE,
        )
        body_match = body_start_re.search(text, first.end())
        if body_match:
            heading_start = body_match.start(1) if body_match.group(1) is not None else body_match.start(2)
            yaml_body = text[first.end():heading_start].strip("\n")
            rest = text[heading_start:]
            return f"---\n{yaml_body}\n---\n{rest}"

    return text


class SkillEvolutionManager:
    """Manages experience → skill upgrade lifecycle.

    Two LLM touchpoints, one deterministic evidence layer:
    - maybe_spawn_skill(): one LLM call to judge + generate SKILL.md
    - maybe_update_skill_state(): one LLM call at checkpoint to judge state
    - record_episode(): deterministic append to episodes.jsonl

    Usage::

        manager = SkillEvolutionManager()
        skill_name = await manager.maybe_spawn_skill(
            model=agent.auxiliary_model,
            candidates=candidates,
            existing_skills=["slug-a"],
            generated_skills_dir=gen_dir,
        )
    """

    # ── LLM Prompts ──────────────────────────────────────────────────────

    _SPAWN_PROMPT = (
        "You are deciding whether ONE of the experience cards below should "
        "be upgraded into a reusable SKILL.md file.\n\n"
        "A SKILL.md is a 'don't step on this landmine again' note, NOT a "
        "'how to do X' tutorial. Every user-correction card with "
        "repeat_count >= 3 is a rule the user reinforced multiple times — "
        "strongly prefer install_shadow for the highest-repeat correction "
        "card unless it is genuinely a one-off preference. Tool-error cards "
        "alone are NOT skills, but a matching correction card next to them IS.\n\n"
        "Decision recipe:\n"
        "1. Pick the candidate with the highest repeat_count whose type is "
        "'correction'.\n"
        "2. If its repeat_count >= 3, return action=install_shadow.\n"
        "3. Skip only if (a) only tool_error candidates are high-repeat, "
        "(b) an existing generated skill already covers this rule, or "
        "(c) the rule is a one-off preference with no procedural content.\n\n"
        "Return JSON only:\n"
        '{"action": "ignore|install_shadow", '
        '"skill_name": "kebab-case-slug", '
        '"source_experience": "title of the source experience card", '
        '"reason": "why this deserves to be a skill", '
        '"skill_md": "full SKILL.md content (see format below)"}\n\n'
        "## skill_md format (gotcha-first, NOT textbook)\n\n"
        "The skill_md string MUST NOT be wrapped in ```yaml or any other "
        "code fence. It MUST start with '-' (the opening '---').\n\n"
        "FRONTMATTER (minimal, exactly 3 keys):\n"
        "---\n"
        "name: <kebab-case slug, equals skill_name above>\n"
        "description: <one sentence, ≤25 words>\n"
        "when-to-use: <comma-separated keywords for discovery>\n"
        "---\n\n"
        "BODY STRUCTURE (strict, in this order):\n"
        "1. One-line summary (≤30 words).\n"
        "2. ## Gotchas (REQUIRED, MUST have ≥2 items).\n"
        "   - Each gotcha = one observed failure + the fix.\n"
        "   - Format: '⚠️ <symptom>: <root cause>. <minimal fix>'\n"
        "   - Every gotcha MUST be traceable to evidence in the cards / "
        "raw events shown above. Do NOT invent gotchas.\n"
        "3. ## Minimal Example (≤10 lines, real params, no '# TODO' "
        "placeholders, no '<your_value_here>').\n"
        "4. ## Source (auto-filled by the system, leave a blank section).\n\n"
        "FORBIDDEN (will be auto-rejected):\n"
        "- Sections named 'Overview' / 'When To Use' / 'Workflow' / "
        "'Failure Recovery' (these are textbook fluff).\n"
        "- Generic steps the agent could derive from reading docs.\n"
        "- Unverified claims (every gotcha must trace to a real event).\n"
        "- Placeholder code: '# TODO', '<your_*_here>', 'FIXME', "
        "'pass  # implement'.\n"
        "- Skeleton code blocks with <10 chars per line on average.\n\n"
        "Remember: a skill captures lessons that can ONLY be learned by "
        "actually running the tool and getting burned. If you cannot point "
        "to a concrete failure event for a gotcha, do NOT include it.\n\n"
    )

    _JUDGE_PROMPT = (
        "You are evaluating a shadow-installed generated skill based on its "
        "runtime performance episodes.\n\n"
        "Signals to weigh (the most important first):\n"
        "1. gotchas_hit_count > 0 — strong evidence the skill saved the "
        "agent from documented landmines. Lean toward PROMOTE.\n"
        "2. new_gotchas_seen > 0 — the skill is working but incomplete. "
        "Lean toward REVISE and rewrite the gotchas section to cover them.\n"
        "3. consecutive_failures or low success rate without any "
        "gotchas_hit — the skill might be misleading. Lean toward "
        "ROLLBACK.\n"
        "4. Otherwise, KEEP_SHADOW until more data accumulates.\n\n"
        "Decisions:\n"
        "- keep_shadow: not enough data yet, keep running\n"
        "- promote: skill is performing well, promote to full status\n"
        "- revise: skill idea is good but needs changes — prefer returning "
        "section_updates so the system can patch the current SKILL.md "
        "instead of rewriting the whole file\n"
        "- rollback: skill is causing problems, disable it\n\n"
        "Return JSON only:\n"
        '{"decision": "keep_shadow|promote|revise|rollback", '
        '"reason": "...", '
        '"section_updates": {"summary": "...", "gotchas": ["...", "..."], '
        '"minimal_example": "..."} (preferred for revise, otherwise null), '
        '"revised_skill_md": "..." (legacy fallback, only if decision is revise)}\n\n'
    )

    _MAINTENANCE_PROMPT = (
        "You are maintaining a generated SKILL.md that recently "
        "failed multiple times.\n\n"
        "Decide whether this skill should be repaired or discarded. Prefer "
        "repair only when the failures point to a local fix in the skill "
        "instructions. Discard when the method is obsolete, misleading, "
        "conflicts with newer guidance, or depends on a removed tool/API.\n\n"
        "Return JSON only:\n"
        '{"decision": "repair|discard", '
        '"reason": "...", '
        '"revised_skill_md": "full repaired SKILL.md when decision=repair"}\n\n'
        "If the skill cannot be repaired, you may also reply with a line "
        "starting with DISCARD followed by the reason.\n\n"
    )

    # ── Public API ────────────────────────────────────────────────────────

    async def maybe_spawn_skill(
        self,
        model: Any,
        candidates: List[Dict],
        existing_skills: List[str],
        generated_skills_dir: Path,
        event_store: Optional[Any] = None,
        min_success_applications: int = 0,
        admission_critics: Optional[List[Any]] = None,
        write_provenance: bool = True,
    ) -> Optional[str]:
        """Judge candidates and generate SKILL.md in one LLM call.

        Args:
            model: LLM model instance with async response() method.
            candidates: List of dicts with title, content, repeat_count, type, tier.
            existing_skills: Names of already-generated skill slugs.
            generated_skills_dir: Directory for generated skills.
            event_store: Optional ExperienceEventStore — when provided, raw
                tool_error / tool_recovery events are appended to the prompt
                so the LLM can ground every gotcha in real evidence.
            min_success_applications: If > 0, candidates must have at least
                this many ``tool_recovery`` events in the event store before
                being eligible. Without recoveries we only know what failed,
                not whether a fix actually worked.
            admission_critics: Optional VaG critics run before installing the
                candidate as a shadow skill.
            write_provenance: Whether to append gate lifecycle events to
                provenance.jsonl beside generated skills.

        Returns:
            Skill slug name if installed, None if no upgrade.
        """
        if not candidates:
            return None

        # Optional: pull raw events for the evidence chain + recovery gating.
        # event_store.read_all() is a plain file read; if it fails that is an
        # actual I/O / permission bug the caller must see, not a silent
        # degradation — let the exception propagate.
        all_events: List[Dict[str, Any]] = []
        if event_store is not None:
            all_events = await event_store.read_all()

        # One-shot index: O(N) scan up front, then both the gate and the
        # evidence builder do O(1) bucket lookups instead of O(N) re-scans
        # per candidate. Also lets us answer "is this candidate relevant"
        # without a second pass.
        idx = self._index_events_once(all_events)

        # Per-candidate relevance gate. Workspace-global recovery counts
        # let an unrelated tool's recoveries unlock a brand-new skill —
        # that's how silent mis-spawns happen. Filter candidates down to
        # only those with their own evidence of working.
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

        # Build context for LLM
        cards_text = "\n\n".join(
            self._format_card_for_prompt(c) for c in candidates
        )
        existing_text = ", ".join(existing_skills) if existing_skills else "(none)"

        # Evidence chain: attach the most recent raw tool_error / tool_recovery
        # (for tool/success cards) or correction_classification (for
        # correction cards, filtered by correction_key so two distinct
        # corrections don't share each other's user utterances). The LLM
        # is forbidden from inventing gotchas; this is what it cites.
        evidence_text = self._build_evidence_text(candidates, idx)

        prompt = (
            self._SPAWN_PROMPT
            + f"Existing generated skills: {existing_text}\n\n"
            + f"Experience cards to evaluate:\n{cards_text}\n"
        )
        if evidence_text:
            prompt += (
                "\nRaw event evidence (use these and only these to write "
                "gotchas — quote the symptom verbatim):\n" + evidence_text + "\n"
            )

        response = await model.response([
            Message(role="user", content=prompt),
        ])
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
            logger.debug(f"Skill spawn: LLM JSON is not a dict: {type(result).__name__}")
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
            logger.debug(
                f"Skill spawn: LLM returned install_shadow but missing fields "
                f"(skill_name={bool(skill_name)}, skill_md={bool(skill_md)})"
            )
            return None

        # Sanitize slug
        slug = re.sub(r"[^\w\-]", "-", skill_name.lower())[:50].strip("-")
        if not slug:
            return None

        # Skip if already exists
        if slug in existing_skills:
            logger.debug(f"Skill spawn: slug {slug!r} already exists, skipping")
            return None

        # LLMs often wrap frontmatter in ```yaml fences which the parser rejects;
        # normalize to canonical `---` form before persisting.
        skill_md = _normalize_skill_md(skill_md)

        # Aggregate originating tasks across all source candidates so the
        # SKILL.md Source block records every distinct user request that
        # ever produced this skill (capped). Most-recent-first ordering.
        merged_source_tasks: List[str] = []
        seen_tasks: set = set()
        for c in candidates:
            for t in (c.get("source_tasks") or []):
                clean = str(t).strip()
                if not clean or clean in seen_tasks:
                    continue
                seen_tasks.add(clean)
                merged_source_tasks.append(clean)

        # Append the auto-managed Source section so it always reflects truth
        # (LLM is told to leave it blank). Counts let humans audit.
        skill_md = self._append_source_section(
            skill_md,
            source=source,
            event_count=sum(
                1 for e in all_events
                if e.get("event_type") in ("tool_error", "tool_recovery")
            ),
            source_tasks=merged_source_tasks,
        )

        # "No Execution, No Memory" gate: refuse skeletons / placeholders /
        # missing gotchas. LLMs love to hedge with textbook fluff; we don't
        # install those.
        is_valid, reason = self._validate_skill_content(skill_md)
        if not is_valid:
            logger.info(f"Skill spawn rejected by validator: {reason}")
            return None

        skill_dir = generated_skills_dir / slug
        gate_result = await SkillAdmissionGate(
            critics=list(admission_critics or [])
        ).evaluate(
            skill_md,
            task="admit generated skill before shadow install",
            source_experience=source,
        )
        if write_provenance:
            append_provenance_event(
                skill_dir,
                gate_result.to_provenance_event(
                    event="admission",
                    skill_name=slug,
                    stage="spawn",
                    source_experience=source,
                    source_events=[c.get("filename", "") for c in candidates if c.get("filename")],
                ),
            )
        if not gate_result.approved:
            logger.info(
                f"Skill spawn rejected by admission gate: {slug} "
                f"(rejected_by={gate_result.rejected_by})"
            )
            return None

        # Install
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

        # Refresh the L1 keyword index so newly-spawned skills are discoverable
        # without semantic recall. INDEX.md is regenerated from all skill
        # frontmatter on every spawn.
        self.rebuild_index(generated_skills_dir)

        logger.info(f"Installed shadow skill: {slug} from experience '{source}'")
        return slug

    async def maybe_update_skill_state(
        self,
        model: Any,
        skill_dir: Path,
        checkpoint_interval: int = 5,
        rollback_consecutive_failures: int = 2,
        promotion_critics: Optional[List[Any]] = None,
        repair_critics: Optional[List[Any]] = None,
        write_provenance: bool = True,
        maintain_failed_skills: bool = False,
        max_repair_attempts: int = 3,
    ) -> Optional[str]:
        """Judge skill state from accumulated episodes at checkpoint.

        Only runs when total_episodes is a multiple of checkpoint_interval,
        or when consecutive_failures >= rollback_consecutive_failures.

        Args:
            model: LLM model instance.
            skill_dir: Path to generated_skills/{slug}/.
            checkpoint_interval: Run judgment every N episodes.
            rollback_consecutive_failures: Auto-rollback threshold.
            promotion_critics: Optional VaG critics run before promoting a
                shadow skill into active/auto status.
            repair_critics: Optional VaG critics run before accepting an LLM
                maintenance repair. Falls back to promotion_critics for direct
                manager calls.
            write_provenance: Whether to append lifecycle events to
                provenance.jsonl beside the skill.
            maintain_failed_skills: If True, repeated failures trigger an LLM
                repair-or-discard pass instead of deterministic rollback.
            max_repair_attempts: Retire the skill after this many failed
                maintenance repairs.

        Returns:
            Decision string, or None if not at checkpoint.
        """
        meta_path = skill_dir / "meta.json"
        episodes_path = skill_dir / "episodes.jsonl"
        skill_md_path = skill_dir / "SKILL.md"

        meta = self.read_meta(meta_path)
        if not meta or meta.get("status") == "rolled_back":
            return None

        total = meta.get("total_episodes", 0)
        consecutive_failures = meta.get("consecutive_failures", 0)

        # Auto-rollback on consecutive failures (deterministic, no LLM)
        if consecutive_failures >= rollback_consecutive_failures:
            if maintain_failed_skills:
                return await self._maintain_failed_skill(
                    model=model,
                    skill_dir=skill_dir,
                    meta=meta,
                    checkpoint_interval=checkpoint_interval,
                    repair_critics=repair_critics if repair_critics is not None else promotion_critics,
                    write_provenance=write_provenance,
                    max_repair_attempts=max_repair_attempts,
                )
            meta["status"] = "rolled_back"
            self.write_meta(meta_path, meta)
            self._disable_skill_md(skill_dir)
            if write_provenance:
                append_provenance_event(skill_dir, {
                    "event": "rollback",
                    "skill_name": meta.get("skill_name", skill_dir.name),
                    "reason": "consecutive_failures",
                    "consecutive_failures": consecutive_failures,
                    "approved": False,
                    "verdicts": [],
                })
            logger.info(
                f"Auto-rolled back skill {meta.get('skill_name')} "
                f"after {consecutive_failures} consecutive failures"
            )
            return "rollback"

        # Only run LLM judgment at checkpoint intervals
        if total < checkpoint_interval or total % checkpoint_interval != 0:
            return None

        # Read recent episodes
        episodes = self._read_recent_episodes(episodes_path, limit=checkpoint_interval)
        if not episodes:
            return None

        # Read SKILL.md content
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
            self._JUDGE_PROMPT
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

        response = await model.response([
            Message(role="user", content=prompt),
        ])
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
            gate_result = await SkillAdmissionGate(
                critics=list(promotion_critics or [])
            ).evaluate(
                skill_content,
                task="promote generated skill after runtime episodes",
                source_experience=str(meta.get("source_experience", "")),
            )
            if write_provenance:
                append_provenance_event(
                    skill_dir,
                    gate_result.to_provenance_event(
                        event="promotion",
                        skill_name=str(meta.get("skill_name", skill_dir.name)),
                        stage="promote",
                        source_experience=str(meta.get("source_experience", "")),
                    ),
                )
            if gate_result.approved:
                meta["status"] = "auto"
            else:
                decision = "keep_shadow"
        elif decision == "rollback":
            meta["status"] = "rolled_back"
            self._disable_skill_md(skill_dir)
            if write_provenance:
                append_provenance_event(skill_dir, {
                    "event": "rollback",
                    "skill_name": meta.get("skill_name", skill_dir.name),
                    "reason": result.get("reason", ""),
                    "approved": False,
                    "verdicts": [],
                })
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
                    # Validator rejected the revision. Don't write garbage
                    # over a working skill — fall back to keep_shadow so
                    # the next checkpoint can try again.
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

    # ── Deterministic helpers (no LLM) ────────────────────────────────────

    async def _maintain_failed_skill(
        self,
        model: Any,
        skill_dir: Path,
        meta: Dict,
        checkpoint_interval: int,
        repair_critics: Optional[List[Any]],
        write_provenance: bool,
        max_repair_attempts: int,
    ) -> str:
        """Ask the LLM to repair a repeatedly failing skill or retire it."""
        meta_path = skill_dir / "meta.json"
        skill_md_path = skill_dir / "SKILL.md"
        episodes_path = skill_dir / "episodes.jsonl"

        skill_content = ""
        if skill_md_path.exists():
            skill_content = await async_read_text(skill_md_path)
        episodes = self._read_recent_episodes(episodes_path, limit=checkpoint_interval)
        failures_text = "\n".join(
            "- "
            f"[{e.get('outcome', '?')}] "
            f"tool_errors={e.get('tool_errors', 0)} "
            f"user_corrected={e.get('user_corrected', False)} "
            f"query={str(e.get('query', ''))[:160]}"
            for e in episodes
        )
        prompt = (
            self._MAINTENANCE_PROMPT
            + f"Skill: {meta.get('skill_name', skill_dir.name)}\n"
            + f"Status: {meta.get('status', '?')}\n"
            + f"Consecutive failures: {meta.get('consecutive_failures', 0)}\n"
            + f"Repair attempts: {meta.get('repair_attempts', 0)}\n\n"
            + f"## Skill content\n{skill_content[:4000]}\n\n"
            + f"## Recent failures\n{failures_text}\n"
        )

        response = await model.response([
            Message(role="user", content=prompt),
        ])
        if not response or not response.content:
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason="maintenance model returned empty response",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
            )

        text = _strip_code_fences(response.content)
        if text.strip().upper().startswith("DISCARD"):
            reason = text.strip()[len("DISCARD"):].strip(" :-") or "discarded by maintenance model"
            return self._retire_skill(
                skill_dir,
                meta,
                reason=reason,
                write_provenance=write_provenance,
            )

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason="maintenance model returned invalid JSON",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
            )
        if not isinstance(result, dict):
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason="maintenance model returned non-object JSON",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
            )

        decision = str(result.get("decision", "")).lower()
        reason = str(result.get("reason", "") or "no reason provided")
        if decision == "discard":
            return self._retire_skill(
                skill_dir,
                meta,
                reason=reason,
                write_provenance=write_provenance,
            )
        if decision != "repair":
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason=f"unsupported maintenance decision: {decision!r}",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
            )

        revised_md = result.get("revised_skill_md")
        if not isinstance(revised_md, str) or not revised_md.strip():
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason="repair decision missing revised_skill_md",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
            )

        revised_md = _normalize_skill_md(revised_md)
        revised_md = self._append_source_section(
            revised_md,
            source=meta.get("source_experience", ""),
            event_count=meta.get("gotchas_hit_count", 0)
            + meta.get("new_gotchas_seen", 0),
        )
        is_valid, validation_reason = self._validate_skill_content(revised_md)
        if not is_valid:
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason=f"repaired skill failed validator: {validation_reason}",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
            )

        gate_result = await SkillAdmissionGate(
            critics=list(repair_critics or [])
        ).evaluate(
            revised_md,
            task="repair repeatedly failing generated skill",
            source_experience=str(meta.get("source_experience", "")),
        )
        if write_provenance:
            append_provenance_event(
                skill_dir,
                gate_result.to_provenance_event(
                    event="repair",
                    skill_name=str(meta.get("skill_name", skill_dir.name)),
                    stage="maintenance",
                    source_experience=str(meta.get("source_experience", "")),
                ),
            )
        if not gate_result.approved:
            return self._record_failed_repair(
                skill_dir,
                meta,
                reason=f"repaired skill rejected by gate: {gate_result.rejected_by}",
                write_provenance=write_provenance,
                max_repair_attempts=max_repair_attempts,
                repair_event_written=write_provenance,
            )

        await async_write_text(skill_md_path, revised_md)
        meta["version"] = meta.get("version", 1) + 1
        meta["repair_attempts"] = 0
        meta["consecutive_failures"] = 0
        meta["last_maintenance_at"] = date.today().isoformat()
        meta["last_maintenance_reason"] = reason
        self.write_meta(meta_path, meta)
        logger.info(f"Skill {meta.get('skill_name')}: repaired after repeated failures")
        return "repair"

    def _record_failed_repair(
        self,
        skill_dir: Path,
        meta: Dict,
        reason: str,
        write_provenance: bool,
        max_repair_attempts: int,
        repair_event_written: bool = False,
    ) -> str:
        """Track a failed maintenance repair and retire after the budget."""
        attempts = meta.get("repair_attempts", 0) + 1
        meta["repair_attempts"] = attempts
        meta["last_maintenance_at"] = date.today().isoformat()
        meta["last_maintenance_reason"] = reason
        if write_provenance and not repair_event_written:
            append_provenance_event(skill_dir, {
                "event": "repair",
                "skill_name": meta.get("skill_name", skill_dir.name),
                "approved": False,
                "reason": reason,
                "repair_attempts": attempts,
                "verdicts": [],
            })
        if attempts >= max_repair_attempts:
            return self._retire_skill(
                skill_dir,
                meta,
                reason=reason,
                write_provenance=write_provenance,
            )
        self.write_meta(skill_dir / "meta.json", meta)
        return "keep_shadow"

    def _retire_skill(
        self,
        skill_dir: Path,
        meta: Dict,
        reason: str,
        write_provenance: bool,
    ) -> str:
        """Retire/discard a generated skill so it no longer enters runtime."""
        meta["status"] = "retired"
        meta["retired_at"] = date.today().isoformat()
        meta["retire_reason"] = reason
        self.write_meta(skill_dir / "meta.json", meta)
        self._disable_skill_md(skill_dir)
        if write_provenance:
            append_provenance_event(skill_dir, {
                "event": "discard",
                "skill_name": meta.get("skill_name", skill_dir.name),
                "reason": reason,
                "approved": False,
                "verdicts": [],
            })
        logger.info(f"Retired skill {meta.get('skill_name')}: {reason}")
        return "retired"

    @staticmethod
    def _format_gotchas_block(gotchas: Any) -> Optional[str]:
        """Normalize gotcha updates into markdown bullet lines."""
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
        """Replace the one-line summary between frontmatter and the first section."""
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
        """Replace a markdown H2 section body while keeping other sections intact."""
        replacement = f"## {heading}\n{body.strip()}\n\n"
        pattern = re.compile(
            rf"(?ms)^##\s*{re.escape(heading)}\s*\n.*?(?=^##\s|\Z)"
        )
        if pattern.search(skill_md):
            return pattern.sub(replacement, skill_md, count=1)
        return skill_md.rstrip() + "\n\n" + replacement

    @classmethod
    def _apply_section_updates(cls, skill_md: str, section_updates: Dict[str, Any]) -> Optional[str]:
        """Apply structured section updates to the current SKILL.md content."""
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
            updated = cls._replace_named_section(
                updated, "Minimal Example", minimal_example
            )
            changed = True

        return updated if changed else None

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
        """Append a runtime episode to episodes.jsonl.

        Args:
            episodes_path: Path to episodes.jsonl file.
            outcome: "success" or "failure".
            query: User query that triggered this run.
            tool_errors: Number of tool errors in this run.
            user_corrected: Whether user corrected the agent.
            skill_followed: Whether the agent actually followed the skill's
                guidance (False = shadow skill was loaded but ignored).
            skill_gotchas_hit: Gotchas (by symptom) the agent encountered
                that the skill explicitly warned about — strong evidence
                the skill is paying its keep, drives the ``promote`` signal.
            new_gotchas_found: New failure modes that appeared during this
                run but the skill does not yet cover — drives the ``revise``
                signal in the checkpoint judge.
        """
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
        """Update meta.json counters after an episode.

        Args:
            meta_path: Path to meta.json.
            outcome: "success" or "failure".
            skill_gotchas_hit: Gotchas the agent ran into that the skill
                already warned about. Each one bumps ``gotchas_hit_count``,
                which the checkpoint judge reads to decide promote.
            new_gotchas_found: Gotchas not yet covered by the skill. Bumps
                ``new_gotchas_seen``, which drives the revise signal.

        Returns:
            Updated meta dict.
        """
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
        """Read meta.json for a generated skill."""
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def write_meta(meta_path: Path, meta: Dict) -> None:
        """Write/update meta.json."""
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
        """Scan experience .md files and return cards meeting upgrade threshold.

        Args:
            exp_dir: Experience directory containing .md files.
            min_repeat_count: Minimum repeat_count to qualify.
            min_tier: Minimum tier ("hot" means only hot, "warm" means hot+warm).

        Returns:
            List of dicts with title, content, repeat_count, type, tier.
        """
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

            if repeat_count < min_repeat_count:
                continue
            if tier not in allowed_tiers:
                continue

            # Strip frontmatter for content
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
        """List all generated skills with their status.

        Returns:
            List of dicts with skill_name, status, source_experience, etc.
        """
        if not generated_skills_dir.exists():
            return []

        skills = []
        for skill_dir in sorted(generated_skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            meta_path = skill_dir / "meta.json"
            meta = SkillEvolutionManager.read_meta(meta_path)
            if meta:
                skills.append(meta)
        return skills

    # ── Private ───────────────────────────────────────────────────────────

    # Patterns the validator rejects in generated SKILL.md.
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
        """Apply the No-Execution-No-Memory rules to a generated skill body.

        Returns ``(is_valid, reason)``. Callers reject the skill on False.

        Rules:
        1. Body must contain a Gotchas section (⚠️ marker or heading).
        2. Body must NOT contain placeholder markers (TODO / <your_x_here>).
        3. Body must NOT contain the textbook headings the gotcha-first
           prompt explicitly forbids.
        4. Code blocks must look like real examples — average line length
           ≥ 10 chars (rejects skeletons like ``def foo():\\n    pass``).
        """
        if not skill_md or not skill_md.strip():
            return False, "empty skill content"

        # Strip frontmatter for body checks (forbidden headings live in body).
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

    # Token-budget caps for source-task injection. The spawn prompt
    # already carries cards + raw event evidence; weak models (e.g.
    # GLM-4-Flash) start judging ``action=ignore`` if the prompt grows
    # past a few KB. These caps bound the worst case at ≤ 600 chars per
    # candidate (3 tasks × ~200 chars).
    _SOURCE_TASKS_PER_CARD = 3
    _SOURCE_TASK_DISPLAY_LEN = 200

    @classmethod
    def _format_card_for_prompt(cls, card: Dict) -> str:
        """Render one candidate card section for the spawn prompt.

        Includes a "Source tasks (samples)" subsection so the LLM can
        write a more grounded ``description`` / ``when-to-use``. Capped
        per ``_SOURCE_TASKS_PER_CARD`` and ``_SOURCE_TASK_DISPLAY_LEN``
        to control token usage on weak models.
        """
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
    def _index_events_once(all_events: List[Dict[str, Any]]) -> "_EventIndex":
        """Single O(N) scan of events.jsonl into per-bucket indices.

        Used by both the per-candidate recovery gate and the per-candidate
        evidence builder; without this, large workspaces would re-scan all
        events O(C × N) times during a single spawn decision.
        """
        idx = _EventIndex()
        for e in all_events:
            etype = e.get("event_type")
            if etype == "tool_error":
                tool = str(e.get("tool", ""))
                if tool:
                    idx.events_by_tool.setdefault(tool, []).append(e)
            elif etype == "tool_recovery":
                tool = str(e.get("tool", ""))
                if tool:
                    idx.events_by_tool.setdefault(tool, []).append(e)
                    idx.recovery_count_by_tool[tool] = (
                        idx.recovery_count_by_tool.get(tool, 0) + 1
                    )
            elif etype == "correction_classification":
                key = str(e.get("correction_key", ""))
                if not key:
                    continue
                idx.classifications_by_key.setdefault(key, []).append(e)
                if e.get("is_correction") and e.get("should_persist"):
                    idx.correction_count_by_key[key] = (
                        idx.correction_count_by_key.get(key, 0) + 1
                    )
        return idx

    @staticmethod
    def _candidate_recovery_count(
        candidate: Dict, idx: "_EventIndex",
    ) -> int:
        """How many *relevant* recovery / confirmation events exist for one card.

        - ``tool_error`` / ``success_pattern``: count ``tool_recovery``
          events whose ``tool`` matches the candidate's tool.
        - ``correction``: count ``correction_classification`` events with
          ``is_correction=True`` AND matching ``correction_key`` — each
          such event represents the user re-stating the same rule, which
          is the closest signal we have to "this rule is real".
        - Anything missing the join key (no tool / no correction_key)
          returns 0; the gate then refuses to spawn.
        """
        ctype = candidate.get("type", "")
        if ctype == "correction":
            key = (candidate.get("correction_key") or "").strip()
            return idx.correction_count_by_key.get(key, 0) if key else 0
        tool = (candidate.get("tool") or "").strip()
        return idx.recovery_count_by_tool.get(tool, 0) if tool else 0

    @staticmethod
    def _build_evidence_text(
        candidates: List[Dict],
        idx: "_EventIndex",
        per_candidate_limit: int = 5,
    ) -> str:
        """Render the per-candidate raw-event evidence block for the LLM.

        Matching is strictly per-candidate so the LLM cannot be fed
        cross-wired events:

        - ``tool_error`` / ``success_pattern``: events whose ``tool``
          equals the candidate's ``tool`` (strict equality).
        - ``correction``: ``correction_classification`` events whose
          ``correction_key`` equals the candidate's ``correction_key``.
          Two distinct corrections in the same workspace will therefore
          NEVER share evidence even though both fire ``user_message``
          events — this was the audit-failing bug from review.

        Each event is truncated so the prompt does not balloon when the
        workspace has thousands of events.
        """
        sections: List[str] = []
        for c in candidates:
            ctype = c.get("type", "")
            title = c.get("title", "")

            if ctype == "correction":
                key = (c.get("correction_key") or "").strip()
                matches = idx.classifications_by_key.get(key, []) if key else []
            else:
                tool = (c.get("tool") or "").strip()
                matches = idx.events_by_tool.get(tool, []) if tool else []
            if not matches:
                continue

            recent = matches[-per_candidate_limit:]
            lines = [f"### {title}"]
            for e in recent:
                etype = e.get("event_type", "?")
                if etype == "correction_classification":
                    user = str(e.get("user_message", ""))[:200]
                    rule = str(e.get("rule", ""))[:120]
                    lines.append(
                        f"- [correction] user={user!r} rule={rule!r}"
                    )
                else:
                    err = str(e.get("error", "") or e.get("note", ""))[:200]
                    lines.append(f"- [{etype}] {err}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    @staticmethod
    def _append_source_section(
        skill_md: str, source: str, event_count: int,
        source_tasks: Optional[List[str]] = None,
    ) -> str:
        """Replace / append the trailing ``## Source`` section.

        The prompt instructs the LLM to leave Source blank — we fill it in
        deterministically so audit info (origin card + raw event count +
        originating user tasks) is always accurate.
        """
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
        # Drop any existing Source section the LLM emitted to avoid dupes.
        cleaned = re.sub(
            r"\n##\s*Source\b.*\Z", "", skill_md.rstrip(), flags=re.DOTALL,
        )
        return cleaned + block

    # L1 INDEX.md format. Tiny header-only file scanned by humans / agents
    # who need to discover skills by keyword without semantic search cost.
    _INDEX_HEADER = (
        "# Generated Skills Index (L1)\n\n"
        "Auto-generated from skill frontmatter. Do not edit by hand.\n"
        "One row per active skill: `keywords -> name (description)`.\n\n"
    )

    @classmethod
    def rebuild_index(cls, generated_skills_dir: Path) -> Optional[Path]:
        """Regenerate ``INDEX.md`` listing every active generated skill.

        Returns the path to INDEX.md, or None if no skills exist. Skipped
        skills: ``rolled_back`` status or ``SKILL.md.disabled`` files.
        """
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
            if meta.get("status") == "rolled_back":
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
    def _read_recent_episodes(
        episodes_path: Path,
        limit: int = 10,
    ) -> List[Dict]:
        """Read last N episodes from episodes.jsonl."""
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


# ── Module-level helpers ─────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text
