# -*- coding: utf-8 -*-
"""LLM-driven repair-or-discard maintenance for failing generated skills.

Used by VaG paper experiments only. NOT part of the SDK runtime path.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.experience.skill_upgrade import (
    SkillEvolutionManager,
    _normalize_skill_md,
    _strip_code_fences,
)
from agentica.model.message import Message
from agentica.prompts.experience.skill_upgrade import get_skill_maintenance_prompt
from agentica.utils.async_file import async_read_text, async_write_text
from agentica.utils.log import logger

from evaluation.vag.lifecycle.gate import SkillAdmissionGate
from evaluation.vag.lifecycle.provenance import append_provenance_event


async def repair_or_discard(
    skill_dir: Path,
    meta: Dict[str, Any],
    model: Any,
    critics: Optional[List[Any]] = None,
    max_repair_attempts: int = 3,
    write_provenance: bool = True,
    checkpoint_interval: int = 5,
) -> Optional[str]:
    """Ask the LLM to repair a repeatedly failing skill or retire it.

    Returns:
        - ``"repair"`` — SKILL.md rewritten and re-gated successfully.
        - ``"discard"`` — skill retired (status=retired, SKILL.md disabled).
        - ``"keep_shadow"`` — repair attempt failed but budget not exhausted.
        - ``None`` if the model returned an empty response (treat as no-op).
    """
    meta_path = skill_dir / "meta.json"
    skill_md_path = skill_dir / "SKILL.md"
    episodes_path = skill_dir / "episodes.jsonl"

    skill_content = ""
    if skill_md_path.exists():
        skill_content = await async_read_text(skill_md_path)
    episodes = SkillEvolutionManager._read_recent_episodes(
        episodes_path, limit=checkpoint_interval,
    )
    failures_text = "\n".join(
        "- "
        f"[{e.get('outcome', '?')}] "
        f"tool_errors={e.get('tool_errors', 0)} "
        f"user_corrected={e.get('user_corrected', False)} "
        f"query={str(e.get('query', ''))[:160]}"
        for e in episodes
    )
    prompt = (
        get_skill_maintenance_prompt()
        + f"Skill: {meta.get('skill_name', skill_dir.name)}\n"
        + f"Status: {meta.get('status', '?')}\n"
        + f"Consecutive failures: {meta.get('consecutive_failures', 0)}\n"
        + f"Repair attempts: {meta.get('repair_attempts', 0)}\n\n"
        + f"## Skill content\n{skill_content[:4000]}\n\n"
        + f"## Recent failures\n{failures_text}\n"
    )

    response = await model.response([Message(role="user", content=prompt)])
    if not response or not response.content:
        return _record_failed_repair(
            skill_dir, meta,
            reason="maintenance model returned empty response",
            write_provenance=write_provenance,
            max_repair_attempts=max_repair_attempts,
        )

    text = _strip_code_fences(response.content)
    if text.strip().upper().startswith("DISCARD"):
        reason = text.strip()[len("DISCARD"):].strip(" :-") or "discarded by maintenance model"
        return _retire_skill(
            skill_dir, meta, reason=reason, write_provenance=write_provenance,
        )

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return _record_failed_repair(
            skill_dir, meta,
            reason="maintenance model returned invalid JSON",
            write_provenance=write_provenance,
            max_repair_attempts=max_repair_attempts,
        )
    if not isinstance(result, dict):
        return _record_failed_repair(
            skill_dir, meta,
            reason="maintenance model returned non-object JSON",
            write_provenance=write_provenance,
            max_repair_attempts=max_repair_attempts,
        )

    decision = str(result.get("decision", "")).lower()
    reason = str(result.get("reason", "") or "no reason provided")
    if decision == "discard":
        return _retire_skill(
            skill_dir, meta, reason=reason, write_provenance=write_provenance,
        )
    if decision != "repair":
        return _record_failed_repair(
            skill_dir, meta,
            reason=f"unsupported maintenance decision: {decision!r}",
            write_provenance=write_provenance,
            max_repair_attempts=max_repair_attempts,
        )

    revised_md = result.get("revised_skill_md")
    if not isinstance(revised_md, str) or not revised_md.strip():
        return _record_failed_repair(
            skill_dir, meta,
            reason="repair decision missing revised_skill_md",
            write_provenance=write_provenance,
            max_repair_attempts=max_repair_attempts,
        )

    revised_md = _normalize_skill_md(revised_md)
    revised_md = SkillEvolutionManager._append_source_section(
        revised_md,
        source=meta.get("source_experience", ""),
        event_count=meta.get("gotchas_hit_count", 0)
        + meta.get("new_gotchas_seen", 0),
    )
    is_valid, validation_reason = SkillEvolutionManager._validate_skill_content(revised_md)
    if not is_valid:
        return _record_failed_repair(
            skill_dir, meta,
            reason=f"repaired skill failed validator: {validation_reason}",
            write_provenance=write_provenance,
            max_repair_attempts=max_repair_attempts,
        )

    gate_result = await SkillAdmissionGate(critics=list(critics or [])).evaluate(
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
        return _record_failed_repair(
            skill_dir, meta,
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
    SkillEvolutionManager.write_meta(meta_path, meta)
    logger.info(f"Skill {meta.get('skill_name')}: repaired after repeated failures")
    return "repair"


def _record_failed_repair(
    skill_dir: Path,
    meta: Dict[str, Any],
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
        return _retire_skill(
            skill_dir, meta, reason=reason, write_provenance=write_provenance,
        )
    SkillEvolutionManager.write_meta(skill_dir / "meta.json", meta)
    return "keep_shadow"


def _retire_skill(
    skill_dir: Path,
    meta: Dict[str, Any],
    reason: str,
    write_provenance: bool,
) -> str:
    """Retire/discard a generated skill so it no longer enters runtime."""
    meta["status"] = "retired"
    meta["retired_at"] = date.today().isoformat()
    meta["retire_reason"] = reason
    SkillEvolutionManager.write_meta(skill_dir / "meta.json", meta)
    SkillEvolutionManager._disable_skill_md(skill_dir)
    if write_provenance:
        append_provenance_event(skill_dir, {
            "event": "discard",
            "skill_name": meta.get("skill_name", skill_dir.name),
            "reason": reason,
            "approved": False,
            "verdicts": [],
        })
    logger.info(f"Retired skill {meta.get('skill_name')}: {reason}")
    return "discard"


__all__ = ["repair_or_discard"]
