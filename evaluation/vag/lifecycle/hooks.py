# -*- coding: utf-8 -*-
"""``SkillLifecycleHooks`` implementation that wires VaG gates and provenance
onto the SDK skill lifecycle.

Pass an instance to ``SkillUpgradeConfig.lifecycle_hooks`` to opt the agent
into VaG-paper-grade admission/promotion/repair behaviour without touching
SDK code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agentica.experience.skill_lifecycle_hooks import SkillLifecycleHooks
from agentica.utils.log import logger

from evaluation.vag.lifecycle.gate import SkillAdmissionGate
from evaluation.vag.lifecycle.maintenance import repair_or_discard
from evaluation.vag.lifecycle.provenance import append_provenance_event


class VaGLifecycleHooks(SkillLifecycleHooks):
    """VaG hooks. The same critic list is used for admission, promotion and
    repair gating by default (matches how the demo and tests configure VaG).

    Args:
        critics: Critic list for the admission gate.
        promotion_critics: Optional override for the promotion gate.
        repair_critics: Optional override for the repair gate.
        enable_maintenance: When True, ``on_failure_threshold`` triggers an
            LLM repair-or-discard pass. When False (default), the SDK falls
            back to deterministic rollback.
        max_repair_attempts: Hard cap on repair attempts before retiring.
        write_provenance: Append gate verdicts to ``provenance.jsonl``.
    """

    def __init__(
        self,
        critics: Optional[List[Any]] = None,
        promotion_critics: Optional[List[Any]] = None,
        repair_critics: Optional[List[Any]] = None,
        enable_maintenance: bool = False,
        max_repair_attempts: int = 3,
        write_provenance: bool = True,
    ) -> None:
        self._critics = list(critics or [])
        self._promotion_critics = list(promotion_critics or self._critics)
        self._repair_critics = list(repair_critics or self._critics)
        self._enable_maintenance = enable_maintenance
        self._max_repair_attempts = max_repair_attempts
        self._write_provenance = write_provenance

    async def gate_admission(
        self,
        skill_md: str,
        skill_dir: Path,
        meta: Dict[str, Any],
    ) -> bool:
        gate = SkillAdmissionGate(critics=list(self._critics))
        result = await gate.evaluate(
            skill_md,
            task="admit generated skill before shadow install",
            source_experience=str(meta.get("source_experience", "")),
        )
        if self._write_provenance:
            append_provenance_event(
                skill_dir,
                result.to_provenance_event(
                    event="admission",
                    skill_name=str(meta.get("skill_name", skill_dir.name)),
                    stage=str(meta.get("stage", "spawn")),
                    source_experience=str(meta.get("source_experience", "")),
                ),
            )
        if not result.approved:
            logger.info(
                f"Skill spawn rejected by admission gate: "
                f"{meta.get('skill_name')} (rejected_by={result.rejected_by})"
            )
        return result.approved

    async def gate_promotion(
        self,
        skill_md: str,
        skill_dir: Path,
        meta: Dict[str, Any],
    ) -> bool:
        gate = SkillAdmissionGate(critics=list(self._promotion_critics))
        result = await gate.evaluate(
            skill_md,
            task="promote generated skill after runtime episodes",
            source_experience=str(meta.get("source_experience", "")),
        )
        if self._write_provenance:
            append_provenance_event(
                skill_dir,
                result.to_provenance_event(
                    event="promotion",
                    skill_name=str(meta.get("skill_name", skill_dir.name)),
                    stage=str(meta.get("stage", "promote")),
                    source_experience=str(meta.get("source_experience", "")),
                ),
            )
        return result.approved

    async def on_failure_threshold(
        self,
        skill_dir: Path,
        meta: Dict[str, Any],
        model: Any,
    ) -> Optional[str]:
        if not self._enable_maintenance:
            return None
        return await repair_or_discard(
            skill_dir=skill_dir,
            meta=meta,
            model=model,
            critics=self._repair_critics,
            max_repair_attempts=self._max_repair_attempts,
            write_provenance=self._write_provenance,
        )


__all__ = ["VaGLifecycleHooks"]
