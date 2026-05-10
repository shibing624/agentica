# -*- coding: utf-8 -*-
"""Conjunctive admission gate over heterogeneous critics.

Used by VaG paper experiments. NOT part of the SDK runtime path.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agentica.critic import Critic, CritiqueResult, SchemaCritic
from agentica.skills.skill import Skill


@dataclass(frozen=True)
class GateVerdict:
    """Normalized verdict from one critic in a skill admission gate."""

    critic_name: str
    approved: bool
    issues: str = ""
    evidence: str = ""

    @classmethod
    def from_critique(cls, result: CritiqueResult) -> "GateVerdict":
        return cls(
            critic_name=result.critic_name,
            approved=result.approved,
            issues=result.issues,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "critic": self.critic_name,
            "approved": self.approved,
            "issues": self.issues,
        }
        if self.evidence:
            data["evidence"] = self.evidence
        return data


@dataclass(frozen=True)
class SkillGateResult:
    """Full result of evaluating a generated skill against a critic set."""

    approved: bool
    verdicts: List[GateVerdict] = field(default_factory=list)
    rejected_by: List[str] = field(default_factory=list)
    fingerprint: str = ""

    def to_provenance_event(
        self,
        event: str,
        skill_name: str,
        stage: str,
        source_experience: str = "",
        source_events: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        """Render the gate result as one append-only provenance event."""
        return {
            "event": event,
            "skill_name": skill_name,
            "fingerprint": self.fingerprint,
            "source_experience": source_experience,
            "source_events": list(source_events or []),
            "stage": stage,
            "gate": [v.critic_name for v in self.verdicts],
            "approved": self.approved,
            "rejected_by": list(self.rejected_by),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


def skill_fingerprint(skill_md: str) -> str:
    """Stable SHA-256 fingerprint for a generated skill body."""
    digest = hashlib.sha256(skill_md.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass
class SkillAdmissionGate:
    """Conjunctive admission gate over heterogeneous critics.

    Every critic must approve for the skill to pass. ``SchemaCritic``
    receives a JSON representation parsed from SKILL.md so Pydantic schemas
    can validate actual frontmatter and body. Other critics receive the
    original markdown.
    """

    critics: List[Critic] = field(default_factory=list)

    async def evaluate(
        self,
        skill_md: str,
        task: str = "",
        source_experience: str = "",
    ) -> SkillGateResult:
        fingerprint = skill_fingerprint(skill_md)
        if not self.critics:
            return SkillGateResult(
                approved=True,
                verdicts=[],
                rejected_by=[],
                fingerprint=fingerprint,
            )

        verdicts = await asyncio.gather(
            *[
                self._run_critic(critic, task, skill_md, source_experience)
                for critic in self.critics
            ]
        )
        rejected_by = [v.critic_name for v in verdicts if not v.approved]
        return SkillGateResult(
            approved=not rejected_by,
            verdicts=list(verdicts),
            rejected_by=rejected_by,
            fingerprint=fingerprint,
        )

    async def _run_critic(
        self,
        critic: Critic,
        task: str,
        skill_md: str,
        source_experience: str,
    ) -> GateVerdict:
        critic_name = critic.name
        try:
            answer = self._answer_for_critic(critic, skill_md, source_experience)
            result = await critic(task, answer)
        except Exception as exc:
            return GateVerdict(
                critic_name=critic_name,
                approved=False,
                issues=f"critic raised {type(exc).__name__}: {exc}",
            )
        verdict = GateVerdict.from_critique(result)
        if not verdict.critic_name:
            return GateVerdict(
                critic_name=critic_name,
                approved=verdict.approved,
                issues=verdict.issues,
                evidence=verdict.evidence,
            )
        return verdict

    @staticmethod
    def _answer_for_critic(
        critic: Critic,
        skill_md: str,
        source_experience: str,
    ) -> str:
        if isinstance(critic, SchemaCritic):
            frontmatter, body = Skill._parse_frontmatter(skill_md.strip())
            return json.dumps({
                "name": str(frontmatter.get("name") or ""),
                "description": str(frontmatter.get("description") or ""),
                "when_to_use": str(
                    frontmatter.get("when-to-use")
                    or frontmatter.get("when_to_use")
                    or ""
                ),
                "body": body.strip(),
                "source_experience": source_experience,
            })
        return skill_md


__all__ = [
    "GateVerdict",
    "SkillGateResult",
    "SkillAdmissionGate",
    "skill_fingerprint",
]
