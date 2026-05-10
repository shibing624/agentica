# -*- coding: utf-8 -*-
"""Pydantic schema for a generated ``SKILL.md`` candidate.

Used by ``SchemaCritic(SkillCandidate)`` inside the VaG admission gate to
enforce structural validity of LLM-generated skills.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agentica.skills.skill import Skill


class SkillCandidate(BaseModel):
    """Structured representation of a generated ``SKILL.md`` candidate."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    when_to_use: str = ""
    body: str = Field(min_length=1)
    source_experience: str = ""

    @classmethod
    def from_skill_md(
        cls,
        skill_md: str,
        source_experience: str = "",
    ) -> "SkillCandidate":
        """Parse SKILL.md frontmatter and body into a schema payload."""
        frontmatter, body = Skill._parse_frontmatter(skill_md.strip())
        return cls(
            name=str(frontmatter.get("name") or ""),
            description=str(frontmatter.get("description") or ""),
            when_to_use=str(
                frontmatter.get("when-to-use")
                or frontmatter.get("when_to_use")
                or ""
            ),
            body=body.strip(),
            source_experience=source_experience,
        )


__all__ = ["SkillCandidate"]
