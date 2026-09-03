# -*- coding: utf-8 -*-
"""Render the session-guidance skills catalog (name + description, budgeted)."""
from typing import List, Optional, Sequence, Tuple

from agentica.skills.skill import Skill

# Catalog occupies this share of the model context window when the window is known.
SKILL_CATALOG_WINDOW_PERCENT = 2
# Fallback when the model has no context_window (chars, not tokens).
DEFAULT_SKILL_CATALOG_CHAR_BUDGET = 8000
# Front-load trigger language; tails get this cap before budget allocation.
MAX_SKILL_DESCRIPTION_CHARS = 1024
_DESC_TRUNCATION_SUFFIX = "..."

_HYGIENE_INTRO = (
    "# Skills\n"
    "\n"
    "Use a skill only when it clearly matches the current task. If the user names "
    "a skill (with a slash trigger or plain text) OR the task clearly matches a "
    "description below, load that skill for this turn. Multiple mentions mean use "
    "them all. Do not carry skills across turns unless re-mentioned."
)

_WORKFLOW = (
    "## Skill workflow\n"
    "- After deciding to use a skill, load it with `get_skill_info(skill_name)` "
    "and read the instructions completely before taking task actions.\n"
    "- Treat slash commands like `/<something>` as skill references and load the "
    "matching skill first.\n"
    "- Progressive disclosure applies to selecting files, not partially reading a "
    "selected SKILL.md: do not load unrelated references, scripts, or assets, and "
    "do not chase reference chains unless blocked.\n"
    "- Skills provide instructions, not executable actions.\n"
    "- Do not mention a skill without loading it.\n"
    "- Do not reload the same skill within the current turn."
)


def approx_catalog_tokens(text: str) -> int:
    """Byte/4 estimate; same order of magnitude as Codex `approx_token_count`."""
    if not text:
        return 0
    return (len(text.encode("utf-8")) + 3) // 4


def skill_catalog_budget(
    context_window: Optional[int] = None,
) -> Tuple[str, int]:
    """Return ``("tokens"|"chars", limit)`` for the whole catalog block."""
    # isinstance guard, same convention as runner/compress.py: the value
    # reaches here straight off a Model instance, and a test double (bare
    # MagicMock) or an uncoerced config string must degrade to the char
    # budget rather than raise inside agent construction.
    if isinstance(context_window, int) and not isinstance(context_window, bool) and context_window > 0:
        return (
            "tokens",
            max(1, (context_window * SKILL_CATALOG_WINDOW_PERCENT) // 100),
        )
    return "chars", DEFAULT_SKILL_CATALOG_CHAR_BUDGET


def cap_skill_description(description: str) -> str:
    if len(description) <= MAX_SKILL_DESCRIPTION_CHARS:
        return description
    keep = MAX_SKILL_DESCRIPTION_CHARS - len(_DESC_TRUNCATION_SUFFIX)
    return description[:keep] + _DESC_TRUNCATION_SUFFIX


def _cost(text: str, unit: str) -> int:
    if unit == "tokens":
        return approx_catalog_tokens(text)
    return len(text)


def _skill_line(skill: Skill, *, full: bool) -> str:
    trigger = f" (`{skill.trigger}`)" if skill.trigger else ""
    if not full:
        return f"- {skill.name}{trigger}"
    desc = cap_skill_description(skill.description or "")
    return f"- **{skill.name}**{trigger}: {desc}"


def render_skills_catalog(
    skills: Sequence[Skill],
    *,
    context_window: Optional[int] = None,
    include_workflow: bool = True,
) -> str:
    """Render a progressive-disclosure catalog.

    ``skills`` is already in display order (caller ranks). Every skill keeps a
    name line if the budget allows; descriptions are dropped from the tail
    first, then whole entries. The selected SKILL.md is never inlined here.
    """
    skills = list(skills)
    if not skills:
        return ""

    unit, limit = skill_catalog_budget(context_window)
    total = len(skills)
    fixed = _HYGIENE_INTRO
    if include_workflow:
        fixed = f"{fixed}\n\n{_WORKFLOW}"
    line_limit = max(0, limit - _cost(fixed, unit))

    def body(full_count: int, included: int) -> str:
        lines: List[str] = []
        for index, skill in enumerate(skills[:included]):
            lines.append(_skill_line(skill, full=index < full_count))
        omitted = total - included
        heading = (
            f"## Available skills ({included} of {total})"
            if omitted
            else "## Available skills"
        )
        parts = [heading, "\n".join(lines)]
        if omitted:
            parts.extend(
                [
                    "",
                    f"{omitted} additional skill(s) omitted to fit the catalog budget.",
                ]
            )
        return "\n".join(parts)

    def assemble(full_count: int, included: int) -> str:
        parts = [_HYGIENE_INTRO, "", body(full_count, included)]
        if include_workflow:
            parts.extend(["", _WORKFLOW])
        return "\n".join(parts) + "\n"

    def body_fits(full_count: int, included: int) -> bool:
        return _cost(body(full_count, included), unit) <= line_limit

    if body_fits(total, total):
        return assemble(total, total)

    for full_count in range(total - 1, -1, -1):
        if body_fits(full_count, total):
            return assemble(full_count, total)

    for included in range(total - 1, 0, -1):
        if body_fits(0, included):
            return assemble(0, included)

    return assemble(0, 1)
