# -*- coding: utf-8 -*-
"""Local copies of SDK skill-upgrade helpers used by the VaG maintenance loop.

We deliberately duplicate (rather than import via private underscore names)
so that VaG research code is decoupled from SDK refactors. The SDK is allowed
to rename or delete its private helpers without breaking the paper pipeline.

If/when the SDK promotes a stable ``agentica.experience.skill_upgrade.internals``
module, this file should be removed in favour of importing from there.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional


_FRONTMATTER_DASHES_RE = re.compile(r"^---\s*$", re.MULTILINE)
_FENCE_FRONTMATTER_RE = re.compile(
    r"\A\s*```(?:ya?ml)?\s*\n(?P<yaml>.*?)\n```\s*\n?",
    re.DOTALL,
)


def normalize_skill_md(text: str) -> str:
    """Coerce ``--- ... ---`` and ```` ```yaml ... ``` ```` SKILL.md frontmatter
    into canonical form. Idempotent on already-canonical input.
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


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return text


def read_recent_episodes(episodes_path: Path, limit: int = 10) -> List[Dict]:
    if not episodes_path.exists():
        return []
    try:
        lines = episodes_path.read_text(encoding="utf-8").strip().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    episodes: List[Dict] = []
    for line in lines[-limit:]:
        line = line.strip()
        if line:
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return episodes


def append_source_section(
    skill_md: str,
    source: str,
    event_count: int,
    source_tasks: Optional[List[str]] = None,
) -> str:
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


def validate_skill_content(skill_md: str) -> tuple:
    """Returns ``(is_valid, reason)`` for the No-Execution-No-Memory rules."""
    if not skill_md or not skill_md.strip():
        return False, "empty skill content"

    body = skill_md
    if body.startswith("---"):
        end = body.find("\n---", 3)
        if end != -1:
            body = body[end + len("\n---"):]

    if not _GOTCHA_RE.search(body):
        return False, "missing gotchas section (no ⚠️ markers or heading found)"

    m = _PLACEHOLDER_RE.search(body)
    if m:
        return False, f"contains placeholder/TODO marker: {m.group(0)!r}"

    m = _FORBIDDEN_HEADINGS_RE.search(body)
    if m:
        return False, (
            f"contains forbidden textbook heading {m.group(1)!r} "
            "(skill must be gotcha-first, not tutorial-style)"
        )

    for block in _CODE_BLOCK_RE.findall(body):
        non_empty = [ln for ln in block.split("\n") if ln.strip()]
        if not non_empty:
            continue
        avg_len = sum(len(ln) for ln in non_empty) / len(non_empty)
        if avg_len < 10:
            return False, (
                "code block has suspiciously short average line length"
            )

    return True, ""


def disable_skill_md(skill_dir: Path) -> None:
    """Rename SKILL.md → SKILL.md.disabled so SkillLoader won't discover it."""
    skill_md = skill_dir / "SKILL.md"
    if skill_md.exists():
        skill_md.rename(skill_dir / "SKILL.md.disabled")


def write_meta(meta_path: Path, meta: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


__all__ = [
    "normalize_skill_md",
    "strip_code_fences",
    "read_recent_episodes",
    "append_source_section",
    "validate_skill_content",
    "disable_skill_md",
    "write_meta",
]
