# -*- coding: utf-8 -*-
"""Recover the ``/trigger args`` a user typed from a stored skill prompt.

``Skill.render_invocation()`` expands a skill into a long prompt, and that
expanded text is what the session log stores. Session titles and pickers only
need the typed line. This lives outside ``agentica.skills`` on purpose:
importing that package runs its ``__init__`` (loader, installer, curator), and
``SessionLog.session_preview`` walks every listed session at startup.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Markers of Skill.render_invocation(); keep in step with it and with
# web/src/lib/skillInvocation.ts.
_HEADER_RE = re.compile(r"Loading: ([^\n]+)\nBase directory: ([^\n]*)\n")
_TYPED_RE = re.compile(r"\n## Skill invocation\n\nThe user invoked this skill as: `([^`\n]+)`")


def invocation_label(prompt: str) -> Optional[str]:
    """The typed ``/trigger args``, or None when ``prompt`` is not a skill turn.

    A bare header (no arguments, so no trailer) is accepted only when the base
    directory is the skill's own folder, so a pasted document with the same two
    header lines is not relabelled.
    """
    head = _HEADER_RE.match(prompt)
    if head is None:
        return None
    name = head.group(1).strip()
    typed = _TYPED_RE.search(prompt)
    if typed is not None:
        label = typed.group(1).strip()
        return label if label.startswith("/") else f"/{label}"
    if Path(head.group(2).strip()).name.lower() == name.lower():
        return f"/{name}"
    return None
