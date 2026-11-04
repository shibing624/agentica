# -*- coding: utf-8 -*-
"""The web recognises a stored skill turn by markers in render_invocation().

web/src/lib/skillInvocation.ts parses ``Loading: <name>``, ``## Skill
invocation`` and ``Argument: <args>`` to show a chip instead of the whole
SKILL.md. Changing those strings here without the web silently brings the
long bubble back; this test pins them.
"""
import re
from pathlib import Path

from agentica.skills.skill import Skill


def _skill(tmp_path: Path) -> Skill:
    d = tmp_path / "code-review"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: code-review\ndescription: review code\nargument-hint: a path\n---\n\n# Review\n\nRead the diff.\n",
        encoding="utf-8",
    )
    return Skill.from_skill_md(d / "SKILL.md")


def test_header_and_argument_markers(tmp_path):
    out = _skill(tmp_path).render_invocation("src/app.py 的改动")
    assert re.match(r"^Loading: code-review\nBase directory: [^\n]*\n", out)
    inv = out.rindex("\n## Skill invocation\n")
    arg = out.index("\nArgument: ", inv)
    rest = out[arg + len("\nArgument: "):]
    ends = [i for i in (rest.find("\nExpected argument: "), rest.find("\n\nThe argument names the target")) if i >= 0]
    assert rest[: min(ends)].strip() == "src/app.py 的改动"


def test_no_argument_is_just_the_header_and_body(tmp_path):
    out = _skill(tmp_path).render_invocation("")
    assert out.startswith("Loading: code-review\nBase directory: ")
    assert "## Skill invocation" not in out
