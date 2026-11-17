# -*- coding: utf-8 -*-
"""The Web draws what the session API says; it does not re-read the log protocol.

Each pattern below is a piece of log protocol whose one reader lives in
``agentica.session.rows`` / ``agentica.session.trace``. A copy in ``web/src``
is the drift this guards against (titles, turns, steers and footers each once
had a second, diverging reader in the browser).
"""
import re
from pathlib import Path

WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"

# (what, pattern) — patterns match code, not prose in comments.
FORBIDDEN = [
    ("window preamble tag", re.compile(r"""["'`]<context_window>""")),
    ("steer guidance prefix", re.compile(r"""["'`]\[User guidance received""")),
    ("image marker regex", re.compile(r"\\\[图片: ")),
    ("tool error field", re.compile(r"\.(tool_call_error|is_error)\b")),
    ("footer matched by position", re.compile(r"\broundOffset\b")),
]


def _code(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    return "\n".join(line.split("//", 1)[0] if not line.lstrip().startswith("*") else "" for line in text.splitlines())


def test_web_does_not_parse_the_session_log_protocol():
    offenders = []
    for path in sorted(WEB_SRC.rglob("*.ts*")):
        if path.name.endswith(".d.ts"):
            continue
        code = _code(path.read_text(encoding="utf-8"))
        for what, pattern in FORBIDDEN:
            if pattern.search(code):
                offenders.append(f"{path.relative_to(WEB_SRC)}: {what}")
    assert offenders == []


def test_the_guard_catches_a_reintroduced_parser():
    sample = 'const OPEN = "<context_window>";\nif (e.tool_call_error) {}\n// "<context_window>" in a comment is fine'
    code = _code(sample)
    hits = [what for what, pattern in FORBIDDEN if pattern.search(code)]
    assert hits == ["window preamble tag", "tool error field"]
