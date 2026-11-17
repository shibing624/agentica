# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: What a session-log row *means* — one reading for every consumer.

The log is the replay source of truth, and a row carries more than prose: the
runtime folds a new-window preamble in front of the question a compaction
kept, marks a mid-run steer, marks a turn nobody typed. Each consumer used to
decide for itself which of those a row was — the trace page, search, the
``/resume`` picker, the session list, and the Web client each had their own
copy of "strip the preamble" and "is this a steer". The copies drifted: on this
machine's real logs the search copy left a legacy continuation line in the
question, the trace copy left the folded remaining-tokens fragment in it, and
the counters counted a bare preamble row as a user turn.

Everything that needs to know what a row is imports it from here. The Web
client gets the answer from the gateway (``row_payload``) and does not parse.

Two kinds of marker are read here, and nowhere else:

* explicit fields the writer sets (``automated``, ``steer``, ``cron_job``);
* the text the runtime itself folds into a user row's ``content``
  (``<context_window>`` preamble, steer guidance prefix, ``__RELAYED__``).
  The writer and this reader share the constants in
  ``agentica.compression.token_budget`` / below, so a change to what is
  written is a change to what is read, in one place.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from agentica.compression.token_budget import WINDOW_CONTINUATION_MARK

RELAYED_MARKER = "__RELAYED__"
STEER_GUIDANCE_PREFIX = "[User guidance received while you were working]"

_WINDOW_OPEN = "<context_window>"
_WINDOW_CLOSE = "</context_window>"
# Older logs word the continuation line differently ("... started without a
# conversation summary. ..."); every wording starts with this and ends the line.
_CONTINUATION_HEAD = "New context window started"

_NOTES_PLAIN = re.compile(r'<session_notes(?:\s+path="([^"]*)")?>([\s\S]*?)</session_notes>')
_NOTES_LINES = re.compile(r'<session_notes(?:\s+path="([^"]*)")?>\n([\s\S]*?)\n</session_notes>(?=\n|$)')


@dataclass(frozen=True)
class WindowPreamble:
    """A new context window's opening, split off a logged user row.

    ``notes`` / ``notes_path`` come from the row's ``<session_notes>``;
    ``rest`` is what follows every preamble layer — the question the
    compaction kept, or ``""`` for a bare preamble (an idle ``/compact``).
    """

    notes: str
    notes_path: str
    rest: str


@dataclass(frozen=True)
class RowView:
    """One user row, read once.

    ``automated`` and ``steer`` are independent: a peer message delivered
    mid-run is both. ``text`` is the prose the row stands for — the preamble and
    the steer prefix are gone, nothing else is touched (image markers stay; the
    Web client shows those as pictures). ``window`` is set when the row opened
    a new context window.
    """

    automated: bool
    steer: bool
    text: str
    window: Optional[WindowPreamble] = None

    @property
    def is_pure_window(self) -> bool:
        """A preamble with no question behind it: chrome, not a turn."""
        return self.window is not None and not self.text.strip()

    @property
    def starts_turn(self) -> bool:
        """Did a person begin a turn here? Not a steer, not a report, not chrome."""
        return not self.automated and not self.steer and not self.is_pure_window

    @property
    def opens_round(self) -> bool:
        """Does a model round follow this row? A report opens one; a steer joins the running one."""
        return not self.steer and not self.is_pure_window


def _peel_layer(text: str) -> str:
    """Remove one ``<context_window>`` layer and what the runtime folded with it."""
    s = text.lstrip()
    if not s.startswith(_WINDOW_OPEN):
        return text
    close = s.find(_WINDOW_CLOSE)
    if close < 0:
        return text
    s = s[close + len(_WINDOW_CLOSE):].lstrip()

    if s.startswith("<session_notes"):
        ends = [
            s.find("\n</session_notes>\n\n<dropped_span>\n"),
            s.find(f"\n</session_notes>\n\n{_CONTINUATION_HEAD}"),
            s.find(f"\n</session_notes>\n\n{_WINDOW_OPEN}"),
        ]
        end = next((e for e in ends if e >= 0), -1)
        if end >= 0:
            s = s[end + len("\n</session_notes>"):].lstrip()
        else:
            line = s.rfind("\n</session_notes>")
            if line >= 0:
                s = s[line + len("\n</session_notes>"):].lstrip()
            else:
                plain = s.find("</session_notes>")
                if plain >= 0:
                    s = s[plain + len("</session_notes>"):].lstrip()

    if s.startswith("<dropped_span>"):
        cont = s.rfind(f"\n</dropped_span>\n\n{_CONTINUATION_HEAD}")
        nxt = s.find(f"\n</dropped_span>\n\n{_WINDOW_OPEN}")
        if cont >= 0:
            s = s[cont + len("\n</dropped_span>\n\n"):].lstrip()
        elif nxt >= 0:
            s = s[nxt + len("\n</dropped_span>"):].lstrip()
        else:
            line = s.rfind("\n</dropped_span>")
            if line >= 0:
                s = s[line + len("\n</dropped_span>"):].lstrip()
            else:
                plain = s.find("</dropped_span>")
                if plain >= 0:
                    s = s[plain + len("</dropped_span>"):].lstrip()

    if s.startswith(_CONTINUATION_HEAD):
        nl = s.find("\n")
        s = s[nl + 1:].lstrip() if nl >= 0 else ""
    return s


def split_window_preamble(content: str) -> Optional[WindowPreamble]:
    """Split a new-window preamble off ``content``; ``None`` if it has none.

    Peels **every** layer — back-to-back cuts fold a second preamble in front
    of a row that already carried one — and drops the remaining-tokens
    fragment older logs carry after the question (it was folded into the
    message, then the window rolled).
    """
    if not isinstance(content, str) or not content.lstrip().startswith(_WINDOW_OPEN):
        return None
    m = _NOTES_LINES.search(content) or _NOTES_PLAIN.search(content)
    rest = content.lstrip()
    while rest.startswith(_WINDOW_OPEN):
        peeled = _peel_layer(rest)
        if peeled == rest:
            break
        rest = peeled
    tail = rest.rfind(f"\n\n{_WINDOW_OPEN}")
    if tail >= 0 and rest.endswith(_WINDOW_CLOSE):
        rest = rest[:tail]
    return WindowPreamble(
        notes=(m.group(2) if m else "").strip(),
        notes_path=(m.group(1) if m and m.group(1) else ""),
        rest=rest,
    )


def strip_window_preamble(content: str) -> str:
    """The prose behind a window preamble; ``content`` itself when it has none."""
    pre = split_window_preamble(content)
    return content if pre is None else pre.rest


def is_steer_text(content: str) -> bool:
    """Is this prose a steer by what the runtime folded into it?

    The ``steer`` flag is the writer's own mark and is checked by the caller;
    this covers the two prefixes older rows and relayed rows carry instead.
    """
    return content.startswith(RELAYED_MARKER) or content.startswith(STEER_GUIDANCE_PREFIX)


def clean_steer_text(raw: str) -> str:
    text = raw.lstrip()
    if text.startswith(RELAYED_MARKER):
        text = text[len(RELAYED_MARKER):].lstrip()
    if text.startswith(STEER_GUIDANCE_PREFIX):
        text = text[len(STEER_GUIDANCE_PREFIX):].lstrip("\n")
    return text.strip()


def classify_user(entry: Mapping[str, Any]) -> RowView:
    """Read a ``type == "user"`` row. The order is fixed here, once."""
    content = entry.get("content")
    content = content if isinstance(content, str) else ("" if content is None else str(content))
    if entry.get("automated") is True:
        # Nobody typed it, so nothing in it is peeled: it is shown as logged.
        return RowView(True, entry.get("steer") is True or is_steer_text(content), content)
    window = split_window_preamble(content)
    prose = content if window is None else window.rest
    steer = entry.get("steer") is True or is_steer_text(prose)
    return RowView(False, steer, clean_steer_text(prose) if steer else prose, window)


def starts_turn(entry: Mapping[str, Any]) -> bool:
    """Is ``entry`` a row a person's turn begins at?"""
    return entry.get("type") == "user" and classify_user(entry).starts_turn


def row_payload(entry: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The wire form of ``classify_user`` for a user row, ``None`` for any other.

    The gateway attaches this to each user row it serves; the Web client reads
    it instead of parsing ``content`` itself.
    """
    if entry.get("type") != "user":
        return None
    view = classify_user(entry)
    return {
        "automated": view.automated,
        "steer": view.steer,
        "text": view.text,
        "window": None if view.window is None else {
            "notes": view.window.notes,
            "notes_path": view.window.notes_path,
        },
    }
