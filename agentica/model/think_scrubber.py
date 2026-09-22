# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Strip leaked model reasoning ("think") blocks from text.

Some models (DeepSeek-R1 and other reasoners served over OpenAI-compatible
endpoints) emit their chain-of-thought inline in the assistant ``content`` as
``<think>...</think>`` blocks instead of a separate reasoning field. If that
text is replayed into the next turn's history, the model can:

  - treat its own tentative scratch reasoning as established fact,
  - let stale hypotheses bias the new turn,
  - pollute compression summaries with scratchpad noise.

These helpers remove well-formed reasoning blocks. They are deliberately
conservative: only *closed* tag pairs are stripped (an unterminated ``<think>``
with no closing tag is left alone, since it may be truncated real content).
"""
import re

# Matches <think>..</think>, <thinking>..</thinking>, <reasoning>..</reasoning>,
# <scratchpad>..</scratchpad>, <reflection>..</reflection> (case-insensitive,
# spanning newlines). Backreference \1 ensures the close tag matches the open.
_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|scratchpad|reflection)\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
_EXCESS_BLANKLINES_RE = re.compile(r"\n{3,}")


def contains_reasoning_leak(text: str) -> bool:
    """True if ``text`` contains a closed reasoning block."""
    return bool(text) and "<" in text and _REASONING_BLOCK_RE.search(text) is not None


def scrub_reasoning(text: str) -> str:
    """Remove closed reasoning blocks from ``text``.

    Fast no-op when there's no ``<`` at all. Collapses the extra blank lines a
    removed block leaves behind, and strips surrounding whitespace.
    """
    if not text or "<" not in text:
        return text
    cleaned = _REASONING_BLOCK_RE.sub("", text)
    if cleaned == text:
        return text
    cleaned = _EXCESS_BLANKLINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def sanitize_assistant_content_for_history(content):
    """Scrub reasoning leakage from a single assistant ``content`` value.

    Strings are scrubbed directly; list/multimodal content has its text parts
    scrubbed in place; anything else is returned unchanged.
    """
    if isinstance(content, str):
        return scrub_reasoning(content)
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                part = {**part, "text": scrub_reasoning(part["text"])}
            out.append(part)
        return out
    return content
