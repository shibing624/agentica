# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
Pure/stateless experience compiler.

Transforms raw captured data (tool errors, user messages, success patterns)
into compiled experience cards. No I/O, no state — takes inputs, returns outputs.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


# arch_v5 §8: title is the dedup key for `CompiledExperienceStore.write`.
# Letting the LLM freely name the title means semantically-equivalent rules
# end up as different files, so `repeat_count` never crosses the skill-spawn
# threshold. We derive the title deterministically from `rule` instead.
_RULE_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "be", "being", "been", "to", "of", "in",
    "on", "at", "for", "by", "with", "from", "as", "or", "and", "but", "not",
    "no", "always", "never", "should", "must", "when", "if", "then", "else",
    "this", "that", "you", "your", "please", "do", "does", "did", "make",
    "sure", "try", "use", "using", "used", "via",
    # Process / sequence noise that LLMs love to sprinkle in but that
    # carries no rule identity. Dropping these lets paraphrases like
    # "Step 1: check X before Y" and "always check X before Y" collapse
    # to the same dedup key.
    "step", "steps", "first", "second", "third", "fourth", "fifth",
    "follow", "follows", "following", "followed",
    "every", "any", "each", "all", "next", "once", "now", "again",
    "ensure", "ensures", "ensuring",
    "attempt", "attempts", "attempting", "attempted",
    "perform", "performs", "performing", "performed",
    "call", "calls", "calling", "called",
    "appear", "appears", "appearing", "appeared",
    "proceed", "proceeds", "proceeding", "proceeded",
    "remember", "remembers", "remembering", "remembered",
})

# Dedup-key budget. Only the first N stems form the filename — keep this
# small so different LLM rewordings of the same rule still collide.
# 4 stems is the sweet spot in practice: long enough to avoid false
# merges across distinct procedural rules, short enough that the
# common verb-object kernel survives noisy paraphrasing.
_TITLE_TOKEN_CAP = 4


def _summarize_error_for_title(error_msg: str) -> str:
    """Derive a short, readable identifier for a tool_error title.

    Examples:
        "Skill 'paper-analysis' not found. Available skills: ..." -> "skill_not_found"
        "FileNotFoundError: [Errno 2] No such file: '/foo'"       -> "FileNotFoundError"
        "HTTP 500 Internal Server Error"                          -> "http_500"
        ""                                                        -> "unknown"

    Strategy: prefer the exception class (token before ':' if it looks like an
    error class), else strip quoted args and pick the first 3 meaningful words.
    Always returns a snake_case token <= 30 chars, word-boundary-safe.
    """
    if not error_msg:
        return "unknown"
    head = error_msg.split("\n", 1)[0].strip()
    # If it looks like "ExceptionName: details", keep just the class name.
    if ":" in head:
        prefix, _, _ = head.partition(":")
        prefix = prefix.strip()
        if prefix and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*Error", prefix):
            return prefix[:30]
    # Strip quoted user-supplied args ("'paper-analysis'", '"foo.txt"') which
    # would otherwise turn every distinct arg into a new dedup bucket.
    stripped = re.sub(r"['\"][^'\"]{1,80}['\"]", "", head)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", stripped)
    # Keep error-meaningful stopwords like "not"/"no"/"never" — negation is
    # part of the error identity ("skill not found" != "skill found").
    _ERROR_TITLE_DROP = _RULE_STOPWORDS - {"not", "no", "never", "none"}
    keep = [t.lower() for t in tokens if t.lower() not in _ERROR_TITLE_DROP][:3]
    if not keep:
        return "unknown"
    return "_".join(keep)[:30]


def _stem(token: str) -> str:
    """Cheap suffix stripping so 'read' / 'reading' / 'reads' collide.

    Not a real stemmer — just trims the most common English inflections
    that flip an LLM's rewording into a different filename. Conservative:
    only strips when the surviving root is still >= 3 chars.
    """
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            if suf == "ies":
                base += "y"
            return base
    return token


def _rule_to_title(rule: str) -> str:
    """Deterministic snake_case title derived from rule text.

    Stable across LLM rewordings: ``"Always check directory exists before
    reading"`` and ``"Check that the directory exists before you read"`` both
    collapse onto a token set close enough that the resulting filenames will
    bump each other's ``repeat_count``.

    Returns empty string when nothing meaningful survives stop-word filtering;
    callers should treat that as "give up on this rule".
    """
    tokens = re.findall(r"[a-z]+", rule.lower())
    seen: set = set()
    keep: list = []
    for raw in tokens:
        if raw in _RULE_STOPWORDS or len(raw) <= 2:
            continue
        stem = _stem(raw)
        if stem in _RULE_STOPWORDS or len(stem) <= 2:
            continue
        if stem in seen:
            continue
        seen.add(stem)
        keep.append(stem)
    if not keep:
        return ""
    return "_".join(keep[:_TITLE_TOKEN_CAP])


@dataclass(frozen=True)
class CompiledCard:
    """An experience card compiled from raw events.

    Attributes:
        title: Unique identifier for the experience (snake_case)
        content: Full experience text (what happened + lesson)
        experience_type: One of "tool_error", "correction", "success_pattern"
        tool_name: Tool that triggered this experience (empty if N/A)
        source_task: User-facing task that triggered this experience (empty
            if unknown). Carried through events.jsonl into the card's
            ``source_tasks`` frontmatter list and into the spawn prompt so
            generated skills can be grounded in the actual originating task.
        correction_key: Stable association key for ``correction``-type cards
            (empty for tool_error / success_pattern). Used to wire
            ``correction_classification`` events to the correct candidate
            in spawn-prompt evidence selection. Generated by
            ``ExperienceCompiler.correction_key_from_rule(rule)`` and
            persisted to frontmatter so it survives file round-trips.
    """
    title: str
    content: str
    experience_type: str
    tool_name: str = ""
    source_task: str = ""
    correction_key: str = ""


class ExperienceCompiler:
    """Pure compiler: raw data -> compiled cards.

    No I/O, no mutable state, no model calls. All methods are static or
    class methods that take inputs and return outputs.

    Usage::

        cards = ExperienceCompiler.compile_tool_errors(errors)
        card = ExperienceCompiler.compile_success_pattern(successes)
        card = ExperienceCompiler.compile_correction(classification)
        key  = ExperienceCompiler.correction_key_from_rule(rule)
    """

    @staticmethod
    def correction_key_from_rule(rule: str) -> str:
        """Canonical, deterministic association key for a correction rule.

        Produced by the same normalization pipeline as the title (lower +
        word-tokens + stop-word filter + light suffix stemming + cap to
        ``_TITLE_TOKEN_CAP`` stems) so two LLM rewordings of the same rule
        collapse to the same key. Empty string means "rule was too vague /
        all stop-words" — caller must treat that as no-key.

        Exposed as a public static method so persistence layers
        (``CompiledExperienceStore``) and event writers (``hooks.py``) share
        one canonical implementation; never recompute this key by hand.
        """
        return _rule_to_title(rule)

    @staticmethod
    def compile_tool_errors(errors: List[Dict]) -> List[CompiledCard]:
        """Compile tool error dicts into experience cards.

        Each error produces one card. Dedup key is tool + error_type prefix,
        so different error types from the same tool remain separate.

        Args:
            errors: List of dicts with keys: tool, args, error, elapsed.

        Returns:
            List of CompiledCard, one per unique error.
        """
        cards = []
        for err in errors:
            tool = err.get("tool", "unknown")
            error_msg = err.get("error", "")
            error_type = _summarize_error_for_title(error_msg)
            title = f"{tool}_{error_type}"
            args_summary = str(err.get("args", {}))[:200]
            elapsed = err.get("elapsed", 0.0)

            content = (
                f"Tool `{tool}` failed.\n"
                f"Args: {args_summary}\n"
                f"Error: {error_msg}\n"
                f"Elapsed: {elapsed:.2f}s"
            )
            cards.append(CompiledCard(
                title=title,
                content=content,
                experience_type="tool_error",
                tool_name=tool,
                source_task=str(err.get("original_task", "") or "")[:500],
            ))
        return cards

    # Minimum number of DISTINCT tools required for a success pattern to be
    # worth remembering. Single-tool runs (e.g. 76× read_file) and 2-tool
    # combos provide no cross-tool insight and just inflate the prompt with
    # "this worked once" trivia.
    _SUCCESS_MIN_DISTINCT_TOOLS = 3

    @staticmethod
    def compile_success_pattern(successes: List[Dict]) -> Optional[CompiledCard]:
        """Compile a cross-tool success pattern.

        Only retained when the run actually demonstrates a non-trivial
        combination — at least ``_SUCCESS_MIN_DISTINCT_TOOLS`` distinct tools
        used together. Single-tool or 2-tool sequences are dropped because
        they teach the LLM nothing it doesn't already know from tool docs.

        Args:
            successes: List of dicts with keys: tool, elapsed.

        Returns:
            CompiledCard for cross-tool combos only; None for everything else.
        """
        if len(successes) < 3:
            return None

        tool_names = [s.get("tool", "unknown") for s in successes]
        distinct_tools = sorted(set(tool_names))
        if len(distinct_tools) < ExperienceCompiler._SUCCESS_MIN_DISTINCT_TOOLS:
            return None

        unique_tools = "_".join(distinct_tools)[:40]
        title = f"success_combo_{unique_tools}"

        # Order matters for cross-tool patterns: surface the dedupped order
        # the run actually used, not the raw repeat-heavy log.
        ordered_distinct: List[str] = []
        for tool in tool_names:
            if tool not in ordered_distinct:
                ordered_distinct.append(tool)

        content = (
            f"Successful tool combination ({len(successes)} calls across "
            f"{len(distinct_tools)} tools):\n"
            + "\n".join(f"- {t}" for t in ordered_distinct)
        )
        # All entries in `successes` come from the same run, so any of them
        # carries the same original_task. Take the first non-empty.
        source_task = ""
        for s in successes:
            t = str(s.get("original_task", "") or "")[:500]
            if t:
                source_task = t
                break
        return CompiledCard(
            title=title,
            content=content,
            experience_type="success_pattern",
            source_task=source_task,
        )

    @staticmethod
    def compile_correction(classification: Dict) -> Optional[CompiledCard]:
        """Compile a user correction from LLM classification output.

        Args:
            classification: Dict from LLM with keys: is_correction, confidence,
                title, rule, why, how_to_apply, category, scope, persist_target.

        Returns:
            CompiledCard if correction should be persisted as experience,
            None if not a correction or persist_target != "experience".
        """
        if not classification.get("is_correction", False):
            return None
        if not classification.get("should_persist", False):
            return None
        if classification.get("persist_target", "none") != "experience":
            return None

        rule = (classification.get("rule") or "").strip()
        if not rule:
            # rule is the dedup key under arch_v5 — without it the card has
            # no stable identity and would just bump itself per turn.
            return None
        correction_key = ExperienceCompiler.correction_key_from_rule(rule)
        if not correction_key:
            # Rule was all stop-words — refuse rather than bucket every
            # vague utterance under "user_correction".
            return None
        # title and correction_key are the same canonical string. Keeping
        # title aligned avoids file-name vs key drift; the dedicated
        # ``correction_key`` field is what downstream code matches on.
        title = correction_key
        why = classification.get("why", "")
        how_to_apply = classification.get("how_to_apply", "")
        category = classification.get("category", "")
        confidence = classification.get("confidence", 0.0)
        scope = classification.get("scope", "cross_session")

        content = (
            f"Rule: {rule}\n"
            f"Why: {why}\n"
            f"How to apply: {how_to_apply}\n"
            f"Category: {category}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Scope: {scope}"
        )
        return CompiledCard(
            title=title,
            content=content,
            experience_type="correction",
            source_task=str(classification.get("original_task", "") or "")[:500],
            correction_key=correction_key,
        )

    @staticmethod
    def build_raw_events(
        errors: List[Dict],
        user_msg: Optional[str],
        previous_assistant: Optional[str],
        successes: List[Dict],
        capture_corrections: bool = True,
    ) -> List[Dict]:
        """Build raw event dicts from captured data.

        Pure function: does not write anything, just constructs the event list.

        Args:
            errors: Tool error dicts.
            user_msg: Current user message (or None).
            previous_assistant: Previous assistant text (or None).
            successes: Successful tool call dicts.
            capture_corrections: Whether to include user message events.

        Returns:
            List of event dicts ready for ExperienceEventStore.append().
        """
        events = []

        for err in errors:
            events.append({
                "event_type": "tool_error",
                "tool": err.get("tool", ""),
                "args": str(err.get("args", {}))[:200],
                "error": err.get("error", ""),
                "elapsed": err.get("elapsed", 0.0),
            })

        if capture_corrections and user_msg:
            events.append({
                "event_type": "user_message",
                "user_message": user_msg[:500],
                "previous_assistant": (previous_assistant or "")[:500],
            })

        if len(successes) >= 3 and not errors:
            distinct = len({s.get("tool", "") for s in successes})
            # Only log success patterns that span multiple distinct tools.
            # Mirrors the gate inside compile_success_pattern so the raw event
            # log doesn't accumulate "read_file x76"-style entries we'd just
            # discard later.
            if distinct >= ExperienceCompiler._SUCCESS_MIN_DISTINCT_TOOLS:
                events.append({
                    "event_type": "success_pattern",
                    "tool_count": len(successes),
                    "tools": [s.get("tool", "") for s in successes],
                })

        return events
