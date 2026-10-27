# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Detect a new correction rule that contradicts a stored one.

A high-confidence correction is persisted on first sight. When the user later
says the opposite ("use pip" then "never pip, use uv"), the two cards differ
in title, so both land in the prompt with nothing saying they disagree.

After a correction card is written, up to five earlier correction cards
(those sharing a content term first, then the newest) go to one auxiliary
call that asks which of them contradict it. Rules may be Chinese, English or
mixed. Contradicting pairs are marked on both cards
(``conflict_with``) and the prompt shows them with a note to confirm with
the user. Neither card is overwritten: which rule is current is the user's
call, not this module's.
"""
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from agentica.experience.compiler import rule_terms
from agentica.model.message import Message
from agentica.prompts.memory import RULE_CONFLICT_PROMPT
from agentica.utils.log import logger

CONFLICT_MAX_CANDIDATES = 5
CONFLICT_TIMEOUT_S = 20.0
CONFLICT_MAX_OUTPUT_TOKENS = 1_024
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def shortlist_rivals(rule: str, cards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Up to ``CONFLICT_MAX_CANDIDATES`` cards to compare ``rule`` against.

    Cards sharing content terms come first, most shared first; the rest of
    the slots go to the newest other cards (``cards`` is newest first). Term
    overlap alone misses a contradiction across languages: "回复一律用中文"
    and "Always answer in English" share no term, and the classifier writes
    rules in either language.
    """
    mine = rule_terms(rule)
    overlap = [(len(mine & rule_terms(c["rule"])), i, c) for i, c in enumerate(cards)]
    lexical = [c for n, _, c in sorted((o for o in overlap if o[0] > 0), key=lambda o: (-o[0], o[1]))]
    rest = [c for c in cards if c not in lexical]
    return (lexical + rest)[:CONFLICT_MAX_CANDIDATES]


def parse_conflicts(text: str, allowed: set) -> Optional[List[str]]:
    """Ids from ``{"conflicts": [...]}`` limited to ``allowed``; None if unparseable."""
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    ids = data.get("conflicts") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        return None
    return [i for i in ids if isinstance(i, str) and i in allowed]


async def check_rule_conflicts(model, compiled_store, card_path: str, rule: str) -> List[str]:
    """Mark and return the stored cards that contradict the card at ``card_path``.

    No auxiliary call when there is no other stored correction. A failed or
    unparseable call marks nothing: the new card is already on
    disk, and a missed conflict is the state before this check existed.
    """
    filename = Path(card_path).name
    rivals = shortlist_rivals(rule, await compiled_store.correction_rules(exclude=filename))
    if not rivals or model is None:
        return []
    listing = "\n".join(f"- id={c['filename']}: {c['rule']}" for c in rivals)
    request = [
        Message(role="system", content=RULE_CONFLICT_PROMPT),
        Message(role="user", content=f"New rule: {rule}\n\nEarlier rules:\n{listing}"),
    ]
    side = model.side_call_copy(max_output_tokens=CONFLICT_MAX_OUTPUT_TOKENS)
    try:
        response = await asyncio.wait_for(side.response(messages=request), timeout=CONFLICT_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning("Rule conflict check timed out; card kept without a conflict mark")
        return []
    except Exception as e:
        # Degradation boundary: a provider error must not lose the card.
        logger.warning(f"Rule conflict check failed ({type(e).__name__}: {e}); card kept without a conflict mark")
        return []
    finally:
        await side.close_client()
    found = parse_conflicts(response.content or "", {c["filename"] for c in rivals})
    if not found:
        return []
    for other in found:
        await compiled_store.mark_conflict(filename, other)
    logger.debug(f"Correction {filename} conflicts with {found}")
    return found
