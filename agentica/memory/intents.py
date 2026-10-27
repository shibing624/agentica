# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Intent tracking — what the user asked for across sessions, and where it ended.

Fact memory (MEMORY.md) deliberately drops in-progress work. That is exactly
what a new session most needs: "push it" or "continue" carries no content of
its own, the open loop from last time does. The tracker keeps those loops in
the cross-session index as items with a status:

- finished turns are queued durably (``SessionIndex.add_pending``), so a
  session that exits mid-batch is processed by the next session in the same
  project instead of being lost;
- one auxiliary call per batch sees the project's open items plus the new
  turns and returns new items and updates — an item is closed where the work
  is finished, not left to pile up as stale "pending";
- every item keeps the uuids of the user turns it came from, the pointer back
  to the raw transcript.
"""
import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from agentica.memory.session_index import INTENT_STATUSES, SessionIndex, clip_both_ends
from agentica.model.message import Message
from agentica.prompts.memory import INTENT_UPDATE_PROMPT
from agentica.utils.log import logger

# Open items older than this stop being offered to the tracker or to recall.
OPEN_INTENT_DAYS = 14
OPEN_INTENTS_FOR_UPDATE = 12
TURN_TEXT_MAX_CHARS = 1_500
UPDATE_MAX_OUTPUT_TOKENS = 2_048
MAX_NEW_INTENTS = 5
ASK_MAX_CHARS = 200
OUTCOME_MAX_CHARS = 300
ARTIFACTS_MAX = 5

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def build_update_request(open_items: List[Dict[str, Any]], turns: List[Dict[str, Any]]) -> List[Message]:
    lines = ["Open items in this project:"]
    if open_items:
        for item in open_items:
            nxt = f" — next: {item['next_step']}" if item["next_step"] else ""
            lines.append(f"#{item['id']}: {item['ask']} — {item['outcome'] or '(no outcome yet)'}{nxt}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("New turns:")
    for n, turn in enumerate(turns, 1):
        lines.append(f"[{n}] User: {clip_both_ends(turn['user'].strip(), TURN_TEXT_MAX_CHARS)}")
        reply = turn["assistant"].strip()
        if reply:
            lines.append(f"    Assistant: {clip_both_ends(reply, TURN_TEXT_MAX_CHARS)}")
    return [
        Message(role="system", content=INTENT_UPDATE_PROMPT),
        Message(role="user", content="\n".join(lines)),
    ]


def _str(value: Any, limit: int) -> str:
    return clip_both_ends(value.strip(), limit) if isinstance(value, str) else ""


def _turn_numbers(value: Any, n_turns: int) -> List[int]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, int) and 1 <= v <= n_turns]


def parse_update(text: str, open_ids: Set[int], n_turns: int) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Validated ``{"new": [...], "update": [...]}``; ``None`` if unparseable.

    Unknown ids, bad statuses and empty asks are dropped rather than failing
    the whole batch.
    """
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    new: List[Dict[str, Any]] = []
    for raw in data.get("new") or []:
        if not isinstance(raw, dict):
            continue
        ask = _str(raw.get("ask"), ASK_MAX_CHARS)
        status = raw.get("status") if raw.get("status") in INTENT_STATUSES else "open"
        if not ask:
            continue
        new.append({
            "ask": ask,
            "outcome": _str(raw.get("outcome"), OUTCOME_MAX_CHARS),
            "status": status,
            "next_step": _str(raw.get("next_step"), ASK_MAX_CHARS) if status == "open" else "",
            "artifacts": [a.strip() for a in raw.get("artifacts") or [] if isinstance(a, str) and a.strip()][
                :ARTIFACTS_MAX],
            "turns": _turn_numbers(raw.get("turns"), n_turns),
        })
        if len(new) >= MAX_NEW_INTENTS:
            break
    updates: List[Dict[str, Any]] = []
    for raw in data.get("update") or []:
        if not isinstance(raw, dict) or raw.get("id") not in open_ids:
            continue
        status = raw.get("status") if raw.get("status") in INTENT_STATUSES else None
        updates.append({
            "id": raw["id"],
            "outcome": _str(raw.get("outcome"), OUTCOME_MAX_CHARS) or None,
            "status": status,
            "next_step": "" if status in ("done", "dropped") else (_str(raw.get("next_step"), ASK_MAX_CHARS) or None),
            "turns": _turn_numbers(raw.get("turns"), n_turns),
        })
    return {"new": new, "update": updates}


def apply_update(
    index: SessionIndex,
    update: Dict[str, List[Dict[str, Any]]],
    *,
    session_id: str,
    work_dir: str,
    turn_uuids: List[Optional[str]],
) -> Dict[str, int]:
    def uuids_for(nums: List[int]) -> List[str]:
        return [u for u in (turn_uuids[n - 1] for n in nums) if u]

    for item in update["new"]:
        index.add_intent(
            work_dir=work_dir, session_id=session_id, ask=item["ask"], outcome=item["outcome"],
            status=item["status"], artifacts=item["artifacts"], next_step=item["next_step"],
            uuids=uuids_for(item["turns"]),
        )
    for item in update["update"]:
        index.update_intent(
            item["id"], outcome=item["outcome"], status=item["status"], next_step=item["next_step"],
            add_uuids=uuids_for(item["turns"]),
        )
    return {"new": len(update["new"]), "updated": len(update["update"])}


async def process_pending_intents(index: SessionIndex, model: Any, session_id: str, timeout: float) -> bool:
    """Run the tracker over ``session_id``'s queued turns. True when the queue was consumed.

    The turns are claimed first, so two processes never feed the same batch
    twice; a failed call releases them for the next attempt.
    """
    token = uuid.uuid4().hex
    turns = index.claim_pending(session_id, token)
    if not turns:
        return True
    work_dir = turns[0]["work_dir"]
    open_items = index.open_intents(work_dir, time.time() - OPEN_INTENT_DAYS * 86400, OPEN_INTENTS_FOR_UPDATE)
    side = model.side_call_copy(max_output_tokens=UPDATE_MAX_OUTPUT_TOKENS)
    request = build_update_request(open_items, turns)
    try:
        call = side.response(messages=request)
        response = await (asyncio.wait_for(call, timeout=timeout) if timeout > 0 else call)
    except asyncio.TimeoutError:
        logger.warning(f"Intent tracking timed out after {timeout:.1f}s; {len(turns)} turn(s) kept for retry")
        index.finish_claim(token, succeeded=False)
        return False
    except Exception as e:
        # Degradation boundary: a provider error must not fail the user's turn.
        logger.warning(f"Intent tracking failed ({type(e).__name__}: {e}); {len(turns)} turn(s) kept for retry")
        index.finish_claim(token, succeeded=False)
        return False
    finally:
        await side.close_client()
    parsed = parse_update(response.content or "", {i["id"] for i in open_items}, len(turns))
    if parsed is None:
        logger.warning("Intent tracking returned no parseable update; batch dropped")
        index.finish_claim(token, succeeded=True)
        return False
    turn_uuids = [index.turn_uuid(session_id, t["user"]) for t in turns]
    counts = apply_update(index, parsed, session_id=session_id, work_dir=work_dir, turn_uuids=turn_uuids)
    index.finish_claim(token, succeeded=True)
    logger.debug(f"Intent tracking ({session_id[:8]}): {counts['new']} new, {counts['updated']} updated")
    return True
