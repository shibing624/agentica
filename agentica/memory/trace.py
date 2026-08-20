# -*- coding: utf-8 -*-
"""Read-time Trace analysis over a Session JSONL.

The disk file is the only source of truth (history + resume + observation).
This module never writes. Visualization structure is derived here so the SPA
only draws. Old sessions with no ``type=event`` rows still list entries; they
do not get a guessed timeline.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agentica.cost_tracker import CostTracker


RELAYED_MARKER = "__RELAYED__"

# One expandable body is a display payload, not an archive: the file keeps the
# full row either way and the SPA holds every round of the session at once.
DETAIL_CHAR_CAP = 40000
SUMMARY_CHAR_CAP = 160


def _parse_ts(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _ms_between(start: str, end: str) -> Optional[int]:
    a = _parse_ts(start)
    b = _parse_ts(end)
    if a is None or b is None:
        return None
    return int(round((b - a) * 1000))


def _is_steering_user(entry: Dict[str, Any]) -> bool:
    if entry.get("type") != "user":
        return False
    content = entry.get("content") or ""
    return isinstance(content, str) and RELAYED_MARKER in content


def _empty_tokens() -> Dict[str, int]:
    return {"input": 0, "cacheRead": 0, "cacheWrite": 0, "output": 0}


def _clip(text: Any, cap: int) -> str:
    """One-line preview, cut at ``cap`` with a visible marker."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    flat = " ".join(text.split())
    if len(flat) <= cap:
        return flat
    return flat[:cap] + "…"


def _detail(text: Any) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) <= DETAIL_CHAR_CAP:
        return text
    return text[:DETAIL_CHAR_CAP] + f"\n… [{len(text) - DETAIL_CHAR_CAP} more characters]"


def _pretty_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _ensure_task(task_stats: Dict[int, Dict[str, Any]], ti: int) -> Dict[str, Any]:
    t = task_stats.get(ti)
    if t is None:
        t = {
            "taskIndex": ti,
            "messageFrom": -1,
            "messageTo": -1,
            "startTs": "",
            "endTs": "",
            "tokens": _empty_tokens(),
            "llmMs": 0,
        }
        task_stats[ti] = t
    return t


def analyze_entries(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Scan a session log once and return the Trace analysis payload.

    Invariants (aligned with PenguinHarness ``analyze()``, agentica envelope):
    - Model bars are serial: start = previous serial event; first = request_begin.
    - User messages occupy no bar.
    - Tools are their own lane (callTs → approvalTs → outputTs).
    - Tasks are grouped by sequential scan, never by timestamp.
    - A new user Prompt (not steering, not during compaction) starts a Task.
    - A request that issued a tool_call continues the same Task on the next
      request_begin. ``compact_boundary`` breaks continuation and is its own Task.
    - TPS denominator is LLM duration minus approval wait.
    - Context snapshot is the last non-compaction request's buckets, not a sum.
    - elapsedMs is the sum of Task durations, not file first-to-last.

    On top of the lanes it also returns the round view the page renders:
    ``meta`` (model / tool table / system prompt size), ``rounds`` (per user
    turn: phase breakdown, counts, tokens, cost, and the expandable entry list
    with tool arguments, tool results, thinking and the system prompt resolved
    from the conversation rows) and ``totals``.
    """
    messages = list(entries)
    requests: List[Dict[str, Any]] = []
    open_request: Optional[Dict[str, Any]] = None
    model_segments: List[Dict[str, Any]] = []
    tool_spans: List[Dict[str, Any]] = []
    open_spans: Dict[str, Dict[str, Any]] = {}
    prev_serial_ts: Optional[str] = None
    task_index = -1
    continuation = False
    saw_tool_call_this_request = False
    task_stats: Dict[int, Dict[str, Any]] = {}
    msg_task = [-1] * len(messages)
    pending_from: Optional[int] = None
    has_timeline = False
    reconnect_count = 0
    compaction_count = 0
    usage_trend: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "model": None,
        "provider": None,
        "contextWindow": None,
        "cwd": None,
        "gitBranch": None,
        "version": None,
        "tools": [],
        "systemPromptChars": 0,
    }

    for mi, entry in enumerate(messages):
        ts = entry.get("timestamp") or ""
        entry_type = entry.get("type", "")
        event_name = entry.get("name") if entry_type == "event" else None

        if meta["cwd"] is None and entry.get("cwd"):
            meta["cwd"] = entry.get("cwd")
        if meta["gitBranch"] is None and entry.get("git_branch"):
            meta["gitBranch"] = entry.get("git_branch")
        if meta["version"] is None and entry.get("version"):
            meta["version"] = entry.get("version")
        if event_name == "session_meta":
            for src, dst in (("model", "model"), ("provider", "provider"), ("context_window", "contextWindow")):
                if entry.get(src) is not None:
                    meta[dst] = entry.get(src)
        elif event_name == "tool_list_ready":
            tools = entry.get("tools")
            if isinstance(tools, list):
                meta["tools"] = [str(x) for x in tools]
        elif event_name == "system_prompt":
            meta["systemPromptChars"] = int(entry.get("chars") or len(entry.get("content") or ""))
        elif entry_type == "assistant" and meta["model"] is None and entry.get("model"):
            meta["model"] = entry.get("model")

        is_steering = _is_steering_user(entry)
        if is_steering:
            continuation = True

        starts_user_turn = entry_type == "user" and not is_steering
        if starts_user_turn:
            if pending_from is None:
                pending_from = mi
            continuation = False
        if pending_from is None:
            msg_task[mi] = task_index

        if entry_type == "compact_boundary":
            compaction_count += 1
            continuation = False
            task_index += 1
            t = _ensure_task(task_stats, task_index)
            t["compaction"] = True
            if t["startTs"] == "" and ts:
                t["startTs"] = ts
            msg_task[mi] = task_index
            pending_from = None
            continue

        if event_name == "request_begin":
            has_timeline = True
            prev_serial_ts = ts
            if not continuation:
                task_index += 1
            saw_tool_call_this_request = False
            open_request = {"beginTs": ts, "taskIndex": task_index}
            requests.append(open_request)
            t = _ensure_task(task_stats, task_index)
            if t["startTs"] == "":
                t["startTs"] = ts
            if pending_from is not None:
                for k in range(pending_from, mi):
                    msg_task[k] = task_index
                pending_from = None
            msg_task[mi] = task_index
            continue

        if event_name == "approval_decision":
            tool_call_id = entry.get("tool_call_id")
            if isinstance(tool_call_id, str):
                span = open_spans.get(tool_call_id)
                if span is not None and "approvalTs" not in span:
                    span["approvalTs"] = ts
                    decision = entry.get("decision")
                    if isinstance(decision, str):
                        span["decision"] = decision
                    if open_request is not None:
                        wait = _ms_between(span["callTs"], ts)
                        if wait is not None and wait > 0:
                            open_request["approvalWaitMs"] = (
                                open_request.get("approvalWaitMs") or 0
                            ) + wait
            continue

        if event_name == "request_end":
            status = entry.get("status") if isinstance(entry.get("status"), str) else None
            retryable = status in ("timeout", "malformed", "failed")
            prev_serial_ts = None
            continuation = saw_tool_call_this_request or retryable
            if retryable:
                reconnect_count += 1
            if open_request is not None:
                open_request["endTs"] = ts
                dur = _ms_between(open_request["beginTs"], ts)
                if dur is not None:
                    open_request["durationMs"] = dur
                    wait = open_request.get("approvalWaitMs") or 0
                    open_request["activeMs"] = max(0, dur - wait)
                if status is not None:
                    open_request["status"] = status
                if open_request.get("activeMs") is not None:
                    _ensure_task(task_stats, open_request["taskIndex"])["llmMs"] += (
                        open_request["activeMs"]
                    )
                open_request = None
            continue

        if event_name == "token_usage":
            request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
            session = entry.get("session") if isinstance(entry.get("session"), dict) else {}
            usage_trend.append({
                "ts": ts,
                "requestTotal": int(request.get("total") or 0),
                "sessionTotal": int(session.get("total") or 0),
            })
            t = _ensure_task(task_stats, task_index)
            cache_read = int(request.get("cache_read") or 0)
            cache_write = int(request.get("cache_write") or 0)
            output = int(request.get("output") or 0)
            total = int(request.get("total") or 0)
            # Sessions written before the emitter carried a disjoint `input`
            # bucket: derive it, so a cost figure is not silently zero.
            if request.get("input") is None:
                fresh_input = max(total - output - cache_read - cache_write, 0)
            else:
                fresh_input = int(request.get("input") or 0)
            t["tokens"]["input"] += fresh_input
            t["tokens"]["cacheRead"] += cache_read
            t["tokens"]["cacheWrite"] += cache_write
            t["tokens"]["output"] += output
            if not t.get("compaction"):
                t["context"] = {
                    "input": fresh_input,
                    "cacheRead": cache_read,
                    "cacheWrite": cache_write,
                    "output": output,
                }
            continue

        if event_name in ("thinking", "text", "tool_call") and prev_serial_ts:
            kind = event_name
            segment: Dict[str, Any] = {
                "kind": kind,
                "startTs": prev_serial_ts,
                "endTs": ts,
                "taskIndex": task_index,
                "key": str(entry.get("uuid") or f"{kind}:{mi}"),
            }
            if kind == "tool_call":
                tool_call_id = entry.get("tool_call_id")
                if isinstance(tool_call_id, str):
                    segment["toolCallId"] = tool_call_id
                tool_name = entry.get("tool_name")
                if isinstance(tool_name, str) and tool_name:
                    segment["name"] = tool_name
            model_segments.append(segment)
            prev_serial_ts = ts
            if kind == "tool_call":
                tool_call_id = entry.get("tool_call_id")
                stop_reason = entry.get("stop_reason")
                if isinstance(tool_call_id, str):
                    saw_tool_call_this_request = True
                    if stop_reason in (None, "completed"):
                        span = {
                            "toolCallId": tool_call_id,
                            "name": segment.get("name") or "",
                            "callTs": ts,
                            "taskIndex": task_index,
                            "key": tool_call_id,
                        }
                        open_spans[tool_call_id] = span
                        tool_spans.append(span)
            continue

        if entry_type == "tool":
            tool_call_id = entry.get("tool_call_id")
            if isinstance(tool_call_id, str):
                span = open_spans.get(tool_call_id)
                if span is not None and "outputTs" not in span:
                    span["outputTs"] = ts
            continue

    if pending_from is not None:
        if task_index < 0:
            task_index = 0
        t = _ensure_task(task_stats, task_index)
        for k in range(pending_from, len(messages)):
            if msg_task[k] < 0:
                msg_task[k] = task_index

    for mi, ti in enumerate(msg_task):
        if ti < 0:
            continue
        t = _ensure_task(task_stats, ti)
        if t["messageFrom"] < 0:
            t["messageFrom"] = mi
        t["messageTo"] = mi
        ts = messages[mi].get("timestamp") or ""
        if ts:
            t["endTs"] = ts

    tasks = [task_stats[k] for k in sorted(task_stats)]
    elapsed_ms = 0
    for t in tasks:
        dur = _ms_between(t["startTs"], t["endTs"]) if t["startTs"] else 0
        elapsed_ms += dur or 0

    rounds = _build_rounds(messages, msg_task, tasks, model_segments, tool_spans, requests, meta)
    return {
        "elapsedMs": elapsed_ms,
        "hasTimeline": has_timeline,
        "requests": requests,
        "tasks": tasks,
        "modelSegments": model_segments,
        "toolSpans": tool_spans,
        "otherSpans": [],
        "reconnectCount": reconnect_count,
        "compactionCount": compaction_count,
        "usageTrend": usage_trend,
        "messageTask": msg_task,
        "meta": meta,
        "rounds": rounds,
        "totals": _totals(rounds, requests, elapsed_ms, compaction_count),
    }


# ---------------------------------------------------------------------------
# Round view — what the Trace page draws per user turn.
#
# The lifecycle events are timestamps without bodies (a `thinking` row says
# only "reasoning ended here"), and the bodies live in the conversation rows
# that the same turn wrote. Pairing them is done here, once, so the SPA never
# has to guess which assistant message a marker belongs to: within a task the
# k-th `thinking` event is the k-th assistant row carrying reasoning, and the
# same for `text`. Tool calls need no counting — they carry the id.
# ---------------------------------------------------------------------------


def _round_indices(msg_task: List[int], task_count: int) -> Dict[int, List[int]]:
    """Group message indices by task, giving unassigned rows to a neighbour.

    The prelude events (`session_meta`, `tool_list_ready`, `system_prompt`) and
    anything else written before the turn's first `request_begin` scan as task
    -1, because the user row that opens the task is persisted later. They
    belong on screen to the round they configured, so they attach forward to
    the next task that does exist.
    """
    owner = list(msg_task)
    next_task = -1
    for mi in range(len(owner) - 1, -1, -1):
        if owner[mi] >= 0:
            next_task = owner[mi]
        elif next_task >= 0:
            owner[mi] = next_task
    if task_count > 0:
        last = task_count - 1
        for mi, ti in enumerate(owner):
            if ti < 0:
                owner[mi] = last
    grouped: Dict[int, List[int]] = {}
    for mi, ti in enumerate(owner):
        if ti >= 0:
            grouped.setdefault(ti, []).append(mi)
    return grouped


def _tool_call_arguments(messages: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """Map tool_call_id → (tool name, raw argument string) over the whole file."""
    out: Dict[str, Tuple[str, str]] = {}
    for entry in messages:
        if entry.get("type") != "assistant":
            continue
        for tc in entry.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            call_id = tc.get("id") or tc.get("tool_call_id")
            if not call_id:
                continue
            out[str(call_id)] = (
                str(tc.get("tool_name") or fn.get("name") or ""),
                str(fn.get("arguments") or ""),
            )
    return out


def _pair_markers(messages: List[Dict[str, Any]], indices: List[int]) -> Dict[int, int]:
    """Resolve `thinking` / `text` event rows to the assistant row they describe."""
    paired: Dict[int, int] = {}
    for name, field in (("thinking", "reasoning_content"), ("text", "content")):
        events = [
            mi for mi in indices
            if messages[mi].get("type") == "event" and messages[mi].get("name") == name
        ]
        bodies = [
            mi for mi in indices
            if messages[mi].get("type") == "assistant"
            and isinstance(messages[mi].get(field), str)
            and messages[mi][field].strip()
        ]
        for ev_mi, body_mi in zip(events, bodies):
            paired[ev_mi] = body_mi
    return paired


def _entry_row(
    messages: List[Dict[str, Any]],
    mi: int,
    paired: Dict[int, int],
    consumed: set,
    args_by_id: Dict[str, Tuple[str, str]],
) -> Optional[Dict[str, Any]]:
    entry = messages[mi]
    entry_type = entry.get("type", "")
    ts = entry.get("timestamp") or ""
    row: Dict[str, Any] = {"index": mi, "ts": ts, "summary": "", "detail": ""}

    if entry_type == "event":
        name = entry.get("name") or "event"
        row["kind"] = name
        if name == "session_meta":
            row["summary"] = " · ".join(
                str(x) for x in (entry.get("model"), entry.get("provider")) if x
            )
            row["detail"] = _pretty_json({
                k: entry.get(k) for k in ("model", "provider", "context_window", "tool_count")
            })
        elif name == "tool_list_ready":
            tools = entry.get("tools") if isinstance(entry.get("tools"), list) else []
            row["summary"] = f"{entry.get('count') or len(tools)} tools"
            row["detail"] = "\n".join(str(t) for t in tools)
        elif name == "system_prompt":
            content = entry.get("content") or ""
            row["summary"] = f"{entry.get('chars') or len(content)} chars"
            row["detail"] = _detail(content)
        elif name == "request_end":
            row["summary"] = f"status={entry.get('status') or 'unknown'}"
        elif name == "token_usage":
            request = entry.get("request") if isinstance(entry.get("request"), dict) else {}
            row["summary"] = " ".join(f"{k}={v}" for k, v in request.items())
            row["detail"] = _pretty_json(request)
        elif name == "approval_decision":
            row["summary"] = f"{entry.get('decision') or ''} {entry.get('tool_call_id') or ''}".strip()
        elif name == "tool_call":
            call_id = str(entry.get("tool_call_id") or "")
            tool_name, raw_args = args_by_id.get(call_id, ("", ""))
            row["toolName"] = entry.get("tool_name") or tool_name
            row["toolCallId"] = call_id
            row["summary"] = f"{row['toolName']} {_clip(raw_args, 100)}".strip()
            row["detail"] = _detail(_pretty_json(raw_args)) if raw_args else ""
        elif name in ("thinking", "text"):
            body_mi = paired.get(mi)
            if body_mi is not None:
                field = "reasoning_content" if name == "thinking" else "content"
                body = messages[body_mi].get(field) or ""
                row["summary"] = _clip(body, SUMMARY_CHAR_CAP)
                row["detail"] = _detail(body)
        return row

    if entry_type == "user":
        content = entry.get("content") or ""
        row["kind"] = "steering" if _is_steering_user(entry) else "user"
        row["summary"] = _clip(content, SUMMARY_CHAR_CAP)
        row["detail"] = _detail(content)
        return row

    if entry_type == "assistant":
        if mi in consumed:
            return None
        # No `text` / `thinking` marker claimed this row: either a session
        # recorded before the events existed, or a turn whose events were lost.
        # Showing the message itself is the only honest option.
        content = entry.get("content") or ""
        reasoning = entry.get("reasoning_content") or ""
        row["kind"] = "assistant"
        row["summary"] = _clip(content or reasoning, SUMMARY_CHAR_CAP)
        row["detail"] = _detail(f"{reasoning}\n\n---\n\n{content}" if reasoning and content else (content or reasoning))
        return row

    if entry_type in ("tool", "tool_audit"):
        content = entry.get("content") or ""
        row["kind"] = "tool_result"
        row["toolName"] = entry.get("tool_name") or ""
        row["toolCallId"] = entry.get("tool_call_id") or ""
        row["isError"] = bool(entry.get("is_error"))
        row["summary"] = _clip(content, SUMMARY_CHAR_CAP)
        row["detail"] = _detail(content)
        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        seconds = metrics.get("time")
        if isinstance(seconds, (int, float)):
            row["durationMs"] = int(round(seconds * 1000))
        return row

    if entry_type == "compact_boundary":
        summary = entry.get("summary") or ""
        row["kind"] = "compact_boundary"
        row["summary"] = _clip(summary, SUMMARY_CHAR_CAP)
        row["detail"] = _detail(summary)
        return row

    if entry_type == "goal":
        row["kind"] = "goal"
        row["detail"] = _pretty_json(entry.get("goal"))
        row["summary"] = _clip(row["detail"], SUMMARY_CHAR_CAP)
        return row

    return None


def _phase_breakdown(
    task: Dict[str, Any],
    model_segments: List[Dict[str, Any]],
    tool_spans: List[Dict[str, Any]],
    duration_ms: int,
) -> Dict[str, int]:
    """Split a round's wall clock into the phases the timeline bar stacks.

    `other` is the remainder — queueing, compression, the runner's own work —
    and is a residual rather than a measurement, so it is clamped at zero: a
    streamed reply and a parallel tool batch genuinely overlap, and the bar
    must not go negative when they do.
    """
    kind_to_phase = {"thinking": "thinking", "text": "text", "tool_call": "toolArgs"}
    phases = {"thinking": 0, "text": 0, "toolArgs": 0, "toolWait": 0, "toolExec": 0, "other": 0}
    ti = task["taskIndex"]
    for seg in model_segments:
        if seg["taskIndex"] != ti:
            continue
        phase = kind_to_phase.get(seg["kind"])
        if phase is None:
            continue
        phases[phase] += _ms_between(seg["startTs"], seg["endTs"]) or 0
    for span in tool_spans:
        if span["taskIndex"] != ti:
            continue
        approval_ts = span.get("approvalTs")
        if approval_ts:
            phases["toolWait"] += _ms_between(span["callTs"], approval_ts) or 0
        output_ts = span.get("outputTs")
        if output_ts:
            phases["toolExec"] += _ms_between(approval_ts or span["callTs"], output_ts) or 0
    phases["other"] = max(0, duration_ms - sum(phases.values()))
    return phases


def _build_rounds(
    messages: List[Dict[str, Any]],
    msg_task: List[int],
    tasks: List[Dict[str, Any]],
    model_segments: List[Dict[str, Any]],
    tool_spans: List[Dict[str, Any]],
    requests: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    grouped = _round_indices(msg_task, len(tasks))
    args_by_id = _tool_call_arguments(messages)
    model_id = meta.get("model")
    rounds: List[Dict[str, Any]] = []

    for task in tasks:
        ti = task["taskIndex"]
        indices = grouped.get(ti, [])
        paired = _pair_markers(messages, indices)
        consumed = set(paired.values())
        entries: List[Dict[str, Any]] = []
        for mi in indices:
            row = _entry_row(messages, mi, paired, consumed, args_by_id)
            if row is not None:
                entries.append(row)

        duration_ms = (_ms_between(task["startTs"], task["endTs"]) or 0) if task["startTs"] else 0
        task_requests = [r for r in requests if r["taskIndex"] == ti]
        tool_results = [
            messages[mi] for mi in indices if messages[mi].get("type") == "tool"
        ]
        tool_errors = sum(1 for r in tool_results if r.get("is_error"))
        tokens = task["tokens"]
        title = next((e["summary"] for e in entries if e["kind"] == "user"), "")
        if not title:
            title = "Context compaction" if task.get("compaction") else f"Round {ti + 1}"

        llm_ms = task["llmMs"]
        rounds.append({
            "taskIndex": ti,
            "title": title,
            "compaction": bool(task.get("compaction")),
            "startTs": task["startTs"],
            "endTs": task["endTs"],
            "durationMs": duration_ms,
            "llmMs": llm_ms,
            "requests": len(task_requests),
            "waitMs": sum(r.get("approvalWaitMs") or 0 for r in task_requests),
            "toolCalls": sum(1 for s in tool_spans if s["taskIndex"] == ti),
            "toolResults": len(tool_results),
            "toolErrors": tool_errors,
            "tokens": dict(tokens),
            "costUsd": _cost(model_id, tokens),
            "tps": (tokens["output"] / (llm_ms / 1000)) if llm_ms > 0 else 0.0,
            "phases": _phase_breakdown(task, model_segments, tool_spans, duration_ms),
            "entries": entries,
        })
    return rounds


def _cost(model_id: Optional[str], tokens: Dict[str, int]) -> Optional[float]:
    """USD estimate from the shared pricing table, or None without a model id."""
    if not model_id:
        return None
    tracker = CostTracker()
    return tracker.record(
        model_id=model_id,
        input_tokens=tokens.get("input", 0),
        output_tokens=tokens.get("output", 0),
        cache_read_tokens=tokens.get("cacheRead", 0),
        cache_write_tokens=tokens.get("cacheWrite", 0),
    )


def _totals(
    rounds: List[Dict[str, Any]],
    requests: List[Dict[str, Any]],
    elapsed_ms: int,
    compaction_count: int,
) -> Dict[str, Any]:
    tokens = _empty_tokens()
    for r in rounds:
        for key in tokens:
            tokens[key] += r["tokens"].get(key, 0)
    llm_ms = sum(r["llmMs"] for r in rounds)
    costs = [r["costUsd"] for r in rounds if r["costUsd"] is not None]
    tool_results = sum(r["toolResults"] for r in rounds)
    tool_errors = sum(r["toolErrors"] for r in rounds)
    return {
        "rounds": sum(1 for r in rounds if not r["compaction"]),
        "requests": len(requests),
        "toolCalls": sum(r["toolCalls"] for r in rounds),
        "toolOk": tool_results - tool_errors,
        "toolErrors": tool_errors,
        "compactions": compaction_count,
        "tokens": tokens,
        "elapsedMs": elapsed_ms,
        "llmMs": llm_ms,
        "waitMs": sum(r["waitMs"] for r in rounds),
        "costUsd": sum(costs) if costs else None,
        "tps": (tokens["output"] / (llm_ms / 1000)) if llm_ms > 0 else 0.0,
    }
