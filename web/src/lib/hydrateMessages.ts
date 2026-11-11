import type { ChatMsg, MsgPart } from "../store";
import { settleWork } from "./msgParts";
import { parseSkillTurn } from "./skillInvocation";

/**
 * Rebuild the chat transcript from a SessionLog JSONL dump.
 *
 * Live web turns write tool cards through SSE (`tool_call` / `tool_result`)
 * onto one assistant message. History used to keep only `user` / `assistant`
 * text, so CLI sessions (and a refreshed web turn) rendered as bare answers.
 * The log already has the missing pieces: `assistant.tool_calls` plus
 * `type: "tool"` rows. Fold each user turn into the same parts model the
 * live stream uses (think → tools → text).
 *
 * Lifecycle `type: "event"` rows are Trace markers (no arguments / result)
 * and are ignored here.
 */
export function eventsToChatMsgs(events: unknown[]): ChatMsg[] {
  const msgs: ChatMsg[] = [];
  let assistant: ChatMsg | null = null;
  let turnStartTs = 0;
  let turnEndTs = 0;
  // `trigger` of the boundary whose preamble has not been seen yet ("" = not
  // recorded); null once that preamble is consumed.
  let pendingTrigger: string | null = null;

  const closeAssistant = () => {
    if (!assistant) return;
    settleWork(assistant);
    if (assistant.durationMs == null && turnStartTs > 0 && turnEndTs > turnStartTs) {
      assistant.durationMs = turnEndTs - turnStartTs;
    }
    assistant = null;
  };

  for (const raw of events) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const type = String(e.type || "");
    const explicitTs = rawTsOf(e);
    if (type === "user") {
      let content = String(e.content || "");
      if (e.automated === true) {
        closeAssistant();
        const job = e.cron_job as { id?: unknown; name?: unknown } | undefined;
        msgs.push({
          role: "user",
          automated: true,
          content,
          ts: tsOf(e),
          ...(uuidOf(e) ? { uuid: uuidOf(e) } : {}),
          ...(job && typeof job.id === "string" ? { cronJob: { id: job.id, name: String(job.name || "") } } : {}),
        });
        continue;
      }
      const pre = splitWindowPreamble(content);
      if (pre) {
        closeAssistant();
        // A new window's preamble is chrome, not something the user said: it
        // becomes the compact card (with the notes it started from), and only
        // a question folded in behind it stays a user bubble.
        const last = msgs[msgs.length - 1];
        let card = last?.compact && last.compact.notes == null ? last : null;
        if (!card) {
          card = { role: "assistant", content: "", ts: tsOf(e), compact: {} };
          msgs.push(card);
        }
        Object.assign(card.compact!, { notes: pre.notes, notesPath: pre.notesPath || card.compact!.notesPath || "" });
        // Logs older than the boundary's `trigger`: an idle /compact folds no
        // question behind its preamble; an auto cut keeps the pending one.
        if (pendingTrigger === "" && !pre.rest) insertCompactCommand(msgs, card);
        else if (pendingTrigger === "") card.compact!.auto = true;
        pendingTrigger = null;
        if (!pre.rest) continue;
        content = pre.rest;
      }
      if (isSteerUserEvent(e, content)) {
        const clean = splitImageMarkers(cleanSteerContent(content)).text;
        if (!clean) continue;
        if (!turnStartTs && explicitTs) turnStartTs = explicitTs;
        if (explicitTs > turnEndTs) turnEndTs = explicitTs;
        assistant = ensureAssistant(msgs, assistant, tsOf(e));
        pushSteer(assistant, clean, tsOf(e));
        continue;
      }
      closeAssistant();
      const attached = splitImageMarkers(content);
      content = attached.text;
      const skill = parseSkillTurn(content);
      turnStartTs = explicitTs;
      turnEndTs = explicitTs;
      msgs.push({
        role: "user",
        content,
        ts: tsOf(e),
        ...(uuidOf(e) ? { uuid: uuidOf(e) } : {}),
        ...(skill ? { skill } : {}),
        ...(attached.previews.length ? { previews: attached.previews } : {}),
      });
      continue;
    }
    if (type === "assistant") {
      if (!assistantHasWork(e)) continue;
      if (!turnStartTs && explicitTs) turnStartTs = explicitTs;
      if (explicitTs > turnEndTs) turnEndTs = explicitTs;
      assistant = ensureAssistant(msgs, assistant, tsOf(e));
      applyAssistant(assistant, e);
      // The reply ends at its last row: fork / feedback address that one.
      if (uuidOf(e)) assistant.uuid = uuidOf(e);
      continue;
    }
    if (type === "tool" || type === "tool_audit") {
      if (!turnStartTs && explicitTs) turnStartTs = explicitTs;
      if (explicitTs > turnEndTs) turnEndTs = explicitTs;
      assistant = ensureAssistant(msgs, assistant, tsOf(e));
      applyToolResult(assistant, e);
      if (uuidOf(e)) assistant.uuid = uuidOf(e);
      continue;
    }
    if (type === "compact_boundary") {
      closeAssistant();
      const trigger = typeof e.trigger === "string" ? e.trigger : "";
      const notesPath = typeof e.notes_path === "string" ? e.notes_path : "";
      const card: ChatMsg = {
        role: "assistant",
        content: "",
        ts: tsOf(e),
        compact: { ...(trigger === "auto" ? { auto: true } : {}), ...(notesPath ? { notesPath } : {}) },
      };
      msgs.push(card);
      // The user's `/compact` is not a logged row; the boundary says it was one.
      if (trigger === "manual") insertCompactCommand(msgs, card);
      pendingTrigger = trigger;
    }
  }
  closeAssistant();
  return msgs;
}

const IMAGE_MARK_RE = /(?:^|\n\n)\[图片: ([0-9a-f]{64}\.(?:png|jpg|gif|webp))\]/g;
export const STEER_GUIDANCE_PREFIX = "[User guidance received while you were working]";
const FOLDED_STEER_RE = /(?:^|\n\n)\[User guidance received while you were working\]\n([\s\S]*?)(?=(?:\n\n\[User guidance received while you were working\]\n)|$)/g;

export function isSteerUserEvent(e: Record<string, unknown>, content = String(e.content || "")): boolean {
  if (content.trimStart().startsWith(WINDOW_OPEN)) return false;
  if (e.steer === true) return true;
  if (content.startsWith("__RELAYED__")) return true;
  if (content.startsWith(STEER_GUIDANCE_PREFIX)) return true;
  return false;
}

export function cleanSteerContent(raw: string): string {
  let text = raw.trimStart();
  if (text.startsWith("__RELAYED__")) {
    text = text.slice("__RELAYED__".length).trimStart();
  }
  if (text.startsWith(STEER_GUIDANCE_PREFIX)) {
    text = text.slice(STEER_GUIDANCE_PREFIX.length).replace(/^\n+/, "");
  }
  return text.trim();
}

function splitFoldedSteer(result: string): { toolResult: string; steers: string[] } {
  if (!result.includes(STEER_GUIDANCE_PREFIX)) return { toolResult: result, steers: [] };
  const steers: string[] = [];
  let cut = -1;
  for (const m of result.matchAll(FOLDED_STEER_RE)) {
    if (cut < 0) cut = m.index ?? 0;
    const text = (m[1] || "").trim();
    if (text) steers.push(text);
  }
  if (cut < 0) return { toolResult: result, steers: [] };
  return { toolResult: result.slice(0, cut).replace(/\s+$/, ""), steers };
}

/**
 * A web turn with pictures is logged as `text` + `[图片: <file>]` markers
 * (+ the media model's description when the base model cannot see images).
 * Everything from the first marker on is machine-added: show the pictures again
 * and keep only what the user typed.
 */
export function splitImageMarkers(content: string): { text: string; previews: string[] } {
  const previews: string[] = [];
  let cut = -1;
  for (const m of content.matchAll(IMAGE_MARK_RE)) {
    if (cut < 0) cut = m.index ?? 0;
    previews.push(`/api/chat-images/${m[1]}`);
  }
  return cut < 0 ? { text: content, previews } : { text: content.slice(0, cut), previews };
}

/** Put the `/compact` user bubble right before its card. */
function insertCompactCommand(msgs: ChatMsg[], card: ChatMsg) {
  const at = msgs.indexOf(card);
  msgs.splice(at < 0 ? msgs.length : at, 0, { role: "user", content: "/compact", ts: card.ts });
}

const WINDOW_OPEN = "<context_window>";
const WINDOW_CONTINUATION = "New context window started. Continue from session notes and search_session.";

function peelOneWindowLayer(input: string): string {
  let s = input.trimStart();
  if (!s.startsWith(WINDOW_OPEN)) return input;

  const cwClose = s.indexOf("</context_window>");
  if (cwClose < 0) return input;
  s = s.slice(cwClose + "</context_window>".length).trimStart();

  if (s.startsWith("<session_notes")) {
    const dsAt = s.indexOf("\n</session_notes>\n\n<dropped_span>\n");
    const contAt = s.indexOf(`\n</session_notes>\n\n${WINDOW_CONTINUATION}`);
    const nextCwAt = s.indexOf("\n</session_notes>\n\n<context_window>");
    let endAt = -1;
    if (dsAt >= 0) endAt = dsAt + "\n</session_notes>".length;
    else if (contAt >= 0) endAt = contAt + "\n</session_notes>".length;
    else if (nextCwAt >= 0) endAt = nextCwAt + "\n</session_notes>".length;
    else {
      const lineIdx = s.lastIndexOf("\n</session_notes>");
      if (lineIdx >= 0) endAt = lineIdx + "\n</session_notes>".length;
      else {
        const plain = s.indexOf("</session_notes>");
        if (plain >= 0) endAt = plain + "</session_notes>".length;
      }
    }
    if (endAt >= 0) s = s.slice(endAt).trimStart();
  }

  if (s.startsWith("<dropped_span>")) {
    const contMarker = `\n</dropped_span>\n\n${WINDOW_CONTINUATION}`;
    const contAt = s.lastIndexOf(contMarker);
    const nextCwMarker = "\n</dropped_span>\n\n<context_window>";
    const nextCwAt = s.indexOf(nextCwMarker);
    if (contAt >= 0) {
      s = s.slice(contAt + contMarker.length).trimStart();
    } else if (nextCwAt >= 0) {
      s = s.slice(nextCwAt + "\n</dropped_span>".length).trimStart();
    } else {
      const lineIdx = s.lastIndexOf("\n</dropped_span>");
      if (lineIdx >= 0) s = s.slice(lineIdx + "\n</dropped_span>".length).trimStart();
      else {
        const plain = s.indexOf("</dropped_span>");
        if (plain >= 0) s = s.slice(plain + "</dropped_span>".length).trimStart();
      }
    }
  }

  if (s.startsWith(WINDOW_CONTINUATION)) {
    s = s.slice(WINDOW_CONTINUATION.length).trimStart();
  }
  return s;
}

/**
 * Split a new-context-window preamble off a logged user row, the way
 * `agentica.memory.session_search.strip_window_preamble` does: peel every
 * layer, keep what follows. `notes` / `notesPath` come from the outermost
 * (latest) layer's `<session_notes>`. `null` when the row is not a preamble.
 */
export function splitWindowPreamble(content: string): { notes: string; notesPath: string; rest: string } | null {
  if (!content.trimStart().startsWith(WINDOW_OPEN)) return null;
  const m =
    content.match(/<session_notes(?:\s+path="([^"]*)")?>\n([\s\S]*?)\n<\/session_notes>(?=\n|$)/)
    || content.match(/<session_notes(?:\s+path="([^"]*)")?>([\s\S]*?)<\/session_notes>/);
  let rest = content.trimStart();
  while (rest.startsWith(WINDOW_OPEN)) {
    const peeled = peelOneWindowLayer(rest);
    if (peeled === rest) break;
    rest = peeled;
  }
  return { notes: (m?.[2] || "").trim(), notesPath: m?.[1] || "", rest };
}

export function isPureWindowPreamble(e: Record<string, unknown>): boolean {
  if (e.type !== "user" || e.automated === true || e.steer === true) return false;
  const pre = splitWindowPreamble(String(e.content || ""));
  return Boolean(pre && !pre.rest);
}

function uuidOf(e: Record<string, unknown>): string {
  return typeof e.uuid === "string" ? e.uuid : "";
}

function rawTsOf(e: Record<string, unknown>): number {
  const raw = e.timestamp;
  if (typeof raw === "string" && raw) {
    const ms = Date.parse(raw);
    if (Number.isFinite(ms)) return ms;
  }
  return 0;
}

function tsOf(e: Record<string, unknown>): number {
  return rawTsOf(e) || Date.now();
}

function assistantHasWork(e: Record<string, unknown>): boolean {
  const content = typeof e.content === "string" ? e.content : "";
  const reasoning = reasoningOf(e);
  return Boolean(content || reasoning || toolCallsOf(e).length);
}

function reasoningOf(e: Record<string, unknown>): string {
  for (const key of ["reasoning_content", "thinking"] as const) {
    const v = e[key];
    if (typeof v === "string" && v.trim()) return v;
  }
  return "";
}

function toolCallsOf(e: Record<string, unknown>): unknown[] {
  return Array.isArray(e.tool_calls) ? e.tool_calls : [];
}

function ensureAssistant(msgs: ChatMsg[], current: ChatMsg | null, ts: number): ChatMsg {
  if (current) return current;
  const created: ChatMsg = {
    role: "assistant",
    content: "",
    parts: [],
    steps: [],
    ts,
  };
  msgs.push(created);
  return created;
}

function applyAssistant(m: ChatMsg, e: Record<string, unknown>) {
  const ts = tsOf(e);
  const reasoning = reasoningOf(e);
  if (reasoning) pushThink(m, reasoning, ts);
  // A cancelled / failed / sealed-after-crash turn persists the model-facing
  // marker in ``content`` and the user-visible part in ``partial_content``.
  // A sealed row's ``content`` is only the marker, so older rows written
  // without ``partial_content`` still show nothing.
  const sealed = e.finish_reason === "interrupted";
  const ended = sealed || e.finish_reason === "cancelled" || e.finish_reason === "error";
  const hasPartial = ended && (sealed || typeof e.partial_content === "string");
  const content = hasPartial
    ? (typeof e.partial_content === "string" ? e.partial_content : "")
    : typeof e.content === "string" ? e.content : "";
  if (hasPartial) {
    if (e.finish_reason === "error") {
      if (typeof e.error === "string" && e.error) m.error = e.error;
    } else {
      m.aborted = true;
    }
  }
  if (content) {
    freezeOpenThink(m, ts);
    pushText(m, content);
  }
  for (const tc of toolCallsOf(e)) {
    const parsed = parseToolCall(tc);
    if (!parsed) continue;
    pushTool(m, parsed, ts);
  }
}

function parseToolCall(tc: unknown): { id: string; name: string; argsStr: string } | null {
  if (!tc || typeof tc !== "object") return null;
  const row = tc as Record<string, unknown>;
  const fn = row.function && typeof row.function === "object"
    ? row.function as Record<string, unknown>
    : {};
  const id = String(row.id || row.tool_call_id || "");
  const name = String(row.tool_name || fn.name || "");
  if (!id && !name) return null;
  const args = fn.arguments ?? row.arguments ?? row.tool_args ?? {};
  const argsStr = typeof args === "string" ? args : JSON.stringify(args ?? {});
  return { id, name, argsStr };
}

function applyToolResult(m: ChatMsg, e: Record<string, unknown>) {
  const callId = String(e.tool_call_id || "");
  const name = String(e.tool_name || "");
  const ts = tsOf(e);
  let result = typeof e.content === "string" ? e.content : String(e.content ?? "");
  const folded = splitFoldedSteer(result);
  result = folded.toolResult;
  if (e.is_error || e.tool_call_error) {
    result = result.startsWith("Error: ") ? result : `Error: ${result}`;
  }
  const parts = m.parts || (m.parts = []);
  let hit = findOpenTool(parts, callId);
  if (!hit) {
    pushTool(m, { id: callId, name, argsStr: "{}" }, ts);
    hit = findOpenTool(m.parts || [], callId);
  }
  if (!hit) return;
  hit.result = result;
  const diff = typeof e.diff === "string" && e.diff ? e.diff : undefined;
  if (diff) hit.diff = diff;
  if (hit.t0 != null) hit.ms = Math.max(0, ts - hit.t0);
  const steps = m.steps || [];
  const step = findOpenToolStep(steps, callId);
  if (step) {
    step.result = result;
    if (diff) step.diff = diff;
    if (step.t0 != null) step.ms = Math.max(0, ts - step.t0);
  }
  for (const steeredText of folded.steers) {
    pushSteer(m, steeredText, ts);
  }
}

function findOpenTool(parts: MsgPart[], callId: string): Extract<MsgPart, { kind: "tool" }> | undefined {
  if (callId) {
    const found = parts.find((p) => p.kind === "tool" && p.toolCallId === callId && p.result == null);
    if (found?.kind === "tool") return found;
  }
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    if (p.kind === "tool" && p.result == null) return p;
  }
}

function findOpenToolStep(steps: Array<Record<string, any>>, callId: string) {
  if (callId) {
    const found = steps.find((s) => s.type === "tool" && s.toolCallId === callId && s.result == null);
    if (found) return found;
  }
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].type === "tool" && steps[i].result == null) return steps[i];
  }
}

function freezeOpenThink(m: ChatMsg, end: number) {
  const parts = m.parts || [];
  const last = parts[parts.length - 1];
  if (last?.kind === "think" && last.ms == null && last.t0 != null) {
    last.ms = Math.max(0, end - last.t0);
  }
  const steps = m.steps || [];
  const step = steps[steps.length - 1];
  if (step?.type === "thinking" && step.ms == null && step.t0 != null) {
    step.ms = Math.max(0, end - step.t0);
  }
}

function pushThink(m: ChatMsg, text: string, t0: number) {
  freezeOpenThink(m, t0);
  (m.parts ||= []).push({ kind: "think", text, t0 });
  (m.steps ||= []).push({ type: "thinking", text, t0 });
}

function pushTool(m: ChatMsg, tc: { id: string; name: string; argsStr: string }, t0: number) {
  freezeOpenThink(m, t0);
  (m.parts ||= []).push({
    kind: "tool",
    name: tc.name,
    argsStr: tc.argsStr,
    t0,
    toolCallId: tc.id || undefined,
  });
  (m.steps ||= []).push({
    type: "tool",
    name: tc.name,
    argsStr: tc.argsStr,
    t0,
    toolCallId: tc.id || undefined,
  });
}

function pushText(m: ChatMsg, text: string) {
  m.content = (m.content || "") + text;
  const parts = (m.parts ||= []);
  const last = parts[parts.length - 1];
  if (last?.kind === "text") last.text += text;
  else parts.push({ kind: "text", text });
}

function pushSteer(m: ChatMsg, text: string, ts: number) {
  freezeOpenThink(m, ts);
  const parts = (m.parts ||= []);
  const last = parts[parts.length - 1];
  if (last?.kind === "steer" && last.text === text) return;
  parts.push({ kind: "steer", text, ts });
}

export type ClientSteerAnchor = {
  text: string;
  ts: number;
  afterToolCallId?: string;
  turnIdx: number;
};

const STEER_CACHE_KEY = "ag_steers_v1";
const STEER_CACHE_MAX_SESSIONS = 40;

function isValidSteerText(text: unknown): text is string {
  return typeof text === "string" && Boolean(text.trim()) && !text.trimStart().startsWith(WINDOW_OPEN);
}

function readSteerStore(): Record<string, ClientSteerAnchor[]> {
  if (typeof localStorage === "undefined") return {};
  try {
    const raw = localStorage.getItem(STEER_CACHE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (!parsed || typeof parsed !== "object") return {};
    const store = parsed as Record<string, ClientSteerAnchor[]>;
    let dirty = false;
    for (const [k, list] of Object.entries(store)) {
      if (!Array.isArray(list)) {
        delete store[k];
        dirty = true;
        continue;
      }
      const filtered = list.filter((item) => item && isValidSteerText(item.text));
      if (filtered.length !== list.length) {
        if (filtered.length) store[k] = filtered;
        else delete store[k];
        dirty = true;
      }
    }
    if (dirty) writeSteerStore(store);
    return store;
  } catch {
    return {};
  }
}

function writeSteerStore(store: Record<string, ClientSteerAnchor[]>) {
  if (typeof localStorage === "undefined") return;
  try {
    const keys = Object.keys(store);
    if (keys.length > STEER_CACHE_MAX_SESSIONS) {
      for (const k of keys.slice(0, keys.length - STEER_CACHE_MAX_SESSIONS)) delete store[k];
    }
    localStorage.setItem(STEER_CACHE_KEY, JSON.stringify(store));
  } catch { /* quota */ }
}

/** Harvest any in-memory `steer` parts from `msgs` before hydration replaces them. */
export function rememberSteersFromMsgs(sessionId: string, msgs: ChatMsg[], olderTurns = 0) {
  if (!sessionId || !msgs?.length) return;
  const found: ClientSteerAnchor[] = [];
  let turnIdx = olderTurns || 0;
  for (const m of msgs) {
    if (m.compact) continue;
    if (m.role === "user" && !m.steer && !m.automated) {
      turnIdx += 1;
      continue;
    }
    if (m.role === "assistant" && m.parts?.length) {
      let lastToolId: string | undefined;
      for (const p of m.parts) {
        if (p.kind === "tool" && p.toolCallId) lastToolId = p.toolCallId;
        else if (p.kind === "steer" && isValidSteerText(p.text)) {
          found.push({
            text: p.text,
            ts: p.ts || m.ts || Date.now(),
            afterToolCallId: lastToolId,
            turnIdx: Math.max(1, turnIdx),
          });
        }
      }
    }
  }
  if (!found.length) return;
  const store = readSteerStore();
  const existing = store[sessionId] || [];
  let changed = false;
  for (const item of found) {
    const dup = existing.some(
      (x) => x.text === item.text && (x.afterToolCallId === item.afterToolCallId || x.turnIdx === item.turnIdx),
    );
    if (!dup) {
      existing.push(item);
      changed = true;
    }
  }
  if (changed) {
    store[sessionId] = existing;
    writeSteerStore(store);
  }
}

/** Remove late steers that were popped back to the next-turn queue at run end. */
export function removeUndeliveredSteers(sessionId: string, aiMsg: ChatMsg, lateTexts: string[]) {
  if (!lateTexts.length) return;
  const parts = aiMsg.parts || [];
  for (let i = lateTexts.length - 1; i >= 0; i--) {
    const t = lateTexts[i];
    for (let j = parts.length - 1; j >= 0; j--) {
      const p = parts[j];
      if (p.kind === "steer" && p.text === t) {
        parts.splice(j, 1);
        break;
      }
    }
  }
  if (!sessionId) return;
  const store = readSteerStore();
  const list = store[sessionId];
  if (!list?.length) return;
  let changed = false;
  for (const t of lateTexts) {
    for (let j = list.length - 1; j >= 0; j--) {
      if (list[j].text === t) {
        list.splice(j, 1);
        changed = true;
        break;
      }
    }
  }
  if (changed) writeSteerStore(store);
}

/** Re-attach client-recorded steers when hydrating from a gateway that predates `steer: true` JSONL rows. */
export function mergeClientSteers(sessionId: string, msgs: ChatMsg[], olderTurns = 0): ChatMsg[] {
  if (!sessionId || !msgs.length) return msgs;
  const list = readSteerStore()[sessionId];
  if (!list?.length) return msgs;
  for (const item of list) {
    if (!isValidSteerText(item.text)) continue;
    // Already hydrated from JSONL?
    const alreadyPresent = msgs.some(
      (m) =>
        (m.role === "user" && m.steer && m.content === item.text)
        || (m.role === "assistant" && m.parts?.some((p) => p.kind === "steer" && p.text === item.text)),
    );
    if (alreadyPresent) continue;
    if (item.afterToolCallId) {
      let inserted = false;
      for (const m of msgs) {
        if (m.role !== "assistant" || !m.parts?.length) continue;
        const idx = m.parts.findIndex((p) => p.kind === "tool" && p.toolCallId === item.afterToolCallId);
        if (idx >= 0) {
          let insertAt = idx + 1;
          while (insertAt < m.parts.length && m.parts[insertAt].kind === "steer") insertAt++;
          m.parts.splice(insertAt, 0, { kind: "steer", text: item.text, ts: item.ts });
          inserted = true;
          break;
        }
      }
      if (inserted) continue;
    }
    if (item.turnIdx > olderTurns) {
      let turn = olderTurns;
      for (let i = 0; i < msgs.length; i++) {
        const m = msgs[i];
        if (m.role === "user" && !m.steer && !m.automated && !m.compact) {
          turn += 1;
          if (turn === item.turnIdx) {
            const next = msgs[i + 1];
            if (next && next.role === "assistant" && !next.compact) {
              (next.parts ||= []).unshift({ kind: "steer", text: item.text, ts: item.ts });
            }
            break;
          }
        }
      }
    }
  }
  return msgs;
}
