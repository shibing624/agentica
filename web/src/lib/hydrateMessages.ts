import type { ActiveRunInfo } from "../api";
import type { ChatMsg, MsgPart } from "../store";
import { appendSteerPart, appendThink, settleWork } from "./msgParts";
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
  let assistantTerminal = false;
  let turnStartTs = 0;
  let turnEndTs = 0;
  // `trigger` of the boundary whose preamble has not been seen yet ("" = not
  // recorded); null once that preamble is consumed.
  let pendingTrigger: string | null = null;
  // A cut between two requests of the open turn: a step of that turn, whose
  // notes arrive with the new window's preamble row.
  let inTurnCompact: Extract<MsgPart, { kind: "compact" }> | null = null;

  const closeAssistant = (atTail = false) => {
    if (!assistant) return;
    settleWork(assistant);
    if (assistant.durationMs == null && turnStartTs > 0 && turnEndTs > turnStartTs) {
      assistant.durationMs = turnEndTs - turnStartTs;
    }
    if (atTail && !assistantTerminal) {
      assistant.inFlight = true;
    }
    assistant = null;
    assistantTerminal = false;
  };

  for (const raw of events) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const type = String(e.type || "");
    const explicitTs = rawTsOf(e);
    if (type === "user") {
      const row = rowOf(e);
      if (row.automated) {
        closeAssistant();
        const job = e.cron_job as { id?: unknown; name?: unknown } | undefined;
        msgs.push({
          role: "user",
          automated: true,
          content: row.text,
          ts: tsOf(e),
          ...(uuidOf(e) ? { uuid: uuidOf(e) } : {}),
          ...(job && typeof job.id === "string" ? { cronJob: { id: job.id, name: String(job.name || "") } } : {}),
        });
        continue;
      }
      let content = row.text;
      if (row.window && inTurnCompact) {
        Object.assign(inTurnCompact.card, { notes: row.window.notes, notesPath: row.window.notes_path || inTurnCompact.card.notesPath || "" });
        inTurnCompact = null;
        pendingTrigger = null;
        // The prose behind it is the turn's own question, replayed.
        continue;
      }
      if (row.window) {
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
        Object.assign(card.compact!, { notes: row.window.notes, notesPath: row.window.notes_path || card.compact!.notesPath || "" });
        // Logs older than the boundary's `trigger`: an idle /compact folds no
        // question behind its preamble; an auto cut keeps the pending one.
        const bare = !row.text.trim();
        if (pendingTrigger === "" && bare) insertCompactCommand(msgs, card);
        else if (pendingTrigger === "") card.compact!.auto = true;
        pendingTrigger = null;
        if (bare) continue;
        dropShadowedQuestion(msgs, card, content);
      }
      if (row.steer) {
        const clean = content;
        if (!clean) continue;
        if (!turnStartTs && explicitTs) turnStartTs = explicitTs;
        if (explicitTs > turnEndTs) turnEndTs = explicitTs;
        assistant = ensureAssistant(msgs, assistant, tsOf(e));
        assistantTerminal = false;
        pushSteer(assistant, clean, tsOf(e));
        continue;
      }
      closeAssistant();
      const previews = row.images.map(imageUrl);
      const skill = parseSkillTurn(content);
      turnStartTs = explicitTs;
      turnEndTs = explicitTs;
      msgs.push({
        role: "user",
        content,
        ts: tsOf(e),
        ...(uuidOf(e) ? { uuid: uuidOf(e) } : {}),
        ...(skill ? { skill } : {}),
        ...(row.turn != null ? { turn: row.turn } : {}),
        ...(previews.length ? { previews } : {}),
      });
      continue;
    }
    if (type === "assistant") {
      if (!assistantHasWork(e)) continue;
      if (!turnStartTs && explicitTs) turnStartTs = explicitTs;
      if (explicitTs > turnEndTs) turnEndTs = explicitTs;
      assistant = ensureAssistant(msgs, assistant, tsOf(e));
      applyAssistant(assistant, e);
      const hasTools = toolCallsOf(e).length > 0;
      const fr = typeof e.finish_reason === "string" ? e.finish_reason : "";
      assistantTerminal = !hasTools && fr !== "tool_calls" && fr !== "tool_use";
      // The reply ends at its last row: fork / feedback address that one.
      if (uuidOf(e)) assistant.uuid = uuidOf(e);
      continue;
    }
    if (type === "tool" || type === "tool_audit") {
      if (!turnStartTs && explicitTs) turnStartTs = explicitTs;
      if (explicitTs > turnEndTs) turnEndTs = explicitTs;
      assistant = ensureAssistant(msgs, assistant, tsOf(e));
      assistantTerminal = false;
      applyToolResult(assistant, e);
      if (uuidOf(e)) assistant.uuid = uuidOf(e);
      continue;
    }
    if (type === "compact_boundary") {
      const trigger = typeof e.trigger === "string" ? e.trigger : "";
      const notesPath = typeof e.notes_path === "string" ? e.notes_path : "";
      if (assistant && !assistantTerminal && trigger !== "manual") {
        inTurnCompact = { kind: "compact", card: { auto: true, ...(notesPath ? { notesPath } : {}) } };
        (assistant.parts ||= []).push(inTurnCompact);
        pendingTrigger = trigger;
        continue;
      }
      closeAssistant();
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
  closeAssistant(true);
  return msgs;
}

/** What the gateway decided a row is (`agentica.session.rows.row_payload`).
 *  The client draws from it and never parses `content` for markers. */
export type RowView = {
  automated: boolean;
  steer: boolean;
  /** The prose, with the window preamble, steer prefix and image markers cut. */
  text: string;
  /** Stored picture names sent with the turn. */
  images: string[];
  window: { notes: string; notes_path: string } | null;
  /** 1-based turn number from the start of the log, on rows a person's turn starts at. */
  turn?: number;
};

/** A tool result row: the output, whether it failed, and steers folded behind it. */
export type ToolRowView = { error: boolean; text: string; steers: string[] };

export const imageUrl = (name: string) => `/api/chat-images/${name}`;

function serverRow<T>(e: Record<string, unknown>): T {
  const row = e.row as T | undefined;
  if (!row || typeof (row as { text?: unknown }).text !== "string") {
    throw new Error(`${String(e.type)} row ${String(e.uuid || "?")} has no server-side \`row\`; the gateway and web bundle are out of step`);
  }
  return row;
}

const rowOf = (e: Record<string, unknown>) => serverRow<RowView>(e);
const toolRowOf = (e: Record<string, unknown>) => serverRow<ToolRowView>(e);

/** Put the `/compact` user bubble right before its card. */
function insertCompactCommand(msgs: ChatMsg[], card: ChatMsg) {
  const at = msgs.indexOf(card);
  msgs.splice(at < 0 ? msgs.length : at, 0, { role: "user", content: "/compact", ts: card.ts, command: true });
}

/**
 * An auto compact that fires as a turn starts: the question was logged, then
 * the new window re-logs it behind its preamble, which is the copy the turn
 * runs on. The earlier bubble is the same question shown twice — drop it,
 * the way `drop_shadowed_by_boundary` drops it from search. Only the bubble
 * right before the card counts: a question that already got an answer is a
 * real repeat.
 */
function dropShadowedQuestion(msgs: ChatMsg[], card: ChatMsg, question: string) {
  const at = msgs.indexOf(card);
  const prev = at > 0 ? msgs[at - 1] : undefined;
  if (prev?.role === "user" && !prev.automated && prev.content.trim() === question.trim()) {
    msgs.splice(at - 1, 1);
  }
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
  const view = toolRowOf(e);
  let result = view.text;
  if (view.error) {
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
  for (const steeredText of view.steers) {
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
  /** Which steer this is, so a refused one can be withdrawn by identity. */
  steerId?: string;
};

const STEER_CACHE_KEY = "ag_steers_v1";
const STEER_CACHE_MAX_SESSIONS = 40;

function isValidSteerText(text: unknown): text is string {
  return typeof text === "string" && Boolean(text.trim());
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
            ...(p.steerId ? { steerId: p.steerId } : {}),
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
      (x) =>
        // Same identity wins: two different steers can carry identical text
        // (the user deliberately repeating a line), so text alone merges them
        // into one anchor and the reloaded chip loses its id.
        (item.steerId && x.steerId === item.steerId) ||
        (x.text === item.text && (x.afterToolCallId === item.afterToolCallId || x.turnIdx === item.turnIdx)),
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

/**
 * Withdraw the client record of one steer the server refused.
 *
 * The composer draws its chip before the request, and that chip is written into
 * the steer cache (which re-attaches steers a gateway did not log). A refusal
 * removes the chip from the transcript, so the cached copy has to go too —
 * otherwise a reload would bring back a steer that never happened.
 */
export function forgetClientSteer(sessionId: string, steerId: string) {
  if (!sessionId || !steerId) return;
  const store = readSteerStore();
  const list = store[sessionId];
  if (!list?.length) return;
  const at = list.findIndex((item) => item.steerId === steerId);
  if (at < 0) return;
  list.splice(at, 1);
  if (list.length) writeSteerStore(store);
  else {
    delete store[sessionId];
    writeSteerStore(store);
  }
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
          m.parts.splice(insertAt, 0, {
            kind: "steer", text: item.text, ts: item.ts,
            ...(item.steerId ? { steerId: item.steerId } : {}),
          });
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
              (next.parts ||= []).unshift({
                kind: "steer", text: item.text, ts: item.ts,
                ...(item.steerId ? { steerId: item.steerId } : {}),
              });
            }
            break;
          }
        }
      }
    }
  }
  return msgs;
}

export type PendingSteer = { text: string; ts: number; afterToolCallId?: string; steerId?: string };

function isDisconnectedMessageErr(err: string | undefined): boolean {
  if (!err) return false;
  const low = err.toLowerCase();
  return low.includes("network") || low.includes("failed to fetch") || low.includes("load failed");
}

function extractSteersFromAssistant(m: ChatMsg): PendingSteer[] {
  const out: PendingSteer[] = [];
  let lastToolId: string | undefined;
  for (const p of m.parts || []) {
    if (p.kind === "tool" && p.toolCallId) lastToolId = p.toolCallId;
    else if (p.kind === "steer" && isValidSteerText(p.text)) {
      out.push({
        text: p.text,
        ts: p.ts || m.ts || Date.now(),
        afterToolCallId: lastToolId,
        ...(p.steerId ? { steerId: p.steerId } : {}),
      });
    }
  }
  return out;
}

/**
 * Reconcile a session's hydrated / in-memory ``msgs`` with an active server
 * ``run`` before opening the SSE stream.
 *
 * A mid-turn hydration from ``SessionLog`` closes every flushed think/tool
 * slice (``settleWork``), so checking only for open slices misread the partial
 * turn as a finished reply and pushed a second ``assistant`` bubble below it —
 * exposing ``MessageFilesCard`` and the settled footer mid-run.
 */
export function reconcileLiveTurnMsgs(
  sessionId: string,
  msgs: ChatMsg[],
  run: ActiveRunInfo,
  olderTurns = 0,
): { aiMsg: ChatMsg; afterTool: string; pendingSteers: PendingSteer[] } {
  const startedAt = typeof run.started_at_ms === "number" && run.started_at_ms > 0
    ? run.started_at_ms
    : 0;

  if (run.automated_message) {
    let lastUser: ChatMsg | undefined;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user" && !msgs[i].steer) {
        lastUser = msgs[i];
        break;
      }
    }
    if (!lastUser || !lastUser.automated || lastUser.content.trim() !== run.automated_message.trim()) {
      msgs.push({
        role: "user",
        automated: true,
        content: run.automated_message,
        ts: startedAt || Date.now(),
        ...(run.cron_job ? { cronJob: run.cron_job } : {}),
      });
    }
  } else if (run.user_message) {
    const promptText = run.user_message.trim();
    let lastUser: ChatMsg | undefined;
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user" && !msgs[i].automated && !msgs[i].steer) {
        lastUser = msgs[i];
        break;
      }
    }
    if (promptText && (!lastUser || lastUser.content.trim() !== promptText)) {
      msgs.push({
        role: "user",
        content: run.user_message,
        ts: startedAt || Date.now(),
        files: [],
        previews: (run.user_images || []).map(imageUrl),
      });
    }
  }

  const last = msgs[msgs.length - 1];
  const parts = last?.parts || [];
  const unfinished = parts.some(
    (p) => (p.kind === "think" && p.ms == null) || (p.kind === "tool" && p.result == null),
  );
  const blank = last?.role === "assistant" && !last.content && !parts.length && !last.aborted && last.tokIn == null;
  const sameRunTime = startedAt > 0 && last?.ts != null && last.ts >= startedAt - 2000;
  const reuseLast = Boolean(
    last
    && last.role === "assistant"
    && !last.compact
    && (last.inFlight || unfinished || blank || isDisconnectedMessageErr(last.error) || sameRunTime),
  );

  if (!reuseLast) {
    const aiMsg: ChatMsg = {
      role: "assistant",
      content: "",
      steps: [],
      parts: [],
      ts: startedAt || Date.now(),
      durationSec: 0,
      inFlight: true,
    };
    appendThink(aiMsg, "");
    msgs.push(aiMsg);
    return { aiMsg, afterTool: "", pendingSteers: [] };
  }

  rememberSteersFromMsgs(sessionId, msgs, olderTurns);
  const steers = extractSteersFromAssistant(last);
  delete last.uuid;
  delete last.error;
  delete last.errorTs;
  last.aborted = false;
  delete last.durationMs;
  last.durationSec = 0;
  delete last.tokIn;
  delete last.tokOut;
  delete last.cacheRead;
  delete last.cacheWrite;
  delete last.cacheHitPercent;
  delete last.requests;
  delete last.costUsd;
  last.inFlight = true;
  if (startedAt > 0) {
    last.ts = last.ts ? Math.min(last.ts, startedAt) : startedAt;
  }

  // Ring buffer trimmed (>1MB / >4000 events): keep the hydrated prefix intact
  // and ask the server only for events after the last flushed tool call.
  if ((run.first_seq ?? 1) > 1 && parts.length > 0) {
    let lastSettledToolId = "";
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      if (p.kind === "tool" && p.result != null && p.toolCallId) {
        lastSettledToolId = p.toolCallId;
        break;
      }
    }
    appendThink(last, "");
    return { aiMsg: last, afterTool: lastSettledToolId, pendingSteers: [] };
  }

  last.content = "";
  last.parts = [];
  last.steps = [];
  for (const st of steers) {
    if (!st.afterToolCallId) appendSteerPart(last, st.text, st.ts, [], st.steerId || "");
  }
  appendThink(last, "");
  return {
    aiMsg: last,
    afterTool: "",
    pendingSteers: steers.filter((st) => Boolean(st.afterToolCallId)),
  };
}
