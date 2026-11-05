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
  // `trigger` of the boundary whose preamble has not been seen yet ("" = not
  // recorded); null once that preamble is consumed.
  let pendingTrigger: string | null = null;

  const closeAssistant = () => {
    if (!assistant) return;
    settleWork(assistant);
    assistant = null;
  };

  for (const raw of events) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const type = String(e.type || "");
    if (type === "user") {
      closeAssistant();
      let content = String(e.content || "");
      const pre = splitWindowPreamble(content);
      if (pre) {
        // A new window's preamble is chrome, not something the user said: it
        // becomes the compact card (with the notes it started from), and only
        // a question folded in behind it stays a user bubble.
        const last = msgs[msgs.length - 1];
        let card = last?.compact && last.compact.notes == null ? last : null;
        if (!card) {
          card = { role: "assistant", content: "", ts: tsOf(e), compact: {} };
          msgs.push(card);
        }
        Object.assign(card.compact!, { notes: pre.notes, notesPath: pre.notesPath });
        // Logs older than the boundary's `trigger`: an idle /compact folds no
        // question behind its preamble; an auto cut keeps the pending one.
        if (pendingTrigger === "" && !pre.rest) insertCompactCommand(msgs, card);
        else if (pendingTrigger === "") card.compact!.auto = true;
        pendingTrigger = null;
        if (!pre.rest) continue;
        content = pre.rest;
      }
      const skill = parseSkillTurn(content);
      msgs.push({
        role: "user",
        content,
        ts: tsOf(e),
        ...(uuidOf(e) ? { uuid: uuidOf(e) } : {}),
        ...(skill ? { skill } : {}),
      });
      continue;
    }
    if (type === "assistant") {
      if (!assistantHasWork(e)) continue;
      assistant = ensureAssistant(msgs, assistant, tsOf(e));
      applyAssistant(assistant, e);
      // The reply ends at its last row: fork / feedback address that one.
      if (uuidOf(e)) assistant.uuid = uuidOf(e);
      continue;
    }
    if (type === "tool" || type === "tool_audit") {
      assistant = ensureAssistant(msgs, assistant, tsOf(e));
      applyToolResult(assistant, e);
      if (uuidOf(e)) assistant.uuid = uuidOf(e);
      continue;
    }
    if (type === "compact_boundary") {
      closeAssistant();
      const trigger = typeof e.trigger === "string" ? e.trigger : "";
      const card: ChatMsg = { role: "assistant", content: "", ts: tsOf(e), compact: trigger === "auto" ? { auto: true } : {} };
      msgs.push(card);
      // The user's `/compact` is not a logged row; the boundary says it was one.
      if (trigger === "manual") insertCompactCommand(msgs, card);
      pendingTrigger = trigger;
    }
  }
  closeAssistant();
  return msgs;
}

/** Put the `/compact` user bubble right before its card. */
function insertCompactCommand(msgs: ChatMsg[], card: ChatMsg) {
  const at = msgs.indexOf(card);
  msgs.splice(at < 0 ? msgs.length : at, 0, { role: "user", content: "/compact", ts: card.ts });
}

const WINDOW_OPEN = "<context_window>";
const WINDOW_CLOSERS = ["</context_window>", "</session_notes>", "</dropped_span>"];
const WINDOW_CONTINUATION = "New context window started. Continue from session notes and search_session.";

/**
 * Split a new-context-window preamble off a logged user row, the way
 * `agentica.memory.session_search.strip_window_preamble` does: peel every
 * layer, keep what follows. `notes` / `notesPath` come from the outermost
 * (latest) layer's `<session_notes>`. `null` when the row is not a preamble.
 */
export function splitWindowPreamble(content: string): { notes: string; notesPath: string; rest: string } | null {
  if (!content.trimStart().startsWith(WINDOW_OPEN)) return null;
  const m = content.match(/<session_notes(?:\s+path="([^"]*)")?>([\s\S]*?)<\/session_notes>/);
  let rest = content;
  while (rest.trimStart().startsWith(WINDOW_OPEN)) {
    let peeled = rest;
    for (const closer of WINDOW_CLOSERS) {
      const i = peeled.indexOf(closer);
      if (i >= 0) peeled = peeled.slice(i + closer.length);
    }
    const at = peeled.indexOf(WINDOW_CONTINUATION);
    if (at >= 0) peeled = peeled.slice(at + WINDOW_CONTINUATION.length);
    peeled = peeled.trimStart();
    if (peeled === rest) break;
    rest = peeled;
  }
  return { notes: (m?.[2] || "").trim(), notesPath: m?.[1] || "", rest };
}

function uuidOf(e: Record<string, unknown>): string {
  return typeof e.uuid === "string" ? e.uuid : "";
}

function tsOf(e: Record<string, unknown>): number {
  const raw = e.timestamp;
  if (typeof raw === "string" && raw) {
    const ms = Date.parse(raw);
    if (Number.isFinite(ms)) return ms;
  }
  return Date.now();
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
  const content = typeof e.content === "string" ? e.content : "";
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
  if (hit.t0 != null) hit.ms = Math.max(0, ts - hit.t0);
  const steps = m.steps || [];
  const step = findOpenToolStep(steps, callId);
  if (step) {
    step.result = result;
    if (step.t0 != null) step.ms = Math.max(0, ts - step.t0);
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
