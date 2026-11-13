import type { ChatMsg, MsgPart } from "../store";

export type { MsgPart };

export type PartSeg =
  | { type: "work"; items: MsgPart[] }
  | { type: "single"; part: MsgPart };

export function partsOf(m: ChatMsg): MsgPart[] {
  if (m.parts && m.parts.length) return m.parts;
  const out: MsgPart[] = [];
  for (const st of m.steps || []) {
    if (st.type === "thinking") out.push({ kind: "think", text: st.text || "", t0: st.t0, ms: st.ms });
    else out.push({ kind: "tool", name: st.name || "", argsStr: st.argsStr || "", result: st.result, t0: st.t0, ms: st.ms, toolCallId: st.toolCallId });
  }
  if (m.content) out.push({ kind: "text", text: m.content });
  return out;
}

/**
 * Consecutive think/tool runs become one work group; text and steer break it.
 *
 * This is the whole structure: a reply is never merged into a work group, so an
 * expanded turn reads `G1 → 中间回复 → G2 → … → 最终回答` (what dsh calls one
 * node, two rows — reasoning joins the step run, a reply stands outside it).
 */
export function groupParts(parts: MsgPart[]): PartSeg[] {
  const segs: PartSeg[] = [];
  let run: MsgPart[] = [];
  const flush = () => {
    if (!run.length) return;
    segs.push({ type: "work", items: run });
    run = [];
  };
  for (const p of parts) {
    if (p.kind === "think" || p.kind === "tool") run.push(p);
    else {
      flush();
      segs.push({ type: "single", part: p });
    }
  }
  flush();
  return segs;
}

/** Index of the last `work` segment, or -1 when the turn has no think/tool work. */
export function lastWorkSegmentIndex(segs: PartSeg[]): number {
  for (let i = segs.length - 1; i >= 0; i--) {
    if (segs[i].type === "work") return i;
  }
  return -1;
}

/**
 * Whether `segs[index]` is folded into the turn process line (`已完成，用时 …`):
 * every `work` group plus any intermediate assistant `text` emitted before a later
 * `work` group (e.g. Claude / Gemini / GPT-5 commentary between tool calls while
 * internal CoT is hidden). User `steer` chips and trailing final-answer text stay out.
 *
 * Intermediate replies are still *replies* — they render as their own bubble, not
 * as a row inside a work group — but they belong to this fold: collapsing the turn
 * hides them with the steps, expanding it shows them in place. That is the one
 * level of folding, deliberately: nesting a reply inside the group's own fold hid
 * it behind two clicks.
 */
export function isProcessSegment(segs: PartSeg[], index: number): boolean {
  const seg = segs[index];
  if (!seg) return false;
  if (seg.type === "work") return true;
  const lastWork = lastWorkSegmentIndex(segs);
  if (lastWork < 0) return false;
  return seg.part.kind === "text" && index < lastWork;
}

/** Visible final-answer text (segments after the last `work` group), or full text when no work ran. */
export function finalAnswerOf(m: ChatMsg): string {
  const segs = groupParts(partsOf(m));
  const lastWork = lastWorkSegmentIndex(segs);
  if (lastWork < 0) return m.content || "";
  const trailing: string[] = [];
  for (let i = lastWork + 1; i < segs.length; i++) {
    const seg = segs[i];
    if (seg.type === "single" && seg.part.kind === "text" && seg.part.text.trim()) {
      trailing.push(seg.part.text);
    }
  }
  return trailing.join("\n\n");
}

/** All assistant text blocks joined cleanly with paragraph breaks (for Copy when expanded). */
export function allAssistantTextOf(m: ChatMsg): string {
  const blocks: string[] = [];
  for (const p of partsOf(m)) {
    if (p.kind === "text" && p.text.trim()) blocks.push(p.text);
  }
  return blocks.length ? blocks.join("\n\n") : (m.content || "");
}

/** Append `finalContent` at turn end when intermediate commentary left no trailing answer after tools. */
export function ensureTrailingFinalAnswer(m: ChatMsg, finalContent: string) {
  const clean = (finalContent || "").trim();
  if (!clean || finalAnswerOf(m).trim()) return;
  const all = allAssistantTextOf(m).trim();
  if (all && (all === clean || all.endsWith(clean))) return;
  appendText(m, finalContent);
}

export function ensureParts(m: ChatMsg): MsgPart[] {
  if (!m.parts) m.parts = [];
  if (!m.steps) m.steps = [];
  return m.parts;
}

type TimedItem = { kind?: string; type?: string; t0?: number; ms?: number; result?: string };

function isThinkItem(p: TimedItem): boolean {
  return p.kind === "think" || p.type === "thinking";
}

function isToolItem(p: TimedItem): boolean {
  return p.kind === "tool" || p.type === "tool";
}

function nextSliceT0(items: TimedItem[], i: number): number | undefined {
  for (let j = i + 1; j < items.length; j++) {
    const n = items[j];
    if ((isThinkItem(n) || isToolItem(n)) && n.t0 != null) return n.t0;
  }
}

function freezeSlice(p: TimedItem, items: TimedItem[], i: number, now?: number): boolean {
  if (p.t0 == null || p.ms != null) return false;
  const end = nextSliceT0(items, i) ?? now;
  p.ms = end != null ? Math.max(0, end - p.t0) : 0;
  return true;
}

function freezeOpenThinks(m: ChatMsg, now?: number): boolean {
  let changed = false;
  const freeze = (items: TimedItem[]) => {
    for (let i = 0; i < items.length; i++) {
      const p = items[i];
      if (!isThinkItem(p) || p.ms != null) continue;
      if (freezeSlice(p, items, i, now)) changed = true;
    }
  };
  freeze(m.parts || []);
  freeze(m.steps || []);
  return changed;
}

function freezeOpenTools(m: ChatMsg, now?: number): boolean {
  let changed = false;
  const freeze = (items: TimedItem[]) => {
    for (let i = 0; i < items.length; i++) {
      const p = items[i];
      if (!isToolItem(p) || p.result != null || p.ms != null) continue;
      if (freezeSlice(p, items, i, now)) changed = true;
    }
  };
  freeze(m.parts || []);
  freeze(m.steps || []);
  return changed;
}

/** Freeze the current think slice at `now`. Also closes leftover earlier slices
 *  (their end is the next think/tool `t0`, not wall-clock). */
export function finishThink(m: ChatMsg, now = Date.now()) {
  freezeOpenThinks(m, now);
}

/** Close leftover think/tool timers after the turn is no longer live.
 *  Omit `now` for historical cards so a trailing slice does not become 45h. */
export function settleWork(m: ChatMsg, now?: number): boolean {
  // The placeholder think opened at send time stays empty when the turn is
  // stopped (or fails) before the model says anything; once the turn is
  // over it is a "Thought" group with nothing in it.
  const dropEmpty = <T extends TimedItem & { text?: string }>(items: T[] | undefined) => {
    if (!items) return false;
    const kept = items.filter((p) => !(isThinkItem(p) && !(p.text || "").trim()));
    if (kept.length === items.length) return false;
    items.splice(0, items.length, ...kept);
    return true;
  };
  const droppedParts = dropEmpty(m.parts as any);
  const droppedSteps = dropEmpty(m.steps as any);
  const dropped = droppedParts || droppedSteps;
  const thinks = freezeOpenThinks(m, now);
  const tools = freezeOpenTools(m, now);
  return dropped || thinks || tools;
}

export function appendThink(m: ChatMsg, delta: string) {
  const parts = ensureParts(m);
  const last = parts[parts.length - 1];
  const t0 = Date.now();
  if (last?.kind === "think" && last.ms == null) {
    last.text += delta;
    const step = m.steps![m.steps!.length - 1];
    if (step && step.type === "thinking") step.text = (step.text || "") + delta;
    else m.steps!.push({ type: "thinking", text: delta, t0 });
    return;
  }
  finishThink(m, t0);
  parts.push({ kind: "think", text: delta, t0 });
  m.steps!.push({ type: "thinking", text: delta, t0 });
}

export function appendTool(m: ChatMsg, name: string, argsStr: string, toolCallId?: string) {
  const t0 = Date.now();
  finishThink(m, t0);
  const parts = ensureParts(m);
  if (toolCallId) {
    const existing = parts.find((p) => p.kind === "tool" && p.toolCallId === toolCallId);
    if (existing && existing.kind === "tool") {
      if (name) existing.name = name;
      if (argsStr && argsStr !== "{}") existing.argsStr = argsStr;
      const step = m.steps!.find((s) => s.type === "tool" && s.toolCallId === toolCallId);
      if (step) {
        if (name) step.name = name;
        if (argsStr && argsStr !== "{}") step.argsStr = argsStr;
      }
      return;
    }
  }
  parts.push({ kind: "tool", name, argsStr, t0, toolCallId });
  m.steps!.push({ type: "tool", name, argsStr, t0, toolCallId });
}

export function finishTool(m: ChatMsg, result: string, diff?: string, toolCallId?: string) {
  const now = Date.now();
  const parts = m.parts || [];
  let partHit: Extract<MsgPart, { kind: "tool" }> | undefined;
  if (toolCallId) {
    const found = parts.find((p) => p.kind === "tool" && p.toolCallId === toolCallId && p.result == null);
    if (found?.kind === "tool") partHit = found;
  }
  if (!partHit) {
    for (let i = parts.length - 1; i >= 0; i--) {
      const p = parts[i];
      if (p.kind === "tool" && p.result == null) {
        partHit = p;
        break;
      }
    }
  }
  if (partHit) {
    partHit.result = result;
    if (diff) partHit.diff = diff;
    if (partHit.t0) partHit.ms = now - partHit.t0;
  }
  const steps = m.steps || [];
  let stepHit = -1;
  if (toolCallId) {
    stepHit = steps.findIndex((s) => s.type === "tool" && s.toolCallId === toolCallId && s.result == null);
  }
  if (stepHit < 0) {
    for (let i = steps.length - 1; i >= 0; i--) {
      if (steps[i].type === "tool" && steps[i].result == null) {
        stepHit = i;
        break;
      }
    }
  }
  if (stepHit >= 0) {
    steps[stepHit].result = result;
    if (diff) steps[stepHit].diff = diff;
    if (steps[stepHit].t0) steps[stepHit].ms = now - steps[stepHit].t0;
  }
}

export function unfinishedToolCallId(m: ChatMsg, toolCallId?: string): string | undefined {
  if (toolCallId) return toolCallId;
  const parts = m.parts || [];
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    if (p.kind === "tool" && p.result == null) return p.toolCallId;
  }
  return undefined;
}

export function appendText(m: ChatMsg, delta: string) {
  finishThink(m);
  m.content = (m.content || "") + delta;
  const parts = ensureParts(m);
  const last = parts[parts.length - 1];
  if (last?.kind === "text") last.text += delta;
  else parts.push({ kind: "text", text: delta });
}

export function appendSteerPart(m: ChatMsg, text: string, ts = Date.now()) {
  finishThink(m);
  const parts = ensureParts(m);
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    if (p.kind === "think" && !p.text.trim()) continue;
    if (p.kind === "steer" && p.text === text) return;
    break;
  }
  parts.push({ kind: "steer", text, ts });
}

function sliceMs(p: MsgPart, now: number, live: boolean): number {
  if (p.kind === "think") {
    if (p.ms != null) return p.ms;
    if (!live) return 0;
    return p.t0 != null ? Math.max(0, now - p.t0) : 0;
  }
  if (p.kind === "tool") {
    if (p.ms != null) return p.ms;
    if (!live || p.result != null) return 0;
    return p.t0 != null ? Math.max(0, now - p.t0) : 0;
  }
  return 0;
}

/** Sum of think slices + tool slices. Unfinished steps use `now - t0` only while live. */
export function workSummary(
  items: MsgPart[],
  now = Date.now(),
  live = true,
): { steps: number; ms: number; running: boolean; t0?: number } {
  let steps = 0;
  let ms = 0;
  let running = false;
  let t0: number | undefined;
  for (const p of items) {
    if (p.kind === "tool") {
      steps += 1;
      if (live && p.result == null) running = true;
    } else if (live && p.kind === "think" && p.ms == null && p.t0 != null) {
      running = true;
    }
    if ((p.kind === "think" || p.kind === "tool") && p.t0 != null) {
      if (t0 === undefined || p.t0 < t0) t0 = p.t0;
    }
    ms += sliceMs(p, now, live);
  }
  return { steps, ms, running, t0 };
}
