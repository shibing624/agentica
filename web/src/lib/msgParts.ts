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

/** Consecutive think/tool runs become one work group; text and steer break it. */
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
  const thinks = freezeOpenThinks(m, now);
  const tools = freezeOpenTools(m, now);
  return thinks || tools;
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
  ensureParts(m).push({ kind: "tool", name, argsStr, t0, toolCallId });
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

export function appendSteerPart(m: ChatMsg, text: string) {
  finishThink(m);
  ensureParts(m).push({ kind: "steer", text, ts: Date.now() });
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
