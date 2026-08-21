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
    else out.push({ kind: "tool", name: st.name || "", argsStr: st.argsStr || "", result: st.result, t0: st.t0, ms: st.ms });
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

export function finishThink(m: ChatMsg, now = Date.now()) {
  const parts = m.parts || [];
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    if (p.kind === "think" && p.ms == null && p.t0 != null) {
      p.ms = Math.max(0, now - p.t0);
      break;
    }
  }
  const steps = m.steps || [];
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].type === "thinking" && steps[i].ms == null && steps[i].t0 != null) {
      steps[i].ms = Math.max(0, now - steps[i].t0);
      break;
    }
  }
}

export function appendThink(m: ChatMsg, delta: string) {
  const parts = ensureParts(m);
  const last = parts[parts.length - 1];
  const t0 = Date.now();
  if (last?.kind === "think") last.text += delta;
  else parts.push({ kind: "think", text: delta, t0 });
  const step = m.steps![m.steps!.length - 1];
  if (step && step.type === "thinking") step.text = (step.text || "") + delta;
  else m.steps!.push({ type: "thinking", text: delta, t0 });
}

export function appendTool(m: ChatMsg, name: string, argsStr: string) {
  const t0 = Date.now();
  finishThink(m, t0);
  ensureParts(m).push({ kind: "tool", name, argsStr, t0 });
  m.steps!.push({ type: "tool", name, argsStr, t0 });
}

export function finishTool(m: ChatMsg, result: string, diff?: string) {
  const now = Date.now();
  const parts = m.parts || [];
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i];
    if (p.kind === "tool" && p.result == null) {
      p.result = result;
      if (diff) p.diff = diff;
      if (p.t0) p.ms = now - p.t0;
      break;
    }
  }
  const steps = m.steps || [];
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].type === "tool" && steps[i].result == null) {
      steps[i].result = result;
      if (diff) steps[i].diff = diff;
      if (steps[i].t0) steps[i].ms = now - steps[i].t0;
      break;
    }
  }
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
  ensureParts(m).push({ kind: "steer", text, ts: Date.now() });
}

function sliceMs(p: MsgPart, now: number): number {
  if (p.kind === "think") {
    if (p.ms != null) return p.ms;
    return p.t0 != null ? Math.max(0, now - p.t0) : 0;
  }
  if (p.kind === "tool") {
    if (p.ms != null) return p.ms;
    return p.result == null && p.t0 != null ? Math.max(0, now - p.t0) : 0;
  }
  return 0;
}

/** Sum of think slices + tool slices. Unfinished steps use `now - t0`. */
export function workSummary(items: MsgPart[], now = Date.now()): { steps: number; ms: number; running: boolean; t0?: number } {
  let steps = 0;
  let ms = 0;
  let running = false;
  let t0: number | undefined;
  for (const p of items) {
    if (p.kind === "tool") {
      steps += 1;
      if (p.result == null) running = true;
    } else if (p.kind === "think" && p.ms == null && p.t0 != null) {
      running = true;
    }
    if ((p.kind === "think" || p.kind === "tool") && p.t0 != null) {
      if (t0 === undefined || p.t0 < t0) t0 = p.t0;
    }
    ms += sliceMs(p, now);
  }
  return { steps, ms, running, t0 };
}
