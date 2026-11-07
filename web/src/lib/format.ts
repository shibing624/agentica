export function uid() {
  return "s_" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

export function fmtN(n: number) {
  if (!n) return "0";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(Math.round(n));
}

/** ``-1`` = unlimited, same as CLI ``/goal --tokens -1`` and penguin-harness. */
export const UNLIMITED_TOKEN_BUDGET = -1;

/** Parse the goal chip: empty → unlimited; ``500k`` / ``2m`` / a positive int. Invalid → null. */
export function parseTokenBudget(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return UNLIMITED_TOKEN_BUDGET;
  const m = /^(\d+(?:\.\d+)?)([km])?$/i.exec(trimmed);
  if (!m) return null;
  const scale = m[2]?.toLowerCase() === "m" ? 1_000_000 : m[2]?.toLowerCase() === "k" ? 1_000 : 1;
  const value = Math.round(Number(m[1]) * scale);
  return value > 0 ? value : null;
}

export function fmtCost(cost: number) {
  if (cost > 0 && cost < 0.00005) return "<$0.0001";
  if (cost < 0.00995) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

export function fmtTime(sec: number) {
  if (!sec) return "0s";
  if (sec < 60) return `${sec.toFixed(1)}s`;
  return spanParts(sec).map(([u, n]) => `${n}${COMPACT[u]}`).join(" ");
}

export type SpanUnit = "y" | "mo" | "d" | "h" | "m" | "s";
const SPAN_STEPS: Array<[SpanUnit, number]> = [
  ["y", 365 * 86_400], ["mo", 30 * 86_400], ["d", 86_400], ["h", 3_600], ["m", 60], ["s", 1],
];
const COMPACT: Record<SpanUnit, string> = { y: "y", mo: "mo", d: "d", h: "h", m: "m", s: "s" };

/**
 * A span (seconds, >= 1) as its two largest units — `2h 32m`, `3d 4h`,
 * `1y 2mo` — dropping the smaller one when it is zero. Months are 30 days and
 * years 365, which is what a long-running session needs, not a calendar.
 */
export function spanParts(sec: number): Array<[SpanUnit, number]> {
  const total = Math.round(sec);
  const i = Math.max(0, SPAN_STEPS.findIndex(([, size]) => total >= size));
  const [unit, size] = SPAN_STEPS[i];
  const major = Math.floor(total / size);
  const parts: Array<[SpanUnit, number]> = [[unit, major]];
  const next = SPAN_STEPS[i + 1];
  if (next) {
    const minor = Math.floor((total - major * size) / next[1]);
    if (minor) parts.push([next[0], minor]);
  }
  return parts;
}

/** Wall / LLM duration, shared by the chat footer and the Trace round header. */
export function fmtDurationMs(ms: number) {
  if (ms <= 0) return "0ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const sec = ms / 1000;
  if (sec < 10) return `${sec.toFixed(2)}s`;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  return spanParts(sec).map(([u, n]) => `${n}${COMPACT[u]}`).join("");
}

/**
 * dsh `formatRunDuration`: whole-second elapsed time without leading zeros,
 * e.g. `7分31秒` / `7m 31s` / `1小时2分5秒`. Sub-second runs clamp to 1s.
 */
export function fmtRunDuration(ms: number, units: { h: string; m: string; s: string }): string {
  const total = Math.max(1, Math.floor(ms / 1000));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor(total / 60) % 60;
  const seconds = total % 60;
  let out = "";
  if (hours > 0) out += `${hours}${units.h}`;
  if (total >= 60) out += `${minutes}${units.m}`;
  out += `${seconds}${units.s}`;
  return out;
}

/** Edit one queued message in place without reordering the queue. */
export function updateQueuedMessageText<T extends { id: string; text: string }>(
  queue: readonly T[],
  id: string,
  text: string,
): T[] {
  return queue.map((item) => (item.id === id ? { ...item, text } : item));
}

/** A long span in words: `2小时32分` / `49.8秒` / `3天4小时`, units from the locale. */
export type SpanUnits = Record<SpanUnit, (n: number) => string> & {
  sec: (s: string) => string;
  ms: (ms: number) => string;
  join: string;
};
export function fmtSpan(ms: number, u: SpanUnits) {
  if (ms < 1000) return u.ms(Math.round(ms));
  const sec = ms / 1000;
  if (sec < 60) return u.sec(sec.toFixed(1));
  return spanParts(sec).map(([unit, n]) => u[unit](n)).join(u.join);
}

export function fmtTps(n: number) {
  if (!n) return "0 tok/s";
  return `${n.toFixed(1)} tok/s`;
}

export function shortenPath(p: string) {
  if (!p) return "";
  const parts = p.split("/").filter(Boolean);
  if (parts.length <= 3) return p;
  return "…/" + parts.slice(-2).join("/");
}

export type AgoLabels = {
  agoJustNow: string;
  agoMinutes: (n: number) => string;
  agoHours: (n: number) => string;
  agoDays: (n: number) => string;
  agoMonths: (n: number) => string;
  agoYears: (n: number) => string;
};

/** Last real user query (steer chips are mid-run). Falls back to lastTs then ts. */
export function lastQueryTs(sess: {
  ts: number;
  lastTs?: number;
  msgs?: Array<{ role: string; steer?: boolean; automated?: boolean; ts?: number }>;
}): number {
  const msgs = sess.msgs || [];
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i];
    if (m.role === "user" && !m.steer && !m.automated) return m.ts || sess.lastTs || sess.ts;
  }
  return sess.lastTs || sess.ts;
}

export type SessionSideStatus = "busy" | "unread" | "idle";

/** Sidebar slot to the right of the title: spinner, unread dot, or idle time. */
export function sessionSideStatus(opts: {
  id: string;
  curSess: string | null;
  unread?: boolean;
  running?: boolean;
}): SessionSideStatus {
  if (opts.running) return "busy";
  if (opts.unread && opts.id !== opts.curSess) return "unread";
  return "idle";
}

/**
 * Coarse idle label: just now / 5-min steps / hours / days / months / years.
 * Exact minutes like ``7m`` made the sidebar feel twitchy.
 */
export function agoStr(ts: number, labels: AgoLabels, now = Date.now()): string {
  const d = Math.max(0, now - ts);
  if (d < 5 * 60_000) return labels.agoJustNow;
  if (d < 60 * 60_000) {
    const mins = Math.floor(d / 60_000);
    return labels.agoMinutes(Math.max(5, Math.floor(mins / 5) * 5));
  }
  if (d < 24 * 60 * 60_000) return labels.agoHours(Math.floor(d / 3_600_000));
  if (d < 30 * 86_400_000) return labels.agoDays(Math.floor(d / 86_400_000));
  if (d < 365 * 86_400_000) return labels.agoMonths(Math.floor(d / (30 * 86_400_000)));
  return labels.agoYears(Math.floor(d / (365 * 86_400_000)));
}

export function fmtFileSize(n: number) {
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / 1024 / 1024).toFixed(1) + " MB";
}

export function formatDateTime(iso: string) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Local `YYYY-MM-DD HH:MM:SS` on blocking run errors (same shape as the CLI). */
export function formatErrorStamp(ms: number = Date.now()): string {
  const d = new Date(ms);
  if (Number.isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
