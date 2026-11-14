import { finalAnswerOf } from "./msgParts";
import type { ChatMsg } from "../store";

/** Show the rail once there are at least two user turns to jump between. */
export const CHAT_NAV_MIN_TICKS = 2;

/**
 * Preview budgets, sized to the popover's clamps (one prompt line, three
 * answer lines) and mirrored from dsh `turn-navigation.ts`: a rail entry that
 * cannot be read in one glance is not a preview, and copying whole turns into
 * navigation state would grow with every token of a live turn.
 */
export const PROMPT_PREVIEW_LIMIT = 50;
export const ANSWER_PREVIEW_LIMIT = 120;

export type ChatNavTick = {
  idx: number;
  query: string;
  /** The turn's final answer, already bounded for the popover. */
  answer: string;
};

/**
 * A preview is one glance, not a document: Markdown that only reads well when
 * rendered (a table's pipe grid, a fence, a `---` rule) turns into noise at 50
 * characters. Structural lines are dropped and inline markers stripped, so the
 * popover shows the words the turn was about rather than its formatting.
 *
 * This is a render-time downgrade of a known format, not a reading of what the
 * text means: no line is classified by its wording, only by the syntax that is
 * already unambiguous in Markdown.
 */
function plainText(text: string): string {
  const kept: string[] = [];
  let inFence = false;
  for (const raw of text.split("\n")) {
    let line = raw.trim();
    if (/^(```|~~~)/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    // A table row or its `|---|` rule: every cell boundary is punctuation, so
    // the line survives only as a row of pipes.
    if (line.startsWith("|") || /^\|?[\s:-]*-{3,}[\s:|-]*$/.test(line)) continue;
    // A horizontal rule, or a heading marker with nothing after it.
    if (/^([-*_])\1{2,}$/.test(line)) continue;
    // Block markers are per line: joined into one paragraph, `^` would no
    // longer be the start of the heading and `## 1.` would leak into the text.
    line = line.replace(/^#{1,6}\s+/, "").replace(/^\s*>\s?/, "");
    kept.push(line);
  }
  return kept.join(" ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`~]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Collapse whitespace and cap at `n`, marking the cut with an ellipsis so a
 * clipped preview never reads as the whole turn (dsh `preview()`).
 */
export function previewText(text: string, n = PROMPT_PREVIEW_LIMIT): string {
  const normalized = plainText(text);
  if (normalized.length > n - 1) return normalized.slice(0, n - 1).trimEnd() + "…";
  return normalized;
}

/** One tick per real user query (steer chips are mid-run interrupts, and
 *  automated reports are not something the user asked). */
export function buildChatNavTicks(msgs: ChatMsg[]): ChatNavTick[] {
  const ticks: ChatNavTick[] = [];
  for (let i = 0; i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role !== "user" || m.steer || m.automated) continue;
    ticks.push({
      idx: i,
      query: previewText(m.content || ""),
      answer: "",
    });
  }
  // The answer of a turn is the last thing the model said in it: scan back
  // from the next question so a turn that ended with tool output still shows
  // the reply the reader remembers.
  for (let t = 0; t < ticks.length; t++) {
    const start = ticks[t].idx + 1;
    const end = t + 1 < ticks.length ? ticks[t + 1].idx : msgs.length;
    for (let i = end - 1; i >= start; i--) {
      const m = msgs[i];
      if (m.role !== "assistant" || m.compact) continue;
      const text = finalAnswerOf(m) || m.content || "";
      if (!text.trim()) continue;
      ticks[t].answer = previewText(text, ANSWER_PREVIEW_LIMIT);
      break;
    }
  }
  return ticks;
}

export function offsetIn(el: HTMLElement, container: HTMLElement): number {
  return el.getBoundingClientRect().top - container.getBoundingClientRect().top + container.scrollTop;
}

export function scrollChatToMessage(area: HTMLElement, idx: number): boolean {
  const el = area.querySelector("#msg-" + idx) ?? document.getElementById("msg-" + idx);
  if (!(el instanceof HTMLElement)) return false;
  try {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch {
    area.scrollTop = Math.max(0, offsetIn(el, area) - 12);
  }
  flashUserTurn(el);
  return true;
}

export function flashUserTurn(el: HTMLElement) {
  el.classList.remove("chat-nav-flash");
  void el.offsetWidth;
  el.classList.add("chat-nav-flash");
  const done = () => el.classList.remove("chat-nav-flash");
  el.addEventListener("animationend", done, { once: true });
}

export function activeTickIdx(ticks: ChatNavTick[], area: HTMLElement): number | null {
  if (!ticks.length) return null;
  let active = ticks[0].idx;
  for (const t of ticks) {
    const el = area.querySelector("#msg-" + t.idx) ?? document.getElementById("msg-" + t.idx);
    if (!(el instanceof HTMLElement)) continue;
    if (offsetIn(el, area) <= area.scrollTop + 64) active = t.idx;
    else break;
  }
  return active;
}

export function ensureRailTickVisible(list: HTMLElement, idx: number) {
  const n = list.querySelector(`[data-idx="${idx}"]`);
  if (!(n instanceof HTMLElement)) return;
  if (n.offsetTop < list.scrollTop) list.scrollTop = n.offsetTop;
  else if (n.offsetTop + n.offsetHeight > list.scrollTop + list.clientHeight) {
    list.scrollTop = n.offsetTop + n.offsetHeight - list.clientHeight + 1;
  }
}
