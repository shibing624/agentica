import { cleanSteerContent, isSteerUserEvent } from "./hydrateMessages";

export const COLLAPSED_SESSION_LIMIT = 5;

export type SessionContentHit = {
  turn: number;
  role: "user" | "assistant";
  snippet: string;
};

export type SessionRow = {
  id: string;
  session: {
    title: string;
    preview?: string;
    running?: boolean;
    pinned?: boolean;
    source?: string;
    origin?: string;
  };
};

const IM_SOURCES = new Set([
  "wechat",
  "wecom",
  "feishu",
  "telegram",
  "dingtalk",
  "qq",
  "discord",
  "slack",
]);

const DEFAULT_SOURCE_LABELS: Record<string, string> = {
  cli: "CLI",
  wechat: "微信",
  wecom: "企微",
  feishu: "飞书",
  telegram: "TG",
  dingtalk: "钉钉",
  qq: "QQ",
  discord: "Discord",
  slack: "Slack",
};

/** Resolve the sidebar source badge for a session (CLI or IM channel); null for plain Web/Desktop chats. */
export function sessionSourceBadge(
  source?: string,
  labels: Record<string, string> = DEFAULT_SOURCE_LABELS,
): { key: string; label: string; isIm: boolean } | null {
  const key = (source || "").trim().toLowerCase();
  if (!key || key === "gateway" || key === "web" || key === "legacy") return null;
  if (key === "cli") {
    return { key, label: labels.cli || "CLI", isIm: false };
  }
  if (IM_SOURCES.has(key)) {
    return { key, label: labels[key] || DEFAULT_SOURCE_LABELS[key] || key.toUpperCase(), isIm: true };
  }
  return null;
}

/**
 * Hover text for the badge. `source` is where the latest turn came in,
 * `origin` where the session was started; the second half only appears when
 * they differ (a WeChat conversation picked up in the CLI).
 */
export function sessionSourceTitle(
  source: string | undefined,
  origin: string | undefined,
  labels: Record<string, string>,
  S: { latestVia: string; startedVia: string },
): string {
  const name = (key: string) => labels[key] || DEFAULT_SOURCE_LABELS[key] || key;
  const latest = (source || "").trim().toLowerCase();
  const first = (origin || "").trim().toLowerCase();
  const parts = latest ? [S.latestVia.replace("{0}", name(latest))] : [];
  if (first && first !== latest) parts.push(S.startedVia.replace("{0}", name(first)));
  return parts.join(" · ");
}

export function sessionMatches(
  row: SessionRow,
  projectName: string,
  query: string,
  contentHits?: SessionContentHit[],
) {
  const q = query.trim().toLocaleLowerCase();
  if (!q) return true;
  if (contentHits && contentHits.length > 0) return true;
  const badge = sessionSourceBadge(row.session.source);
  return [
    row.session.title,
    row.session.preview || "",
    projectName,
    row.session.source || "",
    badge ? badge.label : "",
  ].some((value) => value.toLocaleLowerCase().includes(q));
}

function snippetAround(text: string, needle: string, before = 24, after = 56): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const idx = flat.toLocaleLowerCase().indexOf(needle);
  if (idx < 0) return flat.length > 80 ? flat.slice(0, 80) + "…" : flat;
  const start = Math.max(0, idx - before);
  const end = Math.min(flat.length, idx + needle.length + after);
  return (start > 0 ? "…" : "") + flat.slice(start, end) + (end < flat.length ? "…" : "");
}

function stripWindowPreamble(content: string): string {
  if (!content.startsWith("<context_window>")) return content;
  for (const m of ["\n</dropped_span>\n", "\n</session_notes>\n", "\n</context_window>\n"]) {
    const at = content.indexOf(m);
    if (at >= 0) {
      let rest = content.slice(at + m.length).replace(/^\n+/, "");
      if (rest.startsWith("New context window started")) {
        const nl = rest.indexOf("\n");
        rest = nl >= 0 ? rest.slice(nl + 1).replace(/^\n+/, "") : "";
      }
      return rest;
    }
  }
  return "";
}

/** Search user queries and assistant answers inside a hydrated `msgs` list. */
export function searchMsgsConversation(
  msgs: Array<{ role: string; content: string; steer?: boolean; automated?: boolean; compact?: unknown; parts?: Array<{ kind: string; text?: string }> }>,
  olderTurns: number,
  query: string,
  maxHits = 12,
): SessionContentHit[] {
  const needle = query.trim().replace(/\s+/g, " ").toLocaleLowerCase();
  if (!needle) return [];
  const hits: SessionContentHit[] = [];
  let turn = olderTurns || 0;
  const seen = new Set<string>();
  for (const m of msgs) {
    if (m.compact) continue;
    if (m.role === "user") {
      if (m.automated) continue;
      if (!m.steer) turn += 1;
      const clean = (m.content || "").trim();
      if (!clean) continue;
      const key = `${Math.max(1, turn)}:user`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "user", snippet: snippetAround(clean, needle) });
      }
    } else if (m.role === "assistant") {
      for (const p of m.parts || []) {
        if (p.kind !== "steer" || !p.text) continue;
        const cleanSteer = p.text.trim();
        const uKey = `${Math.max(1, turn)}:user`;
        if (cleanSteer && !seen.has(uKey) && cleanSteer.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
          seen.add(uKey);
          hits.push({ turn: Math.max(1, turn), role: "user", snippet: snippetAround(cleanSteer, needle) });
        }
      }
      const clean = (m.content || "").trim();
      if (!clean) continue;
      const key = `${Math.max(1, turn)}:assistant`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "assistant", snippet: snippetAround(clean, needle) });
      }
    }
    if (hits.length >= maxHits) break;
  }
  return hits;
}

/** Search raw JSONL events (client-side fallback when gateway is not restarted). */
export function searchEventsConversation(events: unknown[], query: string, maxHits = 8): SessionContentHit[] {
  const needle = query.trim().replace(/\s+/g, " ").toLocaleLowerCase();
  if (!needle) return [];
  const hits: SessionContentHit[] = [];
  let turn = 0;
  const seen = new Set<string>();
  for (const raw of events) {
    if (!raw || typeof raw !== "object") continue;
    const e = raw as Record<string, unknown>;
    const type = String(e.type || "");
    if (type === "user") {
      if (e.automated === true) continue;
      const content = String(e.content || "");
      const isSteer = isSteerUserEvent(e, content);
      let clean = "";
      if (!isSteer) {
        clean = stripWindowPreamble(content).trim();
        if (!clean) continue;
        turn += 1;
      } else {
        clean = cleanSteerContent(content);
        if (!clean) continue;
      }
      const key = `${Math.max(1, turn)}:user`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "user", snippet: snippetAround(clean, needle) });
      }
    } else if (type === "assistant") {
      const clean = String(e.content || "").trim();
      if (!clean) continue;
      const key = `${Math.max(1, turn)}:assistant`;
      if (!seen.has(key) && clean.replace(/\s+/g, " ").toLocaleLowerCase().includes(needle)) {
        seen.add(key);
        hits.push({ turn: Math.max(1, turn), role: "assistant", snippet: snippetAround(clean, needle) });
      }
    }
    if (hits.length >= maxHits) break;
  }
  return hits;
}

export function collapseSessionRows<T extends SessionRow>(
  rows: T[],
  currentId: string | null,
  expanded: boolean,
  limit = COLLAPSED_SESSION_LIMIT,
): { visible: T[]; hiddenCount: number } {
  if (expanded) return { visible: rows, hiddenCount: 0 };
  let idleCount = 0;
  const visible = rows.filter((row) => {
    if (row.id === currentId || row.session.running || row.session.pinned) return true;
    idleCount += 1;
    return idleCount <= limit;
  });
  return { visible, hiddenCount: rows.length - visible.length };
}

/** Case-insensitive `[start, end)` match ranges of `query` inside `text`. */
export function findQueryRanges(text: string, query: string): Array<[number, number]> {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle || !text) return [];
  const hay = text.toLocaleLowerCase();
  const ranges: Array<[number, number]> = [];
  let pos = 0;
  while (pos <= hay.length - needle.length) {
    const at = hay.indexOf(needle, pos);
    if (at < 0) break;
    ranges.push([at, at + needle.length]);
    pos = at + needle.length;
  }
  return ranges;
}

/**
 * Highlight `query` matches inside settled `.m .bub` message bodies in `root`.
 * Restores any previous `<mark data-chat-search-mark>` wrappers first so React's
 * text nodes stay normalized, then wraps matching spans and returns the first
 * active mark in the focused turn/role (or the first mark overall).
 */
export function applyChatSearchHighlights(
  root: HTMLElement,
  query: string,
  focusTurn?: number,
  focusRole?: "user" | "assistant",
): HTMLElement | null {
  const doc = root.ownerDocument;
  const existing = root.querySelectorAll("mark[data-chat-search-mark]");
  const touchedParents = new Set<Node>();
  for (const mark of existing) {
    const parent = mark.parentNode;
    if (!parent) continue;
    parent.replaceChild(doc.createTextNode(mark.textContent || ""), mark);
    touchedParents.add(parent);
  }
  for (const parent of touchedParents) parent.normalize();

  const needle = query.trim();
  if (!needle) return null;

  let firstActiveMark: HTMLElement | null = null;
  let firstAnyMark: HTMLElement | null = null;
  const msgEls = root.querySelectorAll<HTMLElement>(".m:not(.streaming)[data-turn]");
  for (const msgEl of msgEls) {
    const turn = Number(msgEl.dataset.turn || 0);
    const role = msgEl.dataset.role as "user" | "assistant" | undefined;
    const isFocusMsg = focusTurn != null && turn === focusTurn && (!focusRole || role === focusRole);
    const bubbles = msgEl.querySelectorAll<HTMLElement>(".bub");
    for (const bub of bubbles) {
      const walker = doc.createTreeWalker(bub, NodeFilter.SHOW_TEXT);
      const textNodes: Text[] = [];
      for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const el = n.parentElement;
        if (!el || el.closest(".code-frame-bar, .line-num, button, script, style")) continue;
        if (n.nodeValue && findQueryRanges(n.nodeValue, needle).length > 0) {
          textNodes.push(n as Text);
        }
      }
      for (const textNode of textNodes) {
        const raw = textNode.nodeValue || "";
        const ranges = findQueryRanges(raw, needle);
        if (!ranges.length || !textNode.parentNode) continue;
        const frag = doc.createDocumentFragment();
        let cursor = 0;
        for (const [start, end] of ranges) {
          if (start > cursor) frag.appendChild(doc.createTextNode(raw.slice(cursor, start)));
          const mark = doc.createElement("mark");
          mark.className = isFocusMsg ? "chat-hit-mark active" : "chat-hit-mark";
          mark.setAttribute("data-chat-search-mark", "1");
          mark.textContent = raw.slice(start, end);
          frag.appendChild(mark);
          if (!firstAnyMark) firstAnyMark = mark;
          if (isFocusMsg && !firstActiveMark) firstActiveMark = mark;
          cursor = end;
        }
        if (cursor < raw.length) frag.appendChild(doc.createTextNode(raw.slice(cursor)));
        textNode.parentNode.replaceChild(frag, textNode);
      }
    }
  }
  return firstActiveMark || firstAnyMark;
}
